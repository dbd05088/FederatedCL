import logging
import os
import copy
import math
import time
import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from flops_counter.ptflops import get_model_complexity_info
from utils.data_loader import ImageDataset, cutmix_data, MultiProcessLoader, get_statistics
from utils.augment import get_transform
from utils.train_utils import select_model, select_optimizer, select_scheduler, get_llavamodel
from utils.block_utils import MODEL_BLOCK_DICT, get_blockwise_flops
from peft.tuners.lora import LoraLayer

# logger = logging.getLogger()
writer = SummaryWriter("tensorboard")

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    # logger,
)
import numpy as np

from transformers.optimization import get_scheduler
from utils.train_utils import get_llavamodel
from peft import get_peft_model
from collections import OrderedDict
from utils.data_worker import ManagerWatchdog
import queue
from collections.abc import Mapping

OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"



class CLManagerClient: # Client
    def __init__(
        self,
        device,
        data_args,
        model_args,
        model = None,
        args = None,
        bnb_model_from_pretrained_args=None,
        tokenizer = None,
        receive_channel=None,
        send_channel=None,
        logger=None
    ):
        # super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

    # def __init__(self, train_datalist, test_datalist, device, model_args, training_args, bnb_model_from_pretrained_args, **kwargs):
        kwargs = vars(args)
        self.args = args
        self.data_args = data_args
        self.model_args = model_args
        self.bnb_model_from_pretrained_args=bnb_model_from_pretrained_args
        self.device = device
        # model = model.to(args.device)
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.task_id = 0
        self.method_name = kwargs["mode"]
        self.dataset = kwargs["dataset"]
        # self.samples_per_task = kwargs["samples_per_task"]
        self.memory_size = kwargs["memory_size"]
        self.online_iter = kwargs["online_iter"]

        self.send_channel=send_channel
        self.receive_channel=receive_channel

        self.lr = kwargs["learning_rate"]
        # self.block_names = MODEL_BLOCK_DICT[self.model_name]
        # self.num_blocks = len(self.block_names) - 1

        assert kwargs["temp_batchsize"] <= kwargs["per_gpu_train_batch_size"]
        self.batch_size = kwargs["per_gpu_train_batch_size"]
        self.temp_batch_size = kwargs["temp_batchsize"]
        self.memory_batch_size = self.batch_size - self.temp_batch_size
        self.memory_size -= self.temp_batch_size
        self.transforms = kwargs["transforms"]

        # self.data_dir = kwargs["data_dir"]
        # if self.data_dir is None:
        self.data_dir = os.path.join("dataset", self.dataset, 'images')
        self.n_worker = kwargs["dataloader_num_workers"]
        self.future_steps = kwargs["future_steps"]
        # self.transform_on_gpu = kwargs["transform_on_gpu"]
        # self.use_kornia = kwargs["use_kornia"]
        # self.transform_on_worker = kwargs["transform_on_worker"]

        self.eval_period = kwargs["eval_period"]
        self.topk = kwargs["topk"]
        self.f_period = kwargs["f_period"]

        self.logger = logger
        
        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.sample_num = 0
        self.train_count = 0
        self.seen = 0

        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.note = kwargs['note']
        self.rnd_seed = kwargs['seed']
        self.save_path = f'results/{self.dataset}/{self.note}/seed_{self.rnd_seed}'
        self.f_next_time = 0
        self.start_time = time.time()
        # num_samples = {'cifar10': 50000, 'cifar100': 50000, 'clear10':30000, 'clear100':100000, 'tinyimagenet': 100000, 'imagenet': 1281167}
        # self.total_samples = num_samples[self.dataset]

        self.exposed_domains = []
        self.waiting_batch = []
        # self.get_flops_parameter()
        # self.init_training()

        # self.initialize_future()
        self.total_flops = 0.0
        # self.writer = SummaryWriter(f'tensorboard/{self.dataset}/{self.note}/seed_{self.rnd_seed}')
        self.state = {}

        self.watchdog = ManagerWatchdog()
        
    def setup(self):
        model, tokenizer, data_args = get_llavamodel(self.model_args, self.args, self.bnb_model_from_pretrained_args, self.data_args)
        self.model = model
        self.tokenizer = tokenizer
        self.data_args = data_args

        max_steps = 10000 # FIXME
        self.create_optimizer()
        self.create_scheduler(max_steps, optimizer=self.optimizer)

        # Activate gradient checkpointing if needed
        if self.args.gradient_checkpointing: # default False
            if self.args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = self.args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
    
    # from llava_traininer
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        self.logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        self.logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                self.logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """
        if self.lr_scheduler is None:
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer=self.optimizer if optimizer is None else optimizer,
                num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return
        map_location = self.args.device
        if os.path.isfile(os.path.join(checkpoint, f"{self.state['client_id']}_client_{OPTIMIZER_NAME}")):
            self.optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint, f"{self.state['client_id']}_client_{OPTIMIZER_NAME}"), map_location=map_location)
            )
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, f"{self.state['client_id']}_client_{SCHEDULER_NAME}")))

    def save_model(self, client_id, output_dir):
        state_dict = OrderedDict()
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer) or 'vision_tower' in name or 'mm_projector' in name:
                state_dict.update(module.state_dict())
        torch.save(state_dict, os.path.join(output_dir, f"{client_id}_client_model.pth"))
    
    def load_model(self, client_id, output_dir):
        state_dict = torch.load(os.path.join(output_dir, f"{client_id}_client_model.pth"), map_location=self.args.device)
        self.model.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def init_model(self):
        # reinit vision tower & mm_projector of llava model
        self.model.load_state_dict(torch.load('./llava_vision_tower_mm_projector.pth', map_location='cpu'), strict=False)
        self.logger.info("done loading init llava vision tower and mm projector")
        # reset lora layers
        # model = self.model.unload()
        # self.model = get_peft_model(model, self.args.lora_config)
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                module.reset_lora_parameters('default', True)
        self.logger.info("done reset lora layers")

    def is_new(self, client_id):
        return not os.path.exists(os.path.join(self.args.state_dir, f'{client_id}_client_trainerstate.npy'))

    def switch_state(self, client_id, train_datalist):
        if self.is_new(client_id):
            # model, tokenizer, _ = get_llavamodel(training_args=self.args, model_args=self.model_args, bnb_model_from_pretrained_args={}, lora_config=self.args.lora_config)
            # print(client_id)
            self.init_state(client_id, len(train_datalist))
            self.init_model()
            self.initialize_future(train_datalist)
        else: # load_state
            if client_id != self.state['client_id']:
                self.logger.info(f"load client {client_id}")
                trainer_state = np.load(os.path.join(self.args.state_dir, '{}_client_trainerstate.npy'.format(client_id))).item()
                self.state['client_id'] = trainer_state['client_id']
                self.state['sample_cnt'] = trainer_state['sample_cnt']
                self.state['round_cnt'] = trainer_state['round_cnt']
                self.state['done'] = trainer_state['done']
                self.temp_future_batch = trainer_state['temp_future_batch']
                self.waiting_batch = trainer_state['waiting_batch']
                self.temp_batch = trainer_state['temp_batch']

                self.memory.load_state(client_id)
                self.dataloader.load_state(client_id, self.args.state_dir)
                self.load_model(client_id, self.args.state_dir)
    
    def init_state(self, cid, data_len):
        self.state['client_id'] = cid
        self.state['sample_cnt'] = 0
        self.state['round_cnt'] = 0
        self.state['done'] = False
        self.state['total_samples'] = data_len
    
    def save_state(self):
        trainer_state = self.state
        trainer_state['temp_future_batch'] = self.temp_future_batch
        trainer_state['waiting_batch'] = self.waiting_batch
        trainer_state['temp_batch'] = self.temp_batch

        np.save(os.path.join(self.args.state_dir, '{}_client_trainerstate.npy'.format(self.state['client_id'])), trainer_state)
        self.memory.save_state(self.state['client_id'], self.args.state_dir)
        self.dataloader.save_state(self.state['client_id'], self.args.state_dir)

        torch.save(self.optimizer.state_dict(), os.path.join(self.args.state_dir, f"{self.state['client_id']}_client_{OPTIMIZER_NAME}"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(self.args.state_dir, f"{self.state['client_id']}_client_{SCHEDULER_NAME}"))

        self.save_model(self.state['client_id'], self.args.state_dir)

    def run(self):
        while True:#self.watchdog.is_alive():
            try:
                r = self.receive_channel.get()
            except queue.Empty:
                continue
            
            if r:
                if type(r) == str and r == "done":
                    break
                else:
                    client_id = r['client_id']
                    curr_round = r['curr_round']
                    train_datalist = r['train_datalist']
                    test_datalist = r['test_datalist']

                    self.train_one_round(client_id, curr_round, train_datalist, test_datalist)

                    self.send_channel.put(f"done {client_id}")

    def train_one_round(self, client_id, curr_round, train_datalist, test_datalist):
        ######################################
        self.switch_state(client_id, train_datalist)
        ######################################
        self.state['round_cnt'] += 1
        self.state['curr_round'] = curr_round
        
        # FIXME
        samples_per_round = 10

        seen_so_far = self.state['sample_cnt']
        
        for i, data in enumerate(train_datalist[seen_so_far:seen_so_far+samples_per_round]):
            # explicit task boundary for twf
            # if samples_cnt % training_args.samples_per_task == 0 and training_args.mode in ["bic", "xder", "der_lider", "er_lider", "xder_lider", "co2l", "trire"]:
            #     method.online_before_task(task_id)
            #     task_id += 1

            self.state['sample_cnt'] += 1
            self.online_step(data, self.state['sample_cnt'], self.args.dataloader_num_workers)
            # if self.state['sample_cnt'] % self.args.eval_period == 0:
        

            #     eval_dict = self.online_evaluate(test_datalist, self.state['sample_cnt'], 512, self.args.dataloader_num_workers, cls_dict,
            #                                     cls_addition, data["time"])
            #     eval_results["test_acc"].append(eval_dict['avg_acc'])
            #     eval_results["percls_acc"].append(eval_dict['cls_acc'])
            #     eval_results["data_cnt"].append(samples_cnt)
            
            # if (training_args.mode in ["remind"]) and samples_cnt == training_args.baseinit_samples:
            #     method.finalize_baseinit()
            
            # if samples_cnt % training_args.samples_per_task == 0 and (training_args.mode in ["memo", "xder", "afec", "sparcl", "trire"]) and samples_cnt != num_samples[args.dataset]:
            #     method.online_after_task()
            
        # if eval_results["data_cnt"][-1] != samples_cnt:
        #     eval_dict = method.online_evaluate(test_datalist, samples_cnt, 512, training_args.dataloader_num_workers, cls_dict, cls_addition,
        #                                     data["time"])

        # A_last = eval_dict['avg_acc']
        self.save_state()
        

    # Memory 새로 정의 (not MemoryBase)
    def initialize_future(self, train_datalist):
        # print('start_init_future')
        self.data_stream = iter(train_datalist)
        self.dataloader = MultiProcessLoader(self.n_worker, self.data_dir, self.device, tokenizer=self.tokenizer, data_args=self.data_args)
        self.memory = MemoryBase(self.memory_size)

        # self.memory_list = []
        self.temp_batch = []
        self.temp_future_batch = []
        self.num_updates = 0
        self.future_num_updates = 0
        self.train_count = 0
        self.seen = 0
        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.waiting_batch = []
        # 미리 future step만큼의 batch를 load
        for i in range(self.future_steps):
            self.load_batch()

    def memory_future_step(self):
        try:
            sample = next(self.data_stream)
        except:
            return 1
            
        # if sample["time"] not in self.exposed_domains and "clear" in self.dataset:
        #     self.exposed_domains.append(sample["time"])

        self.temp_future_batch.append(sample)
        self.future_num_updates += self.online_iter

        if len(self.temp_future_batch) >= self.temp_batch_size:
            self.generate_waiting_batch(int(self.future_num_updates))
            for stored_sample in self.temp_future_batch:
                self.update_memory(stored_sample)
            self.temp_future_batch = []
            self.future_num_updates -= int(self.future_num_updates)
        self.future_sample_num += 1
        return 0

    def update_memory(self, sample):
        self.reservoir_memory(sample)

    def reservoir_memory(self, sample):
        self.seen += 1
        if len(self.memory.images) >= self.memory_size:
            j = np.random.randint(0, self.seen)
            if j < self.memory_size:
                self.memory.replace_sample(sample, j)
        else:
            self.memory.replace_sample(sample)

    # loader로부터 load된 batch를 받아오는 것
    def get_batch(self):
        batch = self.dataloader.get_batch()
        self.load_batch()
        return batch

    # stream 또는 memory를 활용해서 batch를 load해라
    # data loader에 batch를 전달해주는 함수
    def load_batch(self):
        stream_end = False
        while len(self.waiting_batch) == 0:
            stream_end = self.memory_future_step()
            if stream_end:
                break
        if not stream_end:
            self.dataloader.load_batch(self.waiting_batch[0])
            del self.waiting_batch[0]

    def generate_waiting_batch(self, iterations):
        for i in range(iterations):
            self.waiting_batch.append(self.temp_future_batch + self.memory.retrieval(self.memory_batch_size))

    def online_step(self, sample, sample_num, n_worker):
        self.sample_num = sample_num
        self.temp_batch.append(sample)
        self.num_updates += self.online_iter
        if len(self.temp_batch) >= self.temp_batch_size:
            if int(self.num_updates) > 0:
                train_loss = self.online_train(iterations=int(self.num_updates))
                self.report_training(sample_num, train_loss)
                self.num_updates -= int(self.num_updates)
            self.temp_batch = []

    def _prepare_input(self, data):
        """
        Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.args.device}
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError(
                "The batch received was empty, your model won't be able to train on it. Double-check that your "
                "training dataset contains keys expected by the model"
            )
        return inputs
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(**inputs)
        # Save past state if it exists
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def online_train(self, iterations=1):
        # print("start online train")
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            # x = data["image"].to(self.device)
            # y = data["label"].to(self.device)
            self.before_model_update()

            self.optimizer.zero_grad()

            data = self._prepare_inputs(data)

            loss = self.compute_loss(self.model, data)

            # _, preds = logit.topk(self.topk, 1, True, True)
            loss.backward()
            self.optimizer.step()

            # if self.use_amp:
            #     self.scaler.scale(loss).backward()
            #     self.scaler.step(self.optimizer)
            #     self.scaler.update()
            # else:
            #     loss.backward()
            #     self.optimizer.step()

            # self.total_flops += (len(y) * self.backward_flops)
            # self.current_flos += float(self.floating_point_ops(data))

            self.after_model_update()

            total_loss += loss.item()
            # correct += torch.sum(preds == y.unsqueeze(1)).item()
            # num_data += y.size(0)

        return total_loss / iterations

    def before_model_update(self):
        pass

    def after_model_update(self):
        # self.update_schedule()
        self.lr_scheduler.step()

    def report_training(self, sample_num, train_loss):
        writer.add_scalar(f"train/loss", train_loss, sample_num)
        # writer.add_scalar(f"train/acc", train_acc, sample_num)
        # self.logger.info(
        self.logger.info(
            f"Client {self.state['client_id']} Train | Sample # {sample_num} | train_loss {train_loss:.4f} |"# TFLOPs {self.total_flops/1000:.2f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.state['total_samples']-sample_num) / sample_num))}"
        )

    def report_test(self, sample_num, avg_loss, avg_acc):
        writer.add_scalar(f"test/loss", avg_loss, sample_num)
        writer.add_scalar(f"test/acc", avg_acc, sample_num)
        self.logger.info(
            f"Test | Sample # {sample_num} | test_loss {avg_loss:.4f} | test_acc {avg_acc:.4f} | TFLOPs {self.total_flops/1000:.2f}"
        )

    def update_schedule(self, reset=False):
        if reset:
            self.scheduler = select_scheduler(self.sched_name, self.optimizer)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
        else:
            self.scheduler.step()


    def online_evaluate(self, test_list, sample_num, batch_size, n_worker, cls_dict, cls_addition, data_time):
        test_df = pd.DataFrame(test_list)
        '''
        if "clear" in self.dataset:
            exp_test_df = test_df[test_df['time'] < self.exposed_domains[-1]]
            if len(self.exposed_domains) == 0:
                exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        else:
            exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        '''
        exp_test_df = test_df[test_df['klass'].isin(self.exposed_classes)]
        
        print("exposed_domains", self.exposed_domains)
        print("exposed_classes", self.exposed_classes)
        print("exp_test_df", len(exp_test_df))
        test_dataset = ImageDataset(
            exp_test_df,
            dataset=self.dataset,
            transform=self.test_transform,
            cls_list=self.exposed_classes,
            data_dir=self.data_dir
        )
        test_loader = DataLoader(
            test_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=n_worker,
        )
        eval_dict = self.evaluation(test_loader, self.criterion)
        
        self.report_test(sample_num, eval_dict["avg_loss"], eval_dict["avg_acc"])

        # if sample_num >= self.f_next_time:
        #     self.get_forgetting(sample_num, test_list, cls_dict, batch_size, n_worker)
        #     self.f_next_time += self.f_period
        return eval_dict


    def evaluation(self, test_loader, criterion):
        total_correct, total_num_data, total_loss = 0.0, 0.0, 0.0
        correct_l = torch.zeros(self.n_classes)
        num_data_l = torch.zeros(self.n_classes)
        label = []

        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                x = data["image"]
                y = data["label"]
                x = x.to(self.device)
                y = y.to(self.device)
                logit = self.model(x)

                loss = criterion(logit, y)
                pred = torch.argmax(logit, dim=-1)
                _, preds = logit.topk(self.topk, 1, True, True)

                total_correct += torch.sum(preds == y.unsqueeze(1)).item()
                total_num_data += y.size(0)

                xlabel_cnt, correct_xlabel_cnt = self._interpret_pred(y, pred)
                correct_l += correct_xlabel_cnt.detach().cpu()
                num_data_l += xlabel_cnt.detach().cpu()

                total_loss += loss.item()
                label += y.tolist()

        avg_acc = total_correct / total_num_data
        avg_loss = total_loss / len(test_loader)
        cls_acc = (correct_l / (num_data_l + 1e-5)).numpy().tolist()
        ret = {"avg_loss": avg_loss, "avg_acc": avg_acc, "cls_acc": cls_acc}

        return ret

    def get_flops_parameter(self, method=None):
        _, _, _, inp_size, inp_channel = get_statistics(dataset=self.dataset)
        if self.model_name == 'vit':
            inp_size = 224
        
        self.flops_dict = get_model_complexity_info(self.model, (inp_channel, inp_size, inp_size),
                                                                             as_strings=False,
                                                                             print_per_layer_stat=False, verbose=True,
                                                                             criterion=self.criterion,
                                                                             original_opt=self.optimizer,
                                                                    opt_name=self.opt_name, lr=self.lr)
        forward_flops, backward_flops, G_forward_flops, G_backward_flops, F_forward_flops, F_backward_flops  = get_blockwise_flops(self.flops_dict, self.model_name, method)
        self.forward_flops = sum(forward_flops)
        self.backward_flops = sum(backward_flops)
        self.blockwise_forward_flops = forward_flops
        self.blockwise_backward_flops = backward_flops
        self.total_model_flops = self.forward_flops + self.backward_flops
        
        self.G_forward_flops, self.G_backward_flops = sum(G_forward_flops), sum(G_backward_flops)
        self.F_forward_flops, self.F_backward_flops = sum(F_forward_flops), sum(F_backward_flops)
        self.G_blockwise_forward_flops, self.G_blockwise_backward_flops = G_forward_flops, G_backward_flops
        self.F_blockwise_forward_flops, self.F_blockwise_backward_flops = F_forward_flops, F_backward_flops
        
    
class MemoryBase:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.images = []
        self.labels = []

        self.update_buffer = ()
        # self.cls_dict = dict()
        # self.cls_list = []
        # self.cls_count = []
        # self.cls_idx = []
        
        self.usage_count = np.array([])
        # self.class_usage_count = np.array([])
        self.current_images = []
        self.current_labels = []
        # self.current_cls_count = [0 for _ in self.cls_list]
        # self.current_cls_idx = [[] for _ in self.cls_list]

    def __len__(self):
        return len(self.images)

    def replace_sample(self, sample, idx=None):
        # self.cls_count[self.cls_dict[sample['klass']]] += 1
        if idx is None:
            assert len(self.images) < self.memory_size
            # self.cls_idx[self.cls_dict[sample['klass']]].append(len(self.images))
            self.images.append(sample)
            # self.labels.append(self.cls_dict[sample['klass']])
        else:
            assert idx < self.memory_size
            # self.cls_count[self.labels[idx]] -= 1
            # self.cls_idx[self.labels[idx]].remove(idx)
            self.images[idx] = sample
            # self.labels[idx] = self.cls_dict[sample['klass']]
            # self.cls_idx[self.cls_dict[sample['klass']]].append(idx)

    def retrieval(self, size, return_index=False):
        sample_size = min(size, len(self.images))
        memory_batch = []
        indices = np.random.choice(range(len(self.images)), size=sample_size, replace=False)
        for i in indices:
            memory_batch.append(self.images[i])
        if return_index:
            return memory_batch, indices
        else:
            return memory_batch


    def save_state(self, client_id, save_dir):
        path = os.path.join(save_dir, f"{client_id}_client_memory.npy")
        mem_state = {
            'images':self.images,
            'labels':self.labels
        }
        np.save(path, mem_state)

    def load_state(self, client_id, load_dir):
        path = os.path.join(load_dir, f"{client_id}_client_memory.npy")
        mem_state = np.load(path).item()
        self.images = mem_state['images']
        self.labels = mem_state['labels']

def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

from transformers.utils import is_peft_available
import importlib.metadata
from packaging import version

if is_peft_available():
    from peft import PeftModel

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False