import os
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from utils.data_loader import MultiProcessLoader
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset
from utils.train_utils import  get_VLMmodel
from peft.tuners.lora import LoraLayer
import bitsandbytes

# writer = SummaryWriter("tensorboard")

from transformers import Trainer
from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)
import numpy as np

from transformers.optimization import get_scheduler
from collections import OrderedDict
from utils.data_worker import ManagerWatchdog
import queue
from collections.abc import Mapping

from utils.eval_metrics import NLPEvaluator, matching_token_num
from models.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX

OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"

class CLManagerClient: # Client
    def __init__(
        self,
        rank,
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
        
        kwargs = vars(args)
        self.rank = rank
        self.args = args
        self.data_args = data_args
        self.model_args = model_args
        self.bnb_model_from_pretrained_args=bnb_model_from_pretrained_args
        self.device = device
        
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.task_id = 0
        self.method_name = kwargs["mode"]
        self.memory_size = kwargs["memory_size"]
        self.online_iter = kwargs["online_iter"]

        self.send_channel=send_channel
        self.receive_channel=receive_channel

        self.lr = kwargs["learning_rate"]

        assert kwargs["temp_batchsize"] <= kwargs["per_gpu_train_batch_size"]
        self.batch_size = kwargs["per_gpu_train_batch_size"]
        self.temp_batch_size = kwargs["temp_batchsize"]
        self.memory_batch_size = self.batch_size - self.temp_batch_size
        self.memory_size -= self.temp_batch_size
        self.transforms = kwargs["transforms"]

        self.n_worker = kwargs["dataloader_num_workers"]
        self.future_steps = kwargs["future_steps"]

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
        
        self.f_next_time = 0
        self.start_time = time.time()

        self.exposed_domains = []
        self.waiting_batch = []

        self.total_flops = 0.0
        self.state = {}
        
        self.logger = None

        self.watchdog = ManagerWatchdog()
        
        self.gradient_accumulation_steps = kwargs['gradient_accumulation_steps']
        self.gradient_checkpointing = kwargs['gradient_checkpointing']
        
        # 576 for clip image encoder (llava)
        # 729 for siglip (bunny)
        if 'llava' in self.model_args.model_name_or_path.lower():
            self.img_feat_size = 576
        elif 'bunny' in self.model_args.model_name_or_path.lower():
            self.img_feat_size = 729
        
    def setup(self):
        model, tokenizer, data_args = get_VLMmodel(self.model_args, self.args, self.bnb_model_from_pretrained_args, self.data_args)
        self.model = model
        self.tokenizer = tokenizer
        self.data_args = data_args

        # max_steps = 8000 # FIXME
        self.create_optimizer()
        # self.create_scheduler(max_steps, optimizer=self.optimizer)

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
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" or "vision_tower" in name]
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
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        print(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        self.logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                print(f"skipped: {skipped/2**20}M params")

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
            # self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, f"{self.state['client_id']}_client_{SCHEDULER_NAME}")))

    def save_model(self, client_id, output_dir):
        state_dict = OrderedDict()
        with torch.no_grad():
            for name, parameters in self.model.named_parameters():
                if isinstance(parameters, torch.Tensor) and parameters.requires_grad:
                        state_dict[name] = parameters.detach().cpu()
        torch.save(state_dict, os.path.join(output_dir, f"{client_id}_client_model.pth"))
    
    def load_model(self, client_id, output_dir):
        state_dict = torch.load(os.path.join(output_dir, f"{client_id}_client_model.pth"), map_location=self.args.device)
        self.model.load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def init_model(self):
        # reinit vision tower & mm_projector of llava model
        if 'bunny' in self.model_args.model_name_or_path.lower():
            self.model.load_state_dict(torch.load('./bunny_vision_tower_mm_projector.pth', map_location='cpu'), strict=False)
            self.logger.write("done loading init bunny vision tower and mm projector\n")
        elif 'llava' in self.model_args.model_name_or_path.lower():
            self.model.load_state_dict(torch.load('./llava_vision_tower_mm_projector.pth', map_location='cpu'), strict=False)
            self.logger.write("done loading init llava vision tower and mm projector\n")
        else:
            raise ValueError("wrong model name")
        # reset lora layers
        for name, module in self.model.named_modules():
            if isinstance(module, LoraLayer):
                module.reset_lora_parameters('default', True)
        self.logger.write("done reset lora layers\n")

    def is_new(self, client_id):
        return not os.path.exists(os.path.join(self.args.state_dir, f'{client_id}_client_trainerstate.npy'))

    def switch_state(self, client_id, train_datalist):
        self.logger = open(f'./results/{self.method_name}/{self.note}/{client_id}_client.log', 'a')
        if self.is_new(client_id):
            self.init_state(client_id, len(train_datalist))
            self.init_model()
            self.data_stream = iter(train_datalist)
            self.initialize_future(train_datalist)
        else: # load_state
            # if client_id != self.state['client_id']:
            self.logger.write(f"load client {client_id} to rank {self.rank}\n")
            trainer_state = np.load(os.path.join(self.args.state_dir, '{}_client_trainerstate.npy'.format(client_id)), allow_pickle=True).item()
            self.state['client_id'] = trainer_state['client_id']
            self.state['sample_cnt'] = trainer_state['sample_cnt']
            self.state['round_cnt'] = trainer_state['round_cnt']
            self.state['done'] = trainer_state['done']
            self.temp_future_batch = trainer_state['temp_future_batch']
            self.waiting_batch = trainer_state['waiting_batch']
            self.temp_batch = trainer_state['temp_batch']
            self.future_sample_num = trainer_state['future_sample_num']
            self.data_stream = iter(train_datalist[self.future_sample_num:])

            self.memory.load_state(client_id, self.args.state_dir)
            self.dataloader.load_state(client_id, self.args.state_dir)
            self.load_model(client_id, self.args.state_dir)
            self._load_optimizer_and_scheduler(self.args.state_dir)
    
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
        trainer_state['future_sample_num'] = self.future_sample_num

        np.save(os.path.join(self.args.state_dir, '{}_client_trainerstate.npy'.format(self.state['client_id'])), trainer_state)
        self.memory.save_state(self.state['client_id'], self.args.state_dir)
        self.dataloader.save_state(self.state['client_id'], self.args.state_dir)

        torch.save(self.optimizer.state_dict(), os.path.join(self.args.state_dir, f"{self.state['client_id']}_client_{OPTIMIZER_NAME}"))
        # torch.save(self.lr_scheduler.state_dict(), os.path.join(self.args.state_dir, f"{self.state['client_id']}_client_{SCHEDULER_NAME}"))

        self.save_model(self.state['client_id'], self.args.state_dir)
        
        self.logger.close()

    def run(self):
        while True:
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
                    server_msg = r['server_msg']
                    
                    ######################################
                    self.switch_state(client_id, train_datalist)
                    self.handle_server_msg(server_msg)
                    ######################################

                    self.train_one_round(curr_round, train_datalist, test_datalist)

                    self.send_channel.put(self.client_msg())
    
    def client_msg(self):
        return f"done {self.state['client_id']}"

    def handle_server_msg(self, server_msg):
        pass

    def train_one_round(self, curr_round, train_datalist, test_datalist):
        self.state['round_cnt'] += 1
        self.state['curr_round'] = curr_round
        
        # FIXME
        samples_per_round = 1000

        seen_so_far = self.state['sample_cnt']
        
        if self.gradient_checkpointing:
            gradient_checkpointing_kwargs = {'use_reentrant':False}
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        
        self.optimizer.zero_grad()
        for i, data in enumerate(train_datalist[seen_so_far:seen_so_far+samples_per_round]):
            self.state['sample_cnt'] += 1
            self.online_step(data, self.state['sample_cnt'], self.args.dataloader_num_workers)
            if self.state['sample_cnt'] % self.eval_period == 0:
                for dataname, datalist in test_datalist.items():
                    self.evaluate(dataname, datalist)
        self.save_state()

    # Memory 새로 정의 (not MemoryBase)
    def initialize_future(self, train_datalist):
        self.dataloader = MultiProcessLoader(self.n_worker, self.device, tokenizer=self.tokenizer, data_args=self.data_args)
        self.memory = MemoryBase(self.memory_size)

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
        outputs = model(**inputs)
        # Save past state if it exists
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def online_train(self, iterations=1):
        total_loss, correct, num_data = 0.0, 0.0, 0.0

        for i in range(iterations):
            self.model.train()
            data = self.get_batch()
            self.before_model_update()

            data = self._prepare_inputs(data)
            loss = self.compute_loss(self.model, data)
            loss.backward()
            
            if (self.state['sample_cnt']) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                # self.lr_scheduler.step()
                self.optimizer.zero_grad()

            # self.after_model_update()

            total_loss += loss.item()
        return total_loss / iterations

    def before_model_update(self):
        pass

    # def after_model_update(self):
    #     # self.update_schedule()
    #     self.lr_scheduler.step()

    def report_training(self, sample_num, train_loss):
        # writer.add_scalar(f"train/loss", train_loss, sample_num)
        # if sample_num % 5 == 0:
        self.logger.write(
            f"Client {self.state['client_id']} Train | Sample # {sample_num} | train_loss {train_loss:.4f} |"# TFLOPs {self.total_flops/1000:.2f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.state['total_samples']-sample_num) / sample_num))}"
        )
        self.logger.write('\n')

    def report_test(self, sample_num, scores, dataname):
        # writer.add_scalar(f"test/loss_client{self.state['client_id']}", scores["loss"], sample_num)
        # writer.add_scalar(f"test/precision_client{self.state['client_id']}", scores["precision"], sample_num)
        # writer.add_scalar(f"test/Bleu_client{self.state['client_id']}", scores["Bleu_1"], sample_num)
        # writer.add_scalar(f"test/METEOR_client{self.state['client_id']}", scores["METEOR"], sample_num)
        # writer.add_scalar(f"test/RogueL_client{self.state['client_id']}", scores["ROUGE_L"], sample_num)
        # writer.add_scalar(f"test/CIDEr_client{self.state['client_id']}", scores["CIDEr"], sample_num)
        self.logger.write(
            f"Test (Client id {self.state['client_id']}) | Sample # {sample_num} | Data {dataname} | test_loss {scores['loss']:.4f} | precision {scores['precision']:.4f} | Bleu_1 {scores['Bleu_1']} | Bleu_2 {scores['Bleu_2']} | Bleu_3 {scores['Bleu_3']} |Bleu_4 {scores['Bleu_4']} | METEOR {scores['METEOR']} | ROUGE_L {scores['ROUGE_L']} | CIDEr {scores['CIDEr']} |"
        )
        self.logger.write('\n')

    def evaluate(self, dataname, test_datalist):
        self.logger.write(f"client {self.state['client_id']} evaluate {dataname}\n")
        dataset = LazySupervisedDataset(test_datalist, self.tokenizer, self.data_args, preprocess=False)
        dataloader = DataLoader(dataset, batch_size= 8, num_workers=self.n_worker, collate_fn=DataCollatorForSupervisedDataset(tokenizer=self.tokenizer))
        
        self.model.eval()
        predictions = []
        total_loss = 0
        n_word_total = 0
        n_word_correct = 0
        cnt = 0
        with torch.no_grad():
            for i, batch in enumerate((dataloader)):
                # * prepare data
                inputs = batch['input_ids']
                input_labels = batch['labels']
                batch = self._prepare_inputs(batch)
                
                output = self.model(**batch)
                loss = output[0]
                pred_scores_list = output[1]
                n_correct = 0
                n_word = 0
                for inp, pred, gold in zip(inputs, pred_scores_list, input_labels):
                    valid_label_mask = gold.ne(IGNORE_INDEX)
                    valid_idx = valid_label_mask.nonzero()[0].item()
                    
                    n_word += len(torch.unique(gold[valid_label_mask]))#.sum()
                    pred_id = torch.argmax(pred, dim=1).cpu()#.to(device)
                    
                    # image token index
                    img_token_index = (inp==IMAGE_TOKEN_INDEX).nonzero()[0].item()
                    pred_id = torch.cat((pred_id[:img_token_index], torch.tensor([IMAGE_TOKEN_INDEX]), pred_id[img_token_index+self.img_feat_size:]))
                    
                    n_correct += matching_token_num(pred_id, gold, valid_idx, valid_label_mask)
                    
                    gold[valid_label_mask == False] = 0
                    
                    pred_sentence = self.tokenizer.decode(pred_id[valid_idx:], skip_special_tokens=True)#[valid_label_mask])
                    gold_sentence = self.tokenizer.decode(gold[valid_label_mask], skip_special_tokens=True)#[])
                    predictions.append({"sentence":pred_sentence, "gt_sentence":gold_sentence})

                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss
                cnt += 1
        scores = NLPEvaluator(predictions).evaluate()
        scores["precision"] = n_word_correct / n_word_total
        scores["loss"] = total_loss / cnt
                
        self.report_test(self.state['sample_cnt'], scores, dataname)
        
        return predictions
    
class MemoryBase:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.images = []
        self.labels = []

        self.update_buffer = ()
        
        self.usage_count = np.array([])
        self.current_images = []
        self.current_labels = []

    def __len__(self):
        return len(self.images)

    def replace_sample(self, sample, idx=None):
        if idx is None:
            assert len(self.images) < self.memory_size
            self.images.append(sample)
        else:
            assert idx < self.memory_size
            self.images[idx] = sample

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
        mem_state = np.load(path, allow_pickle=True).item()
        self.images = mem_state['images']
        self.labels = mem_state['labels']
