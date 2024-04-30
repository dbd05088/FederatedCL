import os
import time
import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset
from utils.train_utils import get_VLMmodel
from collections.abc import Mapping
import random

# writer = SummaryWriter("tensorboard")

import bitsandbytes
from transformers import Trainer
from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)
from transformers.optimization import get_scheduler
import glob

from utils.data_worker import ManagerWatchdog

from utils.eval_metrics import NLPEvaluator, matching_token_num
from models.llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from collections import OrderedDict
import json

OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"

import queue

class CLManagerServer: # == SERVER
    def __init__(
        self,
        train_datalists,
        test_datalists,
        device,
        data_args,
        model_args,
        args = None,
        bnb_model_from_pretrained_args=None,
        receive_channel=None,
        send_channel=None,
        logger=None
    ):
        kwargs = vars(args)

        self.args = args
        self.model = None
        self.tokenizer = None#tokenizer
        self.optimizer = None
        self.lr_scheduler = None
        
        self.data_args = data_args
        self.model_args = model_args
        self.bnb_model_from_pretrained_args=bnb_model_from_pretrained_args
        self.device = device
        self.task_id = 0
        self.method_name = kwargs["mode"]
        self.memory_size = kwargs["memory_size"]
        self.online_iter = kwargs["online_iter"]

        self.receive_channel = receive_channel
        self.send_channel = send_channel

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
        
        self.train_datalists = train_datalists
        self.test_datalists = test_datalists

        self.future_sample_num = 0
        self.future_sampling = True
        self.future_retrieval = True

        self.note = kwargs['note']
        self.rnd_seed = kwargs['seed']
        self.f_period = kwargs['f_period']
        self.f_next_time = 0
        self.start_time = time.time()
        self.total_samples = 0

        self.exposed_domains = []
        self.waiting_batch = []
        # federated learning
        self.num_clients = kwargs["num_clients"]
        self.frac_clients = 1.0#0.5
        self.num_rounds = kwargs["num_rounds"] # num of federated learning round
        self.n_gpu = kwargs["n_gpu"] # first one is for server

        self.total_flops = 0.0
        # self.writer = SummaryWriter(f'tensorboard/{self.dataset}/{self.note}/seed_{self.rnd_seed}')

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

        # start log file for server
        self.logger = open(f'./results/{self.method_name}/{self.note}/server.log', 'a')

    def run(self):
        for curr_round in range(self.num_rounds):
            # clients turn
            cids = np.arange(self.num_clients).tolist()
            num_selection = int(round(self.num_clients*self.frac_clients)) #4#
            selected_ids = sorted(random.sample(cids, num_selection)) #[0,1,2,3]#
            self.logger.write(f"Round {curr_round} | selected_ids: {selected_ids}\n")
            # print(f"Round {curr_round} | selected_ids: {selected_ids}\n")
            # selected_ids = cids
            for idx in range(num_selection):
                send_queue = self.send_channel[idx % len(self.send_channel)]
                client_id = selected_ids[idx]
                send_msg = {
                    'client_id':client_id,
                    'curr_round':curr_round,
                    'train_datalist':self.train_datalists[client_id],
                    'test_datalist':self.test_datalists[client_id],
                    'server_msg':self.server_msg(),
                }
                send_queue.put(send_msg)
            received_data_from_clients = 0
            
            while True:
                try:
                    r = self.receive_channel.get()
                except queue.Empty:
                    continue
                if r:
                    self.handle_msg_per_client(r)
                    
                    received_data_from_clients+=1
                    if received_data_from_clients == num_selection:
                        break

            self.do_server_work(curr_round)
        
        self.logger.write("total done\n")
        self.logger.close()
        for send_queue in self.send_channel:
            send_queue.put("done")
        return
    
    def server_msg(self):
        pass
    
    def handle_msg_per_client(self, msg):
        pass
    
    def do_server_work(self, cur_rround):
        pass
    
    def save_model(self, output_dir):
        state_dict = OrderedDict()
        with torch.no_grad():
            for name, parameters in self.model.named_parameters():
                if isinstance(parameters, torch.Tensor) and parameters.requires_grad:
                        state_dict[name] = parameters.detach().cpu()
        torch.save(state_dict, os.path.join(output_dir, f"server_model.pth"))
    
    def load_model(self, output_dir):
        state_dict = torch.load(os.path.join(output_dir, f"server_model.pth"), map_location=self.args.device)
        self.model.load_state_dict(state_dict, strict=False)
                
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

        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, OPTIMIZER_NAME) + "_*")
        )
        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
            map_location = self.args.device
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(
                torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location)
            )
            # self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))

    def report_training(self, sample_num, train_loss):
        # writer.add_scalar(f"train/loss", train_loss, sample_num)
        # if sample_num % 5 == 0:
        self.logger.write(
            f"Server Train | Sample # {sample_num} | train_loss {train_loss:.4f} |"# TFLOPs {self.total_flops/1000:.2f} | "
            f"running_time {datetime.timedelta(seconds=int(time.time() - self.start_time))} | "
            f"ETA {datetime.timedelta(seconds=int((time.time() - self.start_time) * (self.total_samples-sample_num) / sample_num))}"
        )
        self.logger.write("\n")

    def report_test(self, dataset_name, scores):
        # writer.add_scalar(f"test/loss_server", scores["loss"], sample_num)
        # writer.add_scalar(f"test/precision_server", scores["precision"], sample_num)
        # writer.add_scalar(f"test/Bleu_server", scores["Bleu_1"], sample_num)
        # writer.add_scalar(f"test/METEOR_server", scores["METEOR"], sample_num)
        # writer.add_scalar(f"test/RogueL_server", scores["ROUGE_L"], sample_num)
        # writer.add_scalar(f"test/CIDEr_server", scores["CIDEr"], sample_num)
        self.logger.write(
            f"Test (Server) | data {dataset_name} | test_loss {scores['loss']:.4f} | precision {scores['precision']:.4f} | Bleu_1 {scores['Bleu_1']} | Bleu_2 {scores['Bleu_2']} | Bleu_3 {scores['Bleu_3']} |Bleu_4 {scores['Bleu_4']} | METEOR {scores['METEOR']} | ROUGE_L {scores['ROUGE_L']} | CIDEr {scores['CIDEr']} |"
        )
        self.logger.write('\n')
        self.logger.flush()
    
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

        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluate_seendata(self, curr_round):
        eval_keys = []
        for i in range(len(self.test_datalists)):
            for data_info in self.test_datalists[i]:
                if data_info['data_name'] not in eval_keys:
                    self.evaluate(data_info['data'], data_info['data_name'], curr_round)
                    eval_keys.append(data_info['data_name'])

    def evaluate(self, test_datalist, dataset_name, curr_round):
        self.logger.write(f"server evaluate {dataset_name}\n")
        dataset = LazySupervisedDataset(test_datalist, self.tokenizer, self.data_args, preprocess=False)
        dataloader = DataLoader(dataset, batch_size= 4, num_workers=self.n_worker, collate_fn=DataCollatorForSupervisedDataset(tokenizer=self.tokenizer))
        
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
                imgs = batch['images']
                batch = self._prepare_inputs(batch)
                
                output = self.model(**batch)
                loss = output[0]
                pred_scores_list = output[1]
                # * keep logs
                n_correct = 0
                n_word = 0
                for inp, pred, gold, img in zip(inputs, pred_scores_list, input_labels, imgs):
                    valid_label_mask = gold.ne(IGNORE_INDEX)
                    valid_idx = valid_label_mask.nonzero()[0].item()
                    
                    n_word += len(torch.unique(gold[valid_label_mask]))#.sum()
                    pred_id = torch.argmax(pred, dim=1).cpu()#.to(device)
                    
                    # image token index
                    img_token_index = (inp==IMAGE_TOKEN_INDEX).nonzero()[0].item()
                    pred_id = torch.cat((pred_id[:img_token_index], torch.tensor([IMAGE_TOKEN_INDEX]), pred_id[img_token_index+self.img_feat_size*img.shape[0]:]))
                    
                    n_correct += matching_token_num(pred_id, gold, valid_idx, valid_label_mask)
                    
                    gold[valid_label_mask == False] = 0
                    
                    pred_sentence = self.tokenizer.decode(pred_id[valid_idx:], skip_special_tokens=True)#[valid_label_mask])
                    gold_sentence = self.tokenizer.decode(gold[valid_label_mask], skip_special_tokens=True)#[])
                    predictions.append({"sentence":pred_sentence, "gt_sentence":gold_sentence})

                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += loss
                cnt += 1
        #save predictions
        with open(f'./client_states/server_round{curr_round}_data{dataset_name}.json', 'w') as fp:
            json.dump(predictions, fp, indent=4)
        
        scores = NLPEvaluator(predictions).evaluate()
        scores["precision"] = n_word_correct / n_word_total
        scores["loss"] = total_loss / cnt
                
        self.report_test(dataset_name, scores)
        
        return predictions