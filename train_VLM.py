import logging.config
import os
import random

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.train_utils import get_VLMmodel, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer, load_deepspeed

# from utils.method_manager_VLM import select_method
from federated_methods.method_manager import select_method
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset
from typing import Dict, Optional, Sequence, List

from torch import multiprocessing
import copy
import torch.distributed as dist
import json
from transformers import BitsAndBytesConfig
from models.llava.llava_trainer import LLaVATrainer
from collections import OrderedDict
from deepspeed import zero
import time
import datetime
# import warnings
# warnings.filterwarnings('ignore')

def main():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))    
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    logging.config.fileConfig("./configuration/logging.conf")
    logger = logging.getLogger()

    os.makedirs(f"results/{training_args.mode}/{training_args.note}", exist_ok=True)
    os.makedirs(f"tensorboard/{training_args.mode}/{training_args.note}", exist_ok=True)
    fileHandler = logging.FileHandler(f'results/{training_args.mode}/{training_args.note}/seed_{training_args.seed}.log', mode="w")

    # writer = SummaryWriter(f'tensorboard/{training_args.mode}/{training_args.note}/federated')

    formatter = logging.Formatter(
        "[%(levelname)s] %(filename)s:%(lineno)d > %(message)s"
    )
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    if training_args.local_rank == 0 or training_args.local_rank == -1: 
        logger.info(training_args)

    # Fix the random seeds
    torch.manual_seed(training_args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(training_args.seed)
    random.seed(training_args.seed)

    model, tokenizer, data_args = get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args)

    train_datalists, test_datalists = get_datalists(training_args, training_args.scenario)
    
    # select functions
    set_state_dict, load_state_dict, create_trainer, aggregate_state_dict, extra_modules = select_method(training_args.mode)
    
    # create folder
    training_args.state_dir = training_args.state_dir + '_' + training_args.note
    if not os.path.exists(training_args.state_dir):
        os.makedirs(training_args.state_dir)
    
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {'use_reentrant':False}

    global_state_dict = get_peft_state_maybe_zero_3(
                model.named_parameters(), training_args.lora_bias
            )
    non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        model.named_parameters()
    )
    global_state_dict.update(non_lora_state_dict)
    local_state_dict_list = [copy.deepcopy(global_state_dict) for i in range(training_args.num_clients)]
    
    extra_state_dict_dict = set_state_dict(model, global_state_dict, local_state_dict_list, training_args)
    training_loss = [[] for i in range(training_args.num_clients)]
    
    # start federated learning
    start_time = time.time()
    frac_clients = 1
    memory = [[]]*training_args.num_clients
    memory_size = 50000
    num_iterations = training_args.num_iter
    total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps
    init_lr = training_args.learning_rate
    final_lr = 1e-6
    lr_step = (init_lr - final_lr)/training_args.num_rounds
    
    for curr_round in range(training_args.num_rounds):
        # clients turn
        cids = np.arange(training_args.num_clients).tolist()
        num_selection = int(round(training_args.num_clients*frac_clients)) #4#
        selected_ids = sorted(random.sample(cids, num_selection)) #[0,1,2,3]#
        if training_args.local_rank == 0 or training_args.local_rank == -1: 
            logger.info(f"Round {curr_round} | selected_ids: {selected_ids}\n")
        # print(f"Round {curr_round} | selected_ids: {selected_ids}\n")
        # selected_ids = cids
        training_args.learning_rate = init_lr - lr_step*curr_round
        training_args.mm_projector_lr = init_lr - lr_step*curr_round
        for idx in range(num_selection):
            model.config.use_cache = False
            torch.cuda.empty_cache()
            client_id = selected_ids[idx]
            
            load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args)
            print('model loading done')
            
            ##### simulate online memory insertion & get_batch ####
            sub_dataset = get_dataset_this_round(train_datalists[client_id], curr_round, training_args)
            
            iteration = 0
            datalist = []
            iter_ratio = num_iterations / len(sub_dataset)
            for i, sample in enumerate(sub_dataset):
                if len(memory[client_id]) == memory_size:
                    memory[client_id].pop(random.randrange(memory_size))
                memory[client_id].append(sample)
                iteration += iter_ratio
                if iteration >= 1:
                    for _ in range(int(iteration)):
                        batch = random.sample(memory[client_id], k=min(len(memory[client_id]), total_batchsize))
                        mul = (total_batchsize//len(batch)) + 1
                        batch = (batch*mul)[:total_batchsize]
                        datalist.extend(batch[:])
                        iteration -= 1
            
            data_module = make_supervised_data_module(client_data=datalist, # sub_dataset
                                                tokenizer=tokenizer,
                                                data_args=copy.deepcopy(data_args))
            
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                logger.info(f'Round {curr_round} | train client {client_id} | num samples {len(sub_dataset)}')

            # ===== Train local model on the client side =====
            extra_state_dict_dict['client_id'] = client_id
            extra_state_dict_dict['curr_round'] = curr_round
            trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict)

            # if curr_round > 0:
            #     path = os.path.join(training_args.state_dir, f"{client_id}_trainer_state.json")
            #     shutil.copy(path, os.path.join(training_args.output_dir, "trainer_state.json"))
            #     results = trainer.train(resume_from_checkpoint=True)
            # else:
            results = trainer.train()
            training_loss[client_id].append(results.training_loss)
            
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                path = os.path.join(training_args.state_dir, f"{client_id}_trainer_state.json")
                trainer.state.save_to_json(path)
            
            model.config.use_cache = True
            
            
            
            # save local model
            output_dir = os.path.join(training_args.state_dir, f"{client_id}_client_model_round{curr_round+1}.pth")
            if training_args.lora_enable:
                state_dict = get_peft_state_maybe_zero_3(
                    model.named_parameters(), training_args.lora_bias
                )
                non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                    model.named_parameters()
                )
                state_dict.update(non_lora_state_dict)
                    # model.config.save_pretrained(training_args.output_dir)
                    # model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                    # torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            else:
                state_dict = {k: t.detach().cpu().clone() for k, t in model.named_parameters() if t.requires_grad}
            if (training_args.local_rank == 0 or training_args.local_rank == -1) and training_args.mode != 'pfedpg': 
                torch.save(state_dict, output_dir)
            local_state_dict_list[client_id] = copy.deepcopy(state_dict)
            
            if training_args.mode == 'scaffold':
                local_auxiliary, auxiliary_delta = trainer.get_auxiliary_param()
                extra_state_dict_dict['auxiliary_model_list'][client_id] = local_auxiliary
                extra_state_dict_dict['auxiliary_delta_dict'][client_id] = auxiliary_delta
            
            trainer.deepspeed.empty_partition_cache()
            del trainer
            logger.info(f"done Round {curr_round} client {client_id} | elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")
        #self.do_server_work(curr_round)
        
        aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **extra_state_dict_dict)
        
        # TODO: Save server model
        if (training_args.local_rank == 0 or training_args.local_rank == -1) and training_args.mode != 'pfedpg': 
            torch.save(global_state_dict, os.path.join(training_args.state_dir, f"server_model_round{curr_round}.pth"))
        
    logger.info("total done\n")


def get_dataset_this_round(train_datalists, curr_round, training_args):
    sample_per_round = len(train_datalists) // training_args.num_rounds
    return train_datalists[curr_round*sample_per_round:(curr_round+1)*sample_per_round]
    

def make_supervised_data_module(client_data, tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(client_data, tokenizer, data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def get_datalists(args, scenario_num):
    with open(f"./scenarios/scenario-{scenario_num}.json") as fp:
        scenario = json.load(fp)
    assert args.num_clients == len(scenario)

    train_datalists = {}
    test_datalists = {}

    for client_data in scenario:
        client_id = client_data['client_id']
        train_datalist = []
        test_datalist = []
        eval_cnt = 0
        for data in client_data['datasets']:
            with open(f"./dataset/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            random.shuffle(datalist)
            train_datalist.extend(datalist)

            with open(f"./dataset/{data['dataset']}/test/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            test_datalist.append({
                "data_name": f"{data['dataset']}-{data['subset_id']}",
                "data": datalist,
                "eval_cnt": eval_cnt})
            eval_cnt += len(datalist)
            
            train_datalists[client_id] = train_datalist
        test_datalists[client_id] = test_datalist

    return train_datalists, test_datalists

if __name__ == "__main__":
    main()