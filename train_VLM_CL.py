import logging.config
import os
import random

import numpy as np
import torch
from configuration.VLM_config_new import ModelArguments, DataArguments, TrainingArguments
import transformers
from utils.train_utils import get_VLMmodel, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer, load_deepspeed

from federated_methods.method_manager import select_method
from utils.data_loader_VLM import LazySupervisedDataset, DataCollatorForSupervisedDataset
from typing import Dict

import copy
import json
from transformers import BitsAndBytesConfig
import time
import datetime
import torch.nn.functional as F

from models.coda_prompt import CodaPrompt

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
    old_local_state_dict_list = [copy.deepcopy(local_state_dict_list[i]) for i in range(len(local_state_dict_list))]
    local_state_dict_keys = local_state_dict_list[0].keys()
    extra_state_dict_dict = set_state_dict(model, global_state_dict, local_state_dict_list, training_args)

    training_loss = [[] for i in range(training_args.num_clients)]
    
    # start federated learning
    start_time = time.time()
    frac_clients = 1
    
    memory = [[] for id in range(training_args.num_clients)]
    memory_size = training_args.memory_size
    total_batchsize = training_args.per_gpu_train_batch_size*training_args.world_size*training_args.gradient_accumulation_steps
    init_lr = training_args.learning_rate
    mm_init_lr = training_args.mm_projector_lr
    final_lr = training_args.final_lr
    mm_final_lr = training_args.mm_final_lr
    
    total_rounds = training_args.num_rounds * training_args.num_tasks
    last_task_id = [-1 for _ in range(training_args.num_clients)]
    fisher_olds = [None for _ in range(training_args.num_clients)]
    task_vectors = [None for _ in range(training_args.num_clients)]
    
    lr_step = (init_lr - final_lr)/total_rounds
    mm_lr_step = (mm_init_lr - mm_final_lr)/total_rounds
    for curr_round in range(total_rounds):
        old_local_state_dict_list = [copy.deepcopy(local_state_dict_list[i]) for i in range(len(local_state_dict_list))]
        if curr_round > 0:
            task_vector = F.normalize(torch.stack(task_vectors, dim=0), dim=-1)
            sim = torch.matmul(task_vector,
                            torch.transpose(task_vector, 1, 0))
            sim = torch.transpose(sim, 1, 0)
            extra_state_dict_dict['task_similarty'] = sim
        # clients turn
        cids = np.arange(training_args.num_clients).tolist()
        num_selection = int(round(training_args.num_clients*frac_clients)) #4#
        selected_ids = sorted(random.sample(cids, num_selection)) #[0,1,2,3]#
        if training_args.local_rank == 0 or training_args.local_rank == -1: 
            logger.info(f"Round {curr_round} | selected_ids: {selected_ids}\n")
        
        # selected_ids = cids
        training_args.learning_rate = init_lr - lr_step*curr_round
        training_args.mm_projector_lr = mm_init_lr - mm_lr_step*curr_round
        if curr_round > 0 and training_args.is_wsd:
            training_args.warmup_ratio = 0
            training_args.warmup_steps = 0
        for idx in range(num_selection):
            model.config.use_cache = False
            torch.cuda.empty_cache()
            client_id = selected_ids[idx]
            
            ##### simulate online memory insertion & get_batch ####
            sub_dataset = train_datalists[client_id][curr_round]['datalist']
            num_iterations = train_datalists[client_id][curr_round]['num_iter']
            
            task_id = train_datalists[client_id][curr_round]['task_id']
            
            extra_state_dict_dict['client_id'] = client_id
            extra_state_dict_dict['curr_round'] = curr_round
            if training_args.use_task_id:
                extra_state_dict_dict['task_id'] = task_id
            
            load_state_dict(model, global_state_dict, old_local_state_dict_list, client_id, training_args, extra_state_dict_dict)
            print('model loading done')
            
            if 'CodaPrompt' in training_args.mode and task_id is not None and task_id != last_task_id[client_id]:
                for n, m in model.named_modules():
                    if isinstance(m, CodaPrompt):
                        m.process_task_count(task_id)
                last_task_id[client_id] = task_id
            
            iteration = 0
            datalist = []
            iter_ratio = num_iterations / len(sub_dataset)
            
            if not training_args.is_streamonly:
                # memory-only
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
                
                if len(datalist) < num_iterations*total_batchsize:
                    batch = random.sample(memory[client_id], k=min(len(memory[client_id]), total_batchsize))
                    mul = (total_batchsize//len(batch)) + 1
                    batch = (batch*mul)[:total_batchsize]
                    datalist.extend(batch[:])
            else:
                # stream-only
                datalist = sub_dataset[:num_iterations*total_batchsize]
            
            data_module = make_supervised_data_module(client_data=datalist, # sub_dataset
                                                tokenizer=tokenizer,
                                                data_args=copy.deepcopy(data_args))
            
            if training_args.local_rank == 0 or training_args.local_rank == -1: 
                logger.info(f'Round {curr_round} | train client {client_id} | num samples {len(sub_dataset)}')

            # ===== Train local model on the client side =====
            
            if training_args.mode == 'ours_generator4':
                extra_state_dict_dict['fisher_old'] = fisher_olds[client_id]
            trainer = create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict)

            results = trainer.train()
            training_loss[client_id].append(results.training_loss)
            if training_args.mode == 'ours_generator4':
                fisher_olds[client_id] = trainer.fisher_old
            
            if training_args.mode == 'ours_generator':
                task_vectors[client_id] = trainer.task_vector
            
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
            else:
                state_dict = {k: t.detach().cpu().clone() for k, t in model.named_parameters() if t.requires_grad}
            
            local_state_dict_list[client_id] = copy.deepcopy(state_dict)
            
            k_to_del = []
            for k in state_dict.keys():
                if k not in local_state_dict_keys:
                    k_to_del.append(k)
            for k in k_to_del:
                del state_dict[k]
            if (training_args.local_rank == 0 or training_args.local_rank == -1):
                torch.save(state_dict, output_dir)
            
            local_state_dict = getattr(trainer, 'global_weight', None)
            if local_state_dict is not None:
                local_state_dict_list[client_id] = copy.deepcopy(local_state_dict)
            
            if training_args.mode == 'scaffold':
                local_auxiliary, auxiliary_delta = trainer.get_auxiliary_param()
                extra_state_dict_dict['auxiliary_model_list'][client_id] = local_auxiliary
                extra_state_dict_dict['auxiliary_delta_dict'][client_id] = auxiliary_delta
            
            trainer.deepspeed.empty_partition_cache()
            del trainer
            logger.info(f"done Round {curr_round} client {client_id} | elapsed time {datetime.timedelta(seconds=int(time.time() - start_time))} | ")

        
        aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **extra_state_dict_dict)
        
        # Save server model
        if (training_args.local_rank == 0 or training_args.local_rank == -1): 
            torch.save(global_state_dict, os.path.join(training_args.state_dir, f"server_model_round{curr_round}.pth"))
            
    if training_args.mode == 'ours_generator':
        task_vector = F.normalize(torch.stack(task_vectors, dim=0), dim=-1)
        sim = torch.matmul(task_vector,
                        torch.transpose(task_vector, 1, 0))
        sim = torch.transpose(sim, 1, 0)
        extra_state_dict_dict['task_similarty'] = sim
        extra_state_dict_dict['curr_round'] += 1
        for client_id in range(training_args.num_clients):
            load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict)
    logger.info("total done\n")

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
    
    max_iterations = args.num_iter
    rounds_per_task = args.num_rounds

    for client_data in scenario:
        client_id = client_data['client_id']
        train_datalist = []
        test_datalist = []
        eval_cnt = 0
        for task_id, data in enumerate(client_data['datasets']):
            with open(f"./dataset/{data['dataset']}/train/dataset-{str(data['subset_id'])}.json") as fp:
                datalist = json.load(fp)
            random.shuffle(datalist)
            samplenum_per_rounds = int(len(datalist) / rounds_per_task)
            num_iter = max(int(max_iterations*samplenum_per_rounds/2000), 2) # 10000 / 5 = 2000
            for i in range(rounds_per_task):
                train_datalist.append(
                    {'datalist':datalist[i*samplenum_per_rounds:(i+1)*samplenum_per_rounds],
                     'num_iter': num_iter,
                     'task_id': task_id})
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