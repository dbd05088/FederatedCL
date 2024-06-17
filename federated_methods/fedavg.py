import torch
from utils.train_utils import load_deepspeed
from models.llava.llava_trainer import LLaVATrainer

def fedavg_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args):
    model_to_load = global_state_dict
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(model_to_load, model, strict=False)
        else:
            model.load_state_dict(model_to_load, strict=False)  

def fedavg_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    trainer = LLaVATrainer(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        **data_module,
        )
    return trainer

def fedavg_aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **kwargs):
    for key in global_state_dict.keys():
        # global_state_dict[key] = sum([local_state_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in selected_ids])
        global_state_dict[key] = sum([local_state_dict_list[client][key] / num_selection for client in selected_ids])
    