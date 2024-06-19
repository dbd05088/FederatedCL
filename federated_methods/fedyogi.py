import torch
from utils.train_utils import load_deepspeed, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from transformers import TrainerCallback
from models.llava.llava_trainer import LLaVATrainer
import copy

TAU = 1e-3
BETA1 = 0.9
BETA2 = 0.99
ETA = 1e-3

def fedyogi_set_state_dict(model, global_state_dict, local_state_dict_list, training_args):
    proxy_dict = {}
    opt_proxy_dict = {}
    for key in global_state_dict.keys():
        proxy_dict[key] = torch.zeros_like(global_state_dict[key]).cuda()
        opt_proxy_dict[key] = (torch.ones_like(global_state_dict[key])*TAU**2).cuda()
    return {
        'proxy_dict':proxy_dict,
        'opt_proxy_dict':opt_proxy_dict,
    }

@torch.no_grad()
def fedyogi_aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **kwargs):
    proxy_dict = kwargs.get('proxy_dict')
    opt_proxy_dict = kwargs.get('opt_proxy_dict')
    curr_round = kwargs.get('curr_round')
    
    for key, param in opt_proxy_dict.items():
        delta_w = sum([state_dict[key] - global_state_dict[key] for state_dict in local_state_dict_list]) / num_selection
        proxy_dict[key] = BETA1 * proxy_dict[key] + (1 - BETA1) * delta_w.cuda() if curr_round > 0 else delta_w.cuda()
        delta_square = torch.square(proxy_dict[key])
        opt_proxy_dict[key] = param - (1-BETA2)*delta_square*torch.sign(param - delta_square)
        global_state_dict[key] += ETA * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+TAU).detach().cpu()