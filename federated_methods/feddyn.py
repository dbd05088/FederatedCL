import torch
from utils.train_utils import load_deepspeed, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from transformers import TrainerCallback
from models.llava.llava_trainer import LLaVATrainer
import copy

ALPHA = 0.01

def feddyn_set_state_dict(model, global_state_dict, local_state_dict_list, training_args):
    global_auxiliary = {}
    for key in global_state_dict.keys():
        global_auxiliary[key] = torch.zeros_like(global_state_dict[key].cuda())
    return {
        'global_auxiliary': global_auxiliary,
    }

@torch.no_grad()
def feddyn_aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **kwargs):
    global_auxiliary = kwargs.get('global_auxiliary')
    # update server state
    model_delta = copy.deepcopy(global_auxiliary)
    for name, param in model_delta.items():
        model_delta[name] = torch.zeros_like(param)

    for client_params in local_state_dict_list:
        for name, param in model_delta.items():
            model_delta[name] = param + (client_params[name].cuda() - global_state_dict[name].cuda()) / num_selection

    for name, state_param in global_auxiliary.items():
        global_auxiliary[name] = state_param - ALPHA*model_delta[name]
    
    for key in global_state_dict.keys():
        global_state_dict[key] = sum([local_state_dict_list[client][key] / num_selection for client in selected_ids])
        global_state_dict[key] -= (1/ALPHA)*global_auxiliary[key].detach().cpu()
        
def feddyn_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    trainer = LLaVATrainerFEDDYN(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        **data_module,
        curr_round = extra_state_dict_dict['curr_round']
        )
    return trainer

class LLaVATrainerFEDDYN(LLaVATrainer):
    def __init__(self, curr_round, **kwargs):
        super(LLaVATrainerFEDDYN, self).__init__(**kwargs)
        self.feddyn_alpha = ALPHA
        old_grad = copy.deepcopy(self.model)
        old_grad = model_parameter_vector(old_grad)
        self.global_model_vector = None
        if curr_round > 0:
            self.global_model_vector = old_grad
        self.old_grad = torch.zeros_like(old_grad)

    def compute_loss(self, model, inputs, return_outputs=False):
        return_values = super(LLaVATrainerFEDDYN, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedDyn Loss
        if self.global_model_vector is not None:
            v1 = model_parameter_vector(self.model)
            loss += self.feddyn_alpha/2 * torch.norm(v1 - self.global_model_vector, 2)**2
            loss -= torch.dot(v1, self.old_grad)

        return (loss, outputs) if return_outputs else loss

def model_parameter_vector(model):
    param = [p.view(-1) for p in model.parameters() if p.requires_grad]
    return torch.cat(param, dim=0)