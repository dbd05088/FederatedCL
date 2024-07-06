import torch
from utils.train_utils import load_deepspeed
from models.llava.llava_trainer import LLaVATrainer
  
def fedprox_set_state_dict(model, global_state_dict, local_state_dict_list, training_args):
    return {
        'global_state': global_state_dict,
    }

def fedprox_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    trainer = LLaVATrainerFEDPROX(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        **data_module,
        global_state=extra_state_dict_dict['global_state'],
        )
    return trainer

class LLaVATrainerFEDPROX(LLaVATrainer):
    def __init__(self, global_state, **kwargs):
        super(LLaVATrainerFEDPROX, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = 0.01
    
    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(LLaVATrainer, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.module.named_parameters():
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name].cuda()) ** 2

        return (loss, outputs) if return_outputs else loss