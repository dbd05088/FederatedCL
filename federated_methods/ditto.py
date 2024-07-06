import torch
from models.llava.llava_trainer import LLaVATrainer
from models.duallora.dualloralayer import DualLoraLayer

            
def ditto_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    trainer = LLaVATrainerDITTO(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        **data_module,
        global_state=extra_state_dict_dict['global_state']
        )
    return trainer


class LLaVATrainerDITTO(LLaVATrainer):   
    def __init__(self, global_state, **kwargs):
        super(LLaVATrainerDITTO, self).__init__(**kwargs)
        self.global_state = global_state
        for k in self.global_state.keys():
            self.global_state[k] = self.global_state[k].cuda()
        self.mu = 0.001
    def compute_loss(self, model, inputs, return_outputs=False):
        
        # global forward
        for name, module in model.module.named_modules():
            if isinstance(module, DualLoraLayer):
                module.set_state('lora1')
        model.module.base_model.model.model.mm_projector = model.module.base_model.model.model.global_mm_projector
        loss_global, outputs = super(LLaVATrainerDITTO, self).compute_loss(model, inputs, return_outputs=True)     
        # local forward
        for name, module in model.module.named_modules():
            if isinstance(module, DualLoraLayer):
                module.set_state('lora2')
        model.module.base_model.model.model.mm_projector = model.module.base_model.model.model.local_mm_projector
        loss_local, local_outputs = super(LLaVATrainerDITTO, self).compute_loss(model, inputs, return_outputs=True) 
        
        loss = loss_global + loss_local
        # Apply FedProx Loss
        for name, param in model.module.named_parameters():
            # only trainable parameters
            if not param.requires_grad:
                continue
            if 'local_mm_projector' in name:
                target_name = name.replace('local_mm_projector', 'global_mm_projector')
                loss += self.mu / 2 * (torch.norm(param - self.global_state[target_name])) ** 2
            elif 'lora2' in name:
                target_name = name.replace('lora2', 'lora1')
                loss += self.mu / 2 * (torch.norm(param - self.global_state[target_name])) ** 2
        model.module.base_model.model.model.mm_projector = None
        return (loss, outputs) if return_outputs else loss
