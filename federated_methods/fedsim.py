import torch
from torch import nn
from models.llava.llava_trainer import LLaVATrainer

from models.duallora.dualloralayer import DualLoraLayer

def fedsim_set_state_dict(model, global_state_dict, local_state_dict_list, training_args):
    # layers_to_del = layer_num[-len(layer_num)//2:]
    keys_to_del = []
    for k in global_state_dict.keys():
        if 'lora2' in k or 'local_mm_projector' in k:
            keys_to_del.append(k)
    for k in keys_to_del:
        del global_state_dict[k]
    
    local_keys_to_del = []
    for k in local_state_dict_list[0].keys():
        if 'lora1' in k or 'global_mm_projector' in k:
            local_keys_to_del.append(k)
    for client_id in range(training_args.num_clients):
        for k in local_keys_to_del:
            del local_state_dict_list[client_id][k]
    
    return {'global_state':global_state_dict}
            
def fedsim_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    trainer = LLaVATrainerFEDSIM(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        **data_module,
        )
    return trainer


class LLaVATrainerFEDSIM(LLaVATrainer):   
    def compute_loss(self, model, inputs, return_outputs=False):
        
        # global forward
        for name, module in model.module.named_modules():
            if isinstance(module, DualLoraLayer):
                module.set_state('lora1')
        model.module.base_model.model.model.mm_projector = model.module.base_model.model.model.global_mm_projector
        _, outputs = super(LLaVATrainerFEDSIM, self).compute_loss(model, inputs, return_outputs=True)     
        # local forward
        for name, module in model.module.named_modules():
            if isinstance(module, DualLoraLayer):
                module.set_state('lora2')
        model.module.base_model.model.model.mm_projector = model.module.base_model.model.model.local_mm_projector
        _, local_outputs = super(LLaVATrainerFEDSIM, self).compute_loss(model, inputs, return_outputs=True) 
        
        final_logits = outputs['logits'] + local_outputs['logits']
        labels = model.module.labels
        # Shift so that tokens < n predict n
        shift_logits = final_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, model.module.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels) 
        
        model.module.base_model.model.model.mm_projector = None
        return (loss, outputs) if return_outputs else loss

