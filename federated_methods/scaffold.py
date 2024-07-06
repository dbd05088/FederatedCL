import torch
from utils.train_utils import load_deepspeed, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from transformers import TrainerCallback
from models.llava.llava_trainer import LLaVATrainer
import copy
from typing import Optional, Dict, Union, Any
from torch import nn
from transformers.trainer import is_sagemaker_mp_enabled
from collections import OrderedDict

def scaffold_set_state_dict(model, global_state_dict, local_state_dict_list, training_args):
    global_auxiliary = {}               # c in SCAFFOLD
    for key in global_state_dict.keys():
        global_auxiliary[key] = torch.zeros_like(global_state_dict[key].cuda())
    auxiliary_model_list = [copy.deepcopy(global_auxiliary) for _ in range(training_args.num_clients)]    # c_i in SCAFFOLD
    auxiliary_delta_dict = [copy.deepcopy(global_auxiliary) for _ in range(training_args.num_clients)]    # delta c_i in SCAFFOLD
    return {
        'global_state':global_state_dict,
        'global_auxiliary':global_auxiliary,
        'auxiliary_model_list':auxiliary_model_list,
        'auxiliary_delta_dict':auxiliary_delta_dict,
    }

def scaffold_aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **kwargs):
    # num_selection -= 1
    # selected_ids.remove(6)
    for key in global_state_dict.keys():
        # global_state_dict[key] = sum([local_state_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in selected_ids])
        global_state_dict[key] = sum([local_state_dict_list[client][key] / num_selection for client in selected_ids])

    global_auxiliary, auxiliary_delta_dict = kwargs.get('global_auxiliary'), kwargs.get('auxiliary_delta_dict')
    for key in global_auxiliary.keys():
        delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in selected_ids]) 
        global_auxiliary[key] += delta_auxiliary / num_selection#training_args.num_clients
        
def scaffold_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    trainer = LLaVATrainerSCAFFOLD(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        **data_module,
        global_state_dict=extra_state_dict_dict['global_state'],
        global_auxiliary=extra_state_dict_dict['global_auxiliary'],
        local_auxiliary=extra_state_dict_dict['auxiliary_model_list'][extra_state_dict_dict['client_id']]
        )
    # trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
    return trainer

class LLaVATrainerSCAFFOLD(LLaVATrainer):
    def __init__(self, global_state_dict, global_auxiliary, local_auxiliary, **kwargs):
        super(LLaVATrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state_dict
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(self.local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    # name = name.replace(".default", "")
                    auxiliary_new_para[name] = (self.global_state[name].cuda() - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        model_params = OrderedDict(self.model.named_parameters())
        for name, param in model_params.items():
            if param.grad is not None:
                model_params[name].grad.data += (self.correction[name])

        return loss.detach() / self.args.gradient_accumulation_steps

class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model
    def on_step_end(self, args, state, control, **kwargs):
        trainable_parameters = get_peft_state_maybe_zero_3(
                self.model.named_parameters(), args.lora_bias
            )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            self.model.named_parameters()
        )
        trainable_parameters.update(non_lora_state_dict)
        trainable_parameters = copy.deepcopy(trainable_parameters)
        for name in trainable_parameters.keys():
            trainable_parameters[name] = trainable_parameters[name].cuda() - args.learning_rate * self.correction[name]
        with torch.no_grad():
            if 'zero3' in args.deepspeed:
                load_deepspeed(trainable_parameters, self.model, strict=False)
            else:
                self.model.load_state_dict(trainable_parameters, strict=False)