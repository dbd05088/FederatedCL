import torch
from utils.train_utils import load_deepspeed, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from transformers import TrainerCallback
from federated_methods.fedavg import LLaVATrainerFEDAVG
import copy
from torch import nn
import torch.nn.functional as F
import math
from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)
from transformers import Trainer
import bitsandbytes
import os
from transformers.trainer import unwrap_model, _is_peft_model, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from models.ia3pool.ia3poollayer import IA3Layer
    
def task_id_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    task_id = extra_state_dict_dict['task_id'] if 'task_id' in extra_state_dict_dict else None
    trainer = LLaVATrainerTaskId(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        **data_module,
        task_id=task_id
        )
    return trainer


class LLaVATrainerTaskId(LLaVATrainerFEDAVG):
    def __init__(self, task_id=None, **kwargs):
        super(LLaVATrainerTaskId, self).__init__(**kwargs)
        
        self.task_id = task_id
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        output = super()._inner_training_loop(batch_size=batch_size, args=args,resume_from_checkpoint=resume_from_checkpoint,ignore_keys_for_eval=ignore_keys_for_eval)
        
        # for name, module in self.model.named_modules():
        #     if isinstance(module, IA3Layer):
        #         module.init_next_ia3(self.task_id)
    
        return output
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        if 'prompt' in inputs:
            text_prompt = inputs.pop('prompt')
        else:
            text_prompt = None
        outputs = model(**inputs, task_id=self.task_id, prompt=text_prompt) if text_prompt else model(**inputs, task_id=self.task_id)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss