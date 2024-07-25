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

    
def dap_attn_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    task_id = extra_state_dict_dict['task_id'] if 'task_id' in extra_state_dict_dict else None
    trainer = LLaVATrainerDAPATTN(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        **data_module,
        task_id=task_id
        )
    return trainer


class LLaVATrainerDAPATTN(LLaVATrainerFEDAVG):
    def __init__(self, task_id=None, **kwargs):
        super(LLaVATrainerDAPATTN, self).__init__(**kwargs)
        
        self.task_id = task_id
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs, task_id=self.task_id)
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

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name or "vision_tower" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if ('_prompt' in n and p.requires_grad and 'lang_prompt_feature_embedding' not in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                        
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if ('_prompt' in n and p.requires_grad and 'lang_prompt_feature_embedding' not in n)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        print(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        self.logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                print(f"skipped: {skipped/2**20}M params")

        return self.optimizer
    

