from federated_methods.task_id import LLaVATrainerTaskId
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import RandomSampler
from packaging import version
from torch import nn
from utils.train_utils import load_deepspeed
from transformers.utils import logging
import sys, os, time, shutil
import math
from typing import Optional, Dict, Union, Any
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    has_length,
    speed_metrics,
)
from transformers.trainer_pt_utils import get_model_param_count, get_dataloader_sampler, reissue_pt_warnings
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers import Trainer
import bitsandbytes
from transformers.trainer import (
    is_sagemaker_mp_enabled, 
    _is_peft_model, 
    TRAINER_STATE_NAME,
    is_torch_xla_available,
    is_accelerate_available,
    is_deepspeed_available,
    get_parameter_names,
    ALL_LAYERNORM_LAYERS, SCHEDULER_NAME
)
from transformers.integrations import hp_params
from transformers.trainer_callback import TrainerState
from transformers.training_args import ParallelMode

if is_accelerate_available():
    from accelerate import skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedType
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]
    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

import warnings

logger = logging.get_logger(__name__)

def fedours_ema_distill_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    task_id = extra_state_dict_dict['task_id'] if 'task_id' in extra_state_dict_dict else None
    ema_ratio = training_args.ema_ratio
    trainer = LLaVATrainerOURS(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        client_id=extra_state_dict_dict['client_id'],
        curr_round=extra_state_dict_dict['curr_round'],
        task_id = task_id,
        ema_ratio=ema_ratio,
        task_vector=extra_state_dict_dict['task_vector'] if 'task_vector' in extra_state_dict_dict else None,
        fisher_old=extra_state_dict_dict['fisher_old'] if 'fisher_old' in extra_state_dict_dict else None,
        fisher_freq=extra_state_dict_dict['fisher_freq'] if 'fisher_freq' in extra_state_dict_dict else 5,
        **data_module,
        )
    return trainer

def fedours_load_state_dict(model, global_state_dict, local_state_dict_list, client_id, training_args, extra_state_dict_dict=None):
    # first load loca model and then load global model
    with torch.no_grad():
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(local_state_dict_list[client_id], model, strict=False)
        else:
            model.load_state_dict(local_state_dict_list[client_id], strict=False)  
            
        # exlude mean
        # if extra_state_dict_dict['curr_round'] > 0:
        #     new_global_state_dict = {}
        #     for name in global_state_dict.keys():
        #         new_param = 0
        #         for id in range(training_args.num_clients):
        #             if id == client_id:
        #                 continue
        #             new_param += local_state_dict_list[id][name]
        #         new_param /= (training_args.num_clients - 1)
        #         new_global_state_dict[name] = new_param
        # else:
        #     new_global_state_dict = global_state_dict
        # if 'zero3' in training_args.deepspeed:
        #     load_deepspeed(new_global_state_dict, model, strict=False)
        # else:
        #     model.load_state_dict(new_global_state_dict, strict=False) 
        
        # gradient based similarity wegithed averaging (exclude own)
        if extra_state_dict_dict['curr_round'] > 0 and 'task_similarity' in extra_state_dict_dict:
            # similarity matrix
            sim = extra_state_dict_dict['task_similarity']
            new_global_state_dict = {}
            
            weights = sim[client_id].clone()
            
            weights[client_id] = -1e9
            weights = (weights/0.5).softmax(dim=0)
            
            sim_sum = weights.sum() - weights[client_id]
            
            # # weights[client_id] = sim_sum
            # # sim_sum += sim_sum
            
            for name in global_state_dict.keys():
                new_param = 0
                if 'lora1' in name:
                    target_key = name.replace('lora1', 'lora2')
                elif 'ia3_l_1' in name:
                    target_key = name.replace('ia3_l_1', 'ia3_l_2')
                elif 'ia3_generator_1' in name:
                    target_key = name.replace('ia3_generator_1', 'ia3_generator_2')
                elif 'lang_prompt_dap_key_embeddings_1' in name:
                    target_key = name.replace('lang_prompt_dap_key_embeddings_1', 'lang_prompt_dap_key_embeddings_2')
                elif 'lang_prompt_downsample_1' in name:
                    target_key = name.replace('lang_prompt_downsample_1', 'lang_prompt_downsample_2')
                elif 'lang_prompt_norm_1' in name:
                    target_key = name.replace('lang_prompt_norm_1', 'lang_prompt_norm_2')
                elif 'lang_prompt_ia3_pool_1' in name:
                    target_key = name.replace('lang_prompt_ia3_pool_1', 'lang_prompt_ia3_pool_2')
                
                for id in range(training_args.num_clients):
                    if id == client_id:
                        continue
                    new_param += weights[id]*local_state_dict_list[id][target_key] / sim_sum
                    
                new_global_state_dict[name] = new_param
            if (training_args.local_rank == 0 or training_args.local_rank == -1):
                output_dir = os.path.join(training_args.state_dir, f"{client_id}_client_global_model_round{extra_state_dict_dict['curr_round']}.pth")
                torch.save(new_global_state_dict, output_dir)
                
            # threshold = 0.7
            # if all(weights < threshold):
            #     new_global_state_dict = global_state_dict
            # else:
            #     similar_client_id = (weights >= threshold).nonzero(as_tuple=True)[0]
            #     sim_sum = weights[similar_client_id].sum()
            #     for name in global_state_dict.keys():
            #         new_param = 0
            #         if 'lora1' in name:
            #             target_key = name.replace('lora1', 'lora2')
            #         elif 'ia3_l_1' in name:
            #             target_key = name.replace('ia3_l_1', 'ia3_l_2')
            #         elif 'ia3_generator_1' in name:
            #             target_key = name.replace('ia3_generator_1', 'ia3_generator_2')
            #         elif 'lang_prompt_dap_key_embeddings_1' in name:
            #             target_key = name.replace('lang_prompt_dap_key_embeddings_1', 'lang_prompt_dap_key_embeddings_2')
            #         elif 'lang_prompt_downsample_1' in name:
            #             target_key = name.replace('lang_prompt_downsample_1', 'lang_prompt_downsample_2')
            #         elif 'lang_prompt_norm_1' in name:
            #             target_key = name.replace('lang_prompt_norm_1', 'lang_prompt_norm_2')
            #         elif 'lang_prompt_ia3_pool_1' in name:
            #             target_key = name.replace('lang_prompt_ia3_pool_1', 'lang_prompt_ia3_pool_2')
            #         for id in similar_client_id:
            #             new_param += weights[id]*local_state_dict_list[id][target_key] / sim_sum
            #         new_global_state_dict[name] = new_param
            # if (training_args.local_rank == 0 or training_args.local_rank == -1):
            #     output_dir = os.path.join(training_args.state_dir, f"{client_id}_client_global_model_round{extra_state_dict_dict['curr_round']}.pth")
            #     torch.save(new_global_state_dict, output_dir)
        else:
            new_global_state_dict = global_state_dict
        if 'zero3' in training_args.deepspeed:
            load_deepspeed(new_global_state_dict, model, strict=False)
        else:
            model.load_state_dict(new_global_state_dict, strict=False) 


def kl_loss(output, target, temp=2):
    if output.shape[-1]>3000:
        p = F.log_softmax(output / temp, dim=-1)
        q = F.softmax(target / temp, dim=-1)
    else:
        p = F.log_softmax(output / temp, dim=1)
        q = F.softmax(target / temp, dim=1)

    l_kl = F.kl_div(p, q, reduction="batchmean") #FIXME
    l_kl = l_kl * temp**2
    return l_kl

@torch.no_grad()
def get_grad_penultimate(logit, label, weight_last, input_penultimate, norm_layer, hidden_states_before_norm):
    # Compute probabilities using softmax
    prob = F.softmax(logit, dim=-1).to(torch.bfloat16)
    
    # One-hot encode the label
    oh_label = F.one_hot(label.long(), num_classes=logit.shape[-1]).to(torch.bfloat16)

    # Compute the gradient for the last layer
    delta = (prob - oh_label) / logit.size(0)
    
    # Now, compute the gradient for the penultimate layer
    grad_penultimate_layer = torch.matmul(delta, weight_last)  # Grad w.r.t. the output of the penultimate layer
    
    hidden_states = hidden_states_before_norm.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    norm_factor = torch.rsqrt(variance + norm_layer.variance_epsilon)
    
    grad_normalized_hidden_states  = norm_layer.weight * norm_factor * ( 1 - hidden_states * hidden_states / ((variance + norm_layer.variance_epsilon) * hidden_states.size(-1)) )
    
    grad = grad_penultimate_layer * grad_normalized_hidden_states.to(torch.bfloat16)
    
    # Compute the gradient w.r.t the weights of the penultimate layer using the input to that layer (penultimate input)
    # grad_weights_penultimate = torch.matmul(grad[:,:1].T, input_penultimate) # col # Backprop through penultimate
    # grad_weights_penultimate = torch.matmul(grad.T, input_penultimate[:,:1]) # row  # Backprop through penultimate
    grad_weights_penultimate = torch.matmul(grad.T, input_penultimate).mean(dim=-1)
    
    return grad_weights_penultimate.view(-1)

class LLaVATrainerOURS(LLaVATrainerTaskId):
    def __init__(self, task_id, ema_ratio=0.996, task_vector=None, fisher_old=None, fisher_freq=5, **kwargs):
        super(LLaVATrainerOURS, self).__init__(**kwargs)
        self.task_id = task_id
        self.ema_ratio = ema_ratio
        self.old_weights = {k: t.detach().clone() for k, t in self.model.named_parameters() if t.requires_grad}
        self.mu = 0.1
        
        self.prompt_ema_ratio = 0.99
        self.task_vector=task_vector.cuda() if task_vector is not None else None
        self.fisher_old = fisher_old #{k:p.cuda() for k, p in fisher_old.items()} if fisher_old is not None else None
        self.fisher_cur = 0
        self.fisher_cnt = 0
        self.fisher_freq = fisher_freq
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # if self.curr_round > 0:
        #     curr_weights = {k: t.detach().clone() for k, t in model.module.named_parameters() if t.requires_grad}
        #     with torch.no_grad():
        #         model.module.load_state_dict(self.old_weights, strict=False)
        #         # model.module.set_state('lora2')
        #         _, outputs = super(LLaVATrainerOURS, self).compute_loss(
        #                 model, inputs, return_outputs=True
        #             )
        #         outputs_target = outputs['logits']
        #         model.module.load_state_dict(curr_weights, strict=False)
                # model.module.set_state('gate')
            
            # with torch.no_grad():
            #     model.module.set_state('lora1')
            #     _, outputs = super(LLaVATrainerOURS, self).compute_loss(
            #             model, inputs, return_outputs=True
            #         )
            #     outputs_target = outputs['logits']
            #     model.module.set_state('lora2')
        loss, outputs = super(LLaVATrainerOURS, self).compute_loss(model, inputs, return_outputs=True)
        
        # l1 loss
        # l1_loss = 0
        # for layer_num in range(len(outputs['local_ia3_layer'])):
        #     l1_loss += torch.sum(torch.abs(outputs['local_ia3_layer'][layer_num])) / len(outputs['local_ia3_layer'])
        # loss += 0.001*l1_loss

        # loss_kl_1 = kl_loss(outputs['logits'], outputs_target.clone().detach())
        # loss_kl_1 = ((outputs['logits'] - outputs_target.detach())**2).mean()
        
        # if self.curr_round > 0:
        #     # loss += self.mu*kl_loss(outputs['logits'], outputs_target.detach())
        #     loss += self.mu * ((outputs['logits'] - outputs_target.detach())**2).mean()
            
        # accumulate task vector
        # layer_num = 30 # 0~31 or multiple
        # task_vector = outputs['ia3_layer'][layer_num].detach().mean(dim=0) -1
        # # task_vector = torch.cat(outputs['ia3_layer'][-3:], dim=0).detach().mean(dim=0) - 1
        
        # if self.task_vector is None:
        #     self.task_vector = task_vector
        # else:
        #     self.task_vector = self.prompt_ema_ratio * self.task_vector + (1-self.prompt_ema_ratio) * task_vector
        
        # pretrained weight fisher info
        

        return (loss, outputs) if return_outputs else loss
    
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        if self.args.auto_find_batch_size:
            if self.state.train_batch_size != self._train_batch_size:
                from accelerate.utils import release_memory

                (self.model_wrapped,) = release_memory(self.model_wrapped)
                self.model_wrapped = self.model

                # Check for DeepSpeed *after* the intial pass and modify the config
                if self.is_deepspeed_enabled:
                    # Temporarily unset `self.args.train_batch_size`
                    original_bs = self.args.per_device_train_batch_size
                    self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                    self.propagate_args_to_deepspeed(True)
                    self.args.per_device_train_batch_size = original_bs
            self.state.train_batch_size = self._train_batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        if self.is_fsdp_xla_v2_enabled:
            train_dataloader = tpu_spmd_dataloader(train_dataloader)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torchrun or torch.distributed.launch (deprecated))."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        # OURS:
        # if self.curr_round > 0:
        #     self.model.set_state('gate')
        # else:
        #     self.model.set_state('lora2')
        self.model.set_state('gate')
        self.model.activate_lora2()
        self.old_weights = {k: t.detach().clone() for k, t in self.model.named_parameters() if t.requires_grad}
        
        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        self.state.train_batch_size = self._train_batch_size

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self._fsdp_qlora_plugin_updates()
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # ckpt loading
        if resume_from_checkpoint is not None:
            if self.is_deepspeed_enabled:
                deepspeed_load_checkpoint(
                    self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                )
            elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)
        
        # FIXME
        if self.args.save_optim and self.curr_round > 0:
            output_dir = f'client_states_{self.args.note}/client_{self.client_id}/'
            self._load_optimizer_and_scheduler(output_dir)
        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            self.compare_trainer_and_checkpoint_args(self.args, self.state)
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()
        grad_norm: Optional[float] = None

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if self.args.include_num_input_tokens_seen:
                    main_input_name = getattr(self.model, "main_input_name", "input_ids")
                    if main_input_name not in inputs:
                        logger.warning(
                            "Tried to track the number of tokens seen, however the current model is "
                            "not configured properly to know what item is the input. To fix this, add "
                            "a `main_input_name` attribute to the model class you are using."
                        )
                    else:
                        input_device = inputs[main_input_name].device
                        self.state.num_input_tokens_seen += torch.sum(
                            self.accelerator.gather(
                                torch.tensor(inputs[main_input_name].numel(), device=input_device, dtype=torch.int64)
                            )
                        ).item()
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    if tr_loss.device != tr_loss_step.device:
                        raise ValueError(
                            f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                        )
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc:
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            _grad_norm = nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            _grad_norm = self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                        if (
                            is_accelerate_available()
                            and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                        ):
                            grad_norm = model.get_global_grad_norm()
                            # In some cases the grad norm may not return a float
                            if hasattr(grad_norm, "item"):
                                grad_norm = grad_norm.item()
                        else:
                            grad_norm = _grad_norm

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    
                    # compute fisher online
                    
                    if step % self.fisher_freq == 0:
                        # for p in self.model.base_model.model.model.layers[15].mlp.down_proj.base_layer.parameters():
                        #     p.requires_grad = True
                        with self.model.disable_adapter():
                            inputs = self._prepare_inputs(inputs)

                            output = self.model(**inputs)#.loss
                            
                            shift_labels = self.model.labels[..., 1:]
                            grads = get_grad_penultimate(output.logits[..., :-1, :][shift_labels != -100].detach(), shift_labels[shift_labels != -100].detach(), 
                                                        self.model.base_model.model.lm_head.weight,
                                                        output.penultimate_input[..., :-1, :][shift_labels != -100].detach(),
                                                        self.model.base_model.model.model.norm,
                                                        output.hidden_states_before_norm[..., :-1, :][shift_labels != -100].detach(),)
                            # output.loss.backward()
                            
                            # grads = []
                            # for p in self.model.base_model.model.model.layers[15].mlp.down_proj.base_layer.parameters():
                            #     grads.append(p.grad)#[:,0].view(-1))
                            #     # grads = p.grad
                            # grads = torch.cat(grads)
                            
                        # self.fisher_cur += (grads**2).detach()
                        self.fisher_cur += (grads).detach()
                        self.fisher_cnt += 1
                        
                        # for p in self.model.base_model.model.model.layers[15].mlp.down_proj.base_layer.parameters():
                        #     p.requires_grad = False
                        
                        model.zero_grad()
                    
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                    
                    # wsd
                    if self.args.is_wsd == 'WSD' and math.ceil(self.state.epoch*steps_in_epoch) == math.ceil(self.args.decay_ratio*steps_in_epoch):
                        self.global_weight = {k: t.detach().cpu().clone() for k, t in self.model.named_parameters() if t.requires_grad}
                    
                    # save client model
                    if step % 5 == 0:
                        output_dir = os.path.join(self.args.state_dir, f"{self.client_id}_client_model_round{self.curr_round+1}_itr{step}.pth")
                        self.model.activate_all()
                        state_dict = {k: t.detach().cpu().clone() for k, t in self.model.named_parameters() if t.requires_grad}
                        
                        if (self.args.local_rank == 0 or self.args.local_rank == -1):
                            torch.save(state_dict, output_dir)
                        
                        self.model.activate_lora2()
                    # ema update
                    # if optimizer_was_run and self.curr_round > 0:
                    #     with torch.no_grad():
                    #         cur_weights = {k: t for k, t in self.model.named_parameters() if t.requires_grad}
                    #         for name, param in self.old_weights.items():
                    #             self.old_weights[name].copy_(self.ema_ratio*param + (1-self.ema_ratio)*cur_weights[name])
                    
                    # global ema update
                    # if optimizer_was_run and self.curr_round > 0:
                    #     with torch.no_grad():
                    #         model_params = OrderedDict(self.model.named_parameters())
                    #         for name, param in model_params.items():
                    #             if 'lang_prompt_downsample_1' in name:
                    #                 target_name = name.replace('lang_prompt_downsample_1', 'lang_prompt_downsample_2')
                    #                 model_params[name].copy_(self.ema_ratio*param + (1-self.ema_ratio)*model_params[target_name])
                    #             elif 'ia3_generator_1' in name:
                    #                 target_name = name.replace('ia3_generator_1', 'ia3_generator_2')
                    #                 model_params[name].copy_(self.ema_ratio*param + (1-self.ema_ratio)*model_params[target_name])
                    
                    self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    # PyTorch/XLA relies on the data loader to insert the mark_step for
                    # each step. Since we are breaking the loop early, we need to manually
                    # insert the mark_step here.
                    if is_torch_xla_available():
                        xm.mark_step()
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_xla_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
        train_loss = self._total_loss_scalar / effective_global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        # remove momentum for l2p before saving
        if 'L2P' in self.args.mode and self.task_id is not None:
            for key in self.optimizer.state.keys():
                if 'exp_avg' not in self.optimizer.state[key]:
                    continue
                self.optimizer.state[key]['exp_avg'][self.optimizer.state[key]['exp_avg']!=0] = 0.0
                self.optimizer.state[key]['exp_avg_sq'][self.optimizer.state[key]['exp_avg_sq']!=0] = 0.0


        if self.args.save_optim:
            output_dir = f'client_states_{self.args.note}/client_{self.client_id}/'
            self._save_optimizer_and_scheduler(output_dir)

        # Optimal Transport Combination within client
        # if self.curr_round > 0:
        #     # self.model.load_state_dict(self.old_weights, strict=False)
            
        #     # ema = 0.5
        #     # with torch.no_grad():
        #     #     model_params = OrderedDict(self.model.named_parameters())
        #     #     for name, param in self.old_weights.items():
        #     #         model_params[name].copy_(ema*param + (1-ema)*model_params[name])
                    
        #     with torch.no_grad():
        #         model_params = OrderedDict(self.model.named_parameters())
        #         for name, param in self.old_weights.items():
        #             fisher_curr = safe_get_full_optimizer_state(model_params[name], "exp_avg_sq")
        #             fisher_old = self.fisher_old[name]
                    
        #             # normalize
        #             fisher_curr_norm = (fisher_curr - fisher_curr.min()) / (fisher_curr.max() - fisher_curr.min())
        #             fisher_old_norm = (fisher_old - fisher_old.min()) / (fisher_old.max() - fisher_old.min())
                    
        #             ratio = fisher_old_norm / (fisher_old_norm + fisher_curr_norm + 1e-9)
        #             model_params[name].copy_(ratio*param + (1-ratio)*model_params[name])
                    
        #             self.fisher_old[name] = fisher_curr
        # else:
        #     self.fisher_old = {}
        #     for name, param in self.model.named_parameters():
        #         if param.requires_grad:
        #             self.fisher_old[name] = safe_get_full_optimizer_state(param, "exp_avg_sq")
        
    

        # for name, param in self.model.named_parameters():
        #     # if param.requires_grad:
        #     if name == 'base_model.model.model.layers.31.lang_prompt_downsample_2.oproj.weight':
        #         self.task_vector = param #safe_get_full_optimizer_state(param, "exp_avg") #_sq
        # self.task_vector = self.task_vector.flatten().detach().clone().cpu()
        
        # if self.curr_round > 0:
        #     with torch.no_grad():
        #         model_params = OrderedDict(self.model.named_parameters())
        #         for name, param in model_params.items():
        #             if 'lang_prompt_downsample_2' in name:
        #                 target_name = name.replace('lang_prompt_downsample_2', 'lang_prompt_downsample_1')
        #                 model_params[name].copy_(0.5*param + 0.5*model_params[target_name])
        #             elif 'ia3_generator_2' in name:
        #                 target_name = name.replace('ia3_generator_2', 'ia3_generator_1')
        #                 model_params[name].copy_(0.5*param + 0.5*model_params[target_name])
        #             elif 'lang_prompt_norm_2' in name:
        #                 target_name = name.replace('lang_prompt_norm_2', 'lang_prompt_norm_1')
        #                 model_params[name].copy_(0.5*param + 0.5*model_params[target_name])
        #             elif 'ia3_l_2' in name:
        #                 target_name = name.replace('ia3_l_2', 'ia3_l_1')
        #                 model_params[name].copy_(0.5*param + 0.5*model_params[target_name])
        
        # self.task_vector = self.task_vector.detach().cpu()
        # k_proj_task_vector = torch.stack([layer.self_attn.k_proj.ia3_l_2['default'].detach().flatten() for layer in self.model.base_model.model.model.layers]).mean(dim=0) - 1
        # v_proj_task_vector = torch.stack([layer.self_attn.v_proj.ia3_l_2['default'].detach().flatten() for layer in self.model.base_model.model.model.layers]).mean(dim=0) - 1
        # down_proj_task_vector = torch.stack([layer.mlp.down_proj.ia3_l_2['default'].detach().flatten() for layer in self.model.base_model.model.model.layers]).mean(dim=0) - 1
        # self.task_vector = torch.cat((k_proj_task_vector,v_proj_task_vector, down_proj_task_vector)).cpu()
        
        # self.task_vector = torch.cat((self.model.base_model.model.model.layers[-1].self_attn.k_proj.ia3_l_2['default'].detach().flatten(),
        #                               self.model.base_model.model.model.layers[-1].self_attn.v_proj.ia3_l_2['default'].detach().flatten(),
        #                               self.model.base_model.model.model.layers[-1].mlp.down_proj.ia3_l_2['default'].detach().flatten())).cpu() - 1
        
        self.fisher_old = ((self.fisher_cur/self.fisher_cnt) + self.fisher_old) / 2 if self.fisher_old is not None else (self.fisher_cur/self.fisher_cnt)
        self.task_vector = self.fisher_old.detach().cpu()
        
        self.model.activate_all()

        return TrainOutput(self.state.global_step, train_loss, metrics)
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            # decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer
    
    # only load deepspeeed optimizer
    def _load_optimizer_and_scheduler(self, checkpoint):
        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        if self.is_deepspeed_enabled:
            from deepspeed.runtime.state_dict_factory import SDLoaderFactory
            from deepspeed.runtime.pipe.module import PipelineModule
            latest_tag = "latest"
            latest_path = os.path.join(checkpoint, latest_tag)
            if os.path.isfile(latest_path):
                with open(latest_path, "r") as fd:
                    tag = fd.read().strip()
                    
            ckpt_list = self.model_wrapped._get_all_ckpt_names(checkpoint, tag)
            sd_loader = SDLoaderFactory.get_sd_loader(ckpt_list, checkpoint_engine=self.model_wrapped.checkpoint_engine)

            is_pipe_parallel = isinstance(self.model_wrapped.module, PipelineModule)

            mp_rank = 0 if self.model_wrapped.mpu is None else self.model_wrapped.mpu.get_model_parallel_rank()
            load_path, checkpoint_state, _ = sd_loader.load(self.model_wrapped.mp_world_size, mp_rank, is_pipe_parallel=is_pipe_parallel)
            self.model_wrapped.loaded_checkpoint_dp_world_size = checkpoint_state['dp_world_size']
            self.model_wrapped.loaded_checkpoint_mp_world_size = checkpoint_state['mp_world_size']
            
            zero_sd_list = self.model_wrapped._get_all_zero_checkpoints(checkpoint, tag)
            self.model_wrapped.optimizer.load_state_dict(state_dict_list=zero_sd_list,
                                       load_optimizer_states=True,
                                       load_from_fp32_weights=self.model_wrapped.zero_load_from_fp32_weights(),
                                       checkpoint_folder=None,
                                       load_serial=None)
            
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            if not isinstance(self.lr_scheduler, DeepSpeedSchedulerWrapper):
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
                reissue_pt_warnings(caught_warnings)
            return

        else:
            super()._load_optimizer_and_scheduler(checkpoint)