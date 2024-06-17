import torch
from utils.train_utils import load_deepspeed, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3
from transformers import TrainerCallback
from models.llava.llava_trainer import LLaVATrainer
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

PROMPT_NUM = 100

def pfedpg_set_state_dict(model, global_state_dict, local_state_dict_list, training_args):
    prompt_generator = Prompt_Generator(512, PROMPT_NUM, model.base_model.mm_projector[-1].out_features, training_args.num_clients).cuda()
    pgen_optim = torch.optim.SGD(prompt_generator.parameters(), lr=0.005)
    return {
        'prompt_generator': prompt_generator,
        'pgen_optim': pgen_optim,
        'init_prompt':{},
    }


def pfedpg_aggregate_state_dict(global_state_dict, local_state_dict_list, selected_ids, num_selection, training_args, **kwargs):
    init_prompt = kwargs.get('init_prompt')
    prompt_generator = kwargs.get('prompt_generator')
    pgen_optim = kwargs.get('pgen_optim')
    curr_round = kwargs.get('curr_round')
    
    with torch.no_grad():
        delta_prompt = {client:local_state_dict_list[client]['lang_prompt'].cuda() - init_prompt[client] for client in selected_ids}

    loss = sum([torch.mm(init_prompt[client][0], (delta_prompt[client][0]).T).sum() for client in selected_ids])
    pgen_optim.zero_grad()
    loss.backward()
    pgen_optim.step()
    
    # save client models in every 10 rounds
    # default: total 100 rounds, 20 iters per round
    if (curr_round+1) % 10 == 0 and (training_args.local_rank == 0 or training_args.local_rank == -1):
        for client_id in selected_ids:
            output_dir = os.path.join(training_args.state_dir, f"{client_id}_client_model_round{curr_round+1}.pth")
            new_prompt = prompt_generator(client_id)
            local_state_dict_list[client_id]['lang_prompt'].copy_(new_prompt.detach().cpu())
            torch.save(local_state_dict_list[client_id], output_dir)
    
def pfedpg_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict):
    client_id = extra_state_dict_dict['client_id']
    init_prompt = extra_state_dict_dict['prompt_generator'](client_id)
    extra_state_dict_dict['init_prompt'][client_id] = init_prompt.clone()
    trainer = LLaVATrainerPFEDPG(model=model,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        max_seq_length=training_args.model_max_length,
        **data_module,
        init_prompt=init_prompt
        )
    return trainer


class LLaVATrainerPFEDPG(LLaVATrainer):
    def __init__(self, init_prompt, **kwargs):
        super(LLaVATrainerPFEDPG, self).__init__(**kwargs)
        
        self.model.set_prompt(init_prompt)

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
                            p for n, p in opt_model.named_parameters() if ('_prompt' in n and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if ('_prompt' in n and p.requires_grad)
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
    

class Prompt_Generator(nn.Module):
    def __init__(self, d_model, prompt_num, d_embed, client_num):
        super().__init__()
        self.base_prompt = nn.Parameter(torch.zeros(1, prompt_num, d_embed))
        self.client_descriptor = nn.Parameter(torch.zeros(client_num, prompt_num, d_embed))
        self.attn = MultiHeadAttentionLayer(d_model, 1, d_embed)
        
        val = math.sqrt(6. / float(d_embed))  # noqa
        
        # xavier_uniform initialization
        nn.init.uniform_(self.base_prompt.data, -val, val)
        for prompt in self.client_descriptor:
            nn.init.uniform_(prompt.data, -val, val)
    
    def forward(self, descriptor_id):
        delta = self.attn(self.client_descriptor[descriptor_id].unsqueeze(0), self.base_prompt, self.base_prompt)
        return self.base_prompt + delta

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, h, d_embed):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_model = d_model
        self.h = h
        self.q_fc = nn.Linear(d_embed, d_model) # (d_embed, d_model)
        self.k_fc = nn.Linear(d_embed, d_model) # (d_embed, d_model)
        self.v_fc = nn.Linear(d_embed, d_model) # (d_embed, d_model)
        self.out_fc = nn.Linear(d_model, d_embed) # (d_model, d_embed)
    
    def forward(self, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)

        def transform(x, fc):  # (n_batch, seq_len, d_embed)
            out = fc(x)        # (n_batch, seq_len, d_model)
            out = out.view(n_batch, -1, self.h, self.d_model//self.h) # (n_batch, seq_len, h, d_k)
            out = out.transpose(1, 2) # (n_batch, h, seq_len, d_k)
            return out

        query = transform(query, self.q_fc) # (n_batch, h, seq_len, d_k)
        key = transform(key, self.k_fc)     # (n_batch, h, seq_len, d_k)
        value = transform(value, self.v_fc) # (n_batch, h, seq_len, d_k)

        out = self.calculate_attention(query, key, value, mask) # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2) # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model) # (n_batch, seq_len, d_model)
        out = self.out_fc(out) # (n_batch, seq_len, d_embed)
        return out
    
    def calculate_attention(self, query, key, value, mask):
        # query, key, value: (n_batch, h, seq_len, d_k)
        # mask: (n_batch, 1, seq_len, seq_len)
        d_k = key.shape[-1]
        attention_score = torch.matmul(query, key.transpose(-2, -1)) # Q x K^T, (n_batch, h, seq_len, seq_len)
        attention_score = attention_score / math.sqrt(d_k)
        if mask is not None:
            attention_score = attention_score.masked_fill(mask==0, -1e9)
        attention_prob = F.softmax(attention_score, dim=-1) # (n_batch, h, seq_len, seq_len)
        out = torch.matmul(attention_prob, value) # (n_batch, h, seq_len, d_k)
        return out