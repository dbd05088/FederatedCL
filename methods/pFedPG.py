from methods.cl_manager_server import CLManagerServer
from methods.cl_manager_client import CLManagerClient
from collections import OrderedDict
import torch
from torch import nn
import copy
import math
import torch.nn.functional as F
from transformers.trainer import (
    get_parameter_names,
    ALL_LAYERNORM_LAYERS,
)
from transformers import Trainer
import bitsandbytes

PROMPT_NUM = 100

class pFedPG_server(CLManagerServer):
    def setup(self):
        super().setup()
        
        self.delta_Ps = {}
        self.prompt_generator = Prompt_Generator(512, PROMPT_NUM, self.model.base_model.mm_projector[-1].out_features, self.num_clients).to(self.device)
        self.client_P = {}
        
        self.optimizer = torch.optim.SGD(self.prompt_generator.parameters(), lr=0.001)
    
    def server_msg(self, client_id=None):
        client_prompt = self.prompt_generator(client_id)
        self.client_P[client_id] = client_prompt[0].clone()
        
        # also aggregate and send mm_projector?
        return (client_prompt.detach().cpu(),) # mm_projector_weight
    
    def handle_msg_per_client(self, msg):
        assert isinstance(msg, tuple)
        
        # self.delta_Ps.append(msg)
        self.delta_Ps[msg[0]] = msg[1]
        
        # get mm_projector weight?
    
    def do_server_work(self, curr_round):
        num_clients = len(self.delta_Ps)
        if num_clients == 0:
            return
        assert len(self.delta_Ps) == len(self.client_P)
        loss = sum([torch.mm(self.client_P[client_id], (self.delta_Ps[client_id].to(self.device)).T).sum() for client_id in self.client_P.keys()])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.delta_Ps = {}
        self.client_P = {}
        
        # self.evaluate_seendata(curr_round)
        self.save_server_model(curr_round)
        
class pFedPG_client(CLManagerClient): 
    def setup(self):
        super().setup()
        
        self.init_prompt = None
        self.model.activate_prompt()
    
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
        
    def client_msg(self):
        updated_prompt = self.model.get_prompt()
        delta_prompt = updated_prompt.detach().cpu() - self.init_prompt.detach().cpu()
        
        # aggregate mm_projector?
        
        return (self.state['client_id'], delta_prompt[0],) # mm_projector_weight
    
    def handle_server_msg(self, server_msg):
        assert isinstance(server_msg, tuple)
        
        self.init_prompt = server_msg[0].clone()
        self.model.set_prompt(server_msg[0])

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