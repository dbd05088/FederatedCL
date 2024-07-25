from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb, LlamaFlashAttention2
import torch
from typing import Optional, Tuple
from transformers.utils import (
    is_flash_attn_greater_or_equal_2_10,
    logging
)
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from torch import nn
import math
from functools import reduce
from operator import mul

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


class prefix_attention(LlamaFlashAttention2):
    def __init__(self, prefix_num=1, hidden_size=4096, head_dim=256, attn_dropout=0.0, attn_bias=False):
        (nn.Module).__init__(self)

        self.attention_dropout = attn_dropout
        self.hidden_size = hidden_size
        self.num_heads = 1
        self.head_dim = head_dim
        self.num_key_value_heads = 1
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = False

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=attn_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=attn_bias)
        self.o_proj = nn.Linear(self.head_dim, self.hidden_size, bias=attn_bias)
        
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
        
        self.prefix_num = prefix_num
        self.prefix_embedding = nn.Parameter(torch.zeros(self.prefix_num, self.hidden_size))
        val = math.sqrt(6. / float(3 * reduce(mul, (hidden_size,), 1)))
        # nn.init.uniform_(self.lang_prompt_dap_key_embeddings.data, -val, val)
        with torch.no_grad():
            self.prefix_embedding.uniform_(-val, val)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        
        combined = torch.concat((self.prefix_embedding.repeat(bsz, 1, 1), hidden_states), dim=1)
        if attention_mask is not None:
            attention_mask = torch.concat((torch.full((bsz, self.prefix_num), True).cuda(), attention_mask), dim=1)
        
        bsz, q_len, _ = combined.size()

        query_states = self.q_proj(combined)
        key_states = self.k_proj(combined)
        value_states = self.v_proj(combined)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        attn_output = attn_output.reshape(bsz, q_len, self.head_dim).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output[:,:self.prefix_num,:]


