from models.llava.language_model.llava_llama import LlavaLlamaForCausalLM
import torch
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from models.duallora.dualloralayer import DualLoraLayer

class FEDSIMLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        self.labels = labels
        
        for name, module in self.named_modules():
            if isinstance(module, DualLoraLayer):
                module.set_state('lora1')
        self.model.model.mm_projector = self.model.model.global_mm_projector
        output1 = super().forward(
            input_ids=input_ids.clone(),
            attention_mask=attention_mask.clone(),
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds.clone(),
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )    
        # local forward
        for name, module in self.named_modules():
            if isinstance(module, DualLoraLayer):
                module.set_state('lora2')
        self.model.model.mm_projector = self.model.model.local_mm_projector
        output2 = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )      
        final_logits = output1['logits'] + output2['logits']
        
        output2['logits'] = final_logits
        self.model.model.mm_projector = None
        return output2