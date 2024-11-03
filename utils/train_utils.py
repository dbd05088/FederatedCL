import torch
import os
import logging
import transformers
from models.llava.language_model.llava_llama import LlavaLlamaForCausalLM
from models.llava.language_model.llava_mpt import LlavaMptForCausalLM
from models.bunny import BunnyPhiForCausalLM, BunnyStableLMForCausalLM, BunnyQwen2ForCausalLM, BunnyMiniCPMForCausalLM, BunnyLlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPTextModel, CLIPProcessor, CLIPModel
import models.llava.conversation as conversation_lib_llava
import models.bunny.conversation as conversation_lib_bunny
from peft.tuners.lora import LoraLayer

from models.llava.ditto import LlavaLlamaFordittoCausalLM
from models.llava.llava_fedsim import FEDSIMLlavaLlamaForCausalLM
from models.llava.ours_ia3 import LlavaLlamaOURSGENIA3ForCausalLM
from models.llava.ours_ia3_2 import LlavaLlamaOURSGENIA3ForCausalLM2

from models.llava.L2P2 import LlavaLlamaForL2PIA3CausalLM2
from models.llava.L2P_T2 import LlavaLlamaForL2PTIA3CausalLM2
from models.llava.L2P2_Dual import LlavaLlamaForL2PIA3DualCausalLM2
from models.llava.L2P_T2_Dual import LlavaLlamaForL2PTIA3DualCausalLM2

from models.llava.DAP import LlavaLlamaDAPForCausalLM
from models.llava.DAP_T import LlavaLlamaDAPTForCausalLM
from models.llava.DAP_Dual import LlavaLlamaDAPDualForCausalLM
from models.llava.DAP_T_Dual import LlavaLlamaDAPTDualForCausalLM

from models.llava.EvoPrompt import LlavaLlamaEVOIA3ForCausalLM
from models.llava.EvoPrompt_T import LlavaLlamaEVOTIA3ForCausalLM

from models.llava.CodaPrompt import LlavaLlamaForCodaIA3CausalLM
from models.llava.CodaPrompt_T import LlavaLlamaForCodaIA3TCausalLM
from models.llava.CodaPrompt_Dual import LlavaLlamaForCodaIA3DualCausalLM
from models.llava.CodaPrompt_T_Dual import LlavaLlamaForCodaIA3TDualCausalLM

from models.llava.LAE import LlavaLlamaForLAEIA3CausalLM
from models.llava.LAE_Dual import LlavaLlamaForDualLAEIA3CausalLM

import copy
ACCESS_TOKEN = "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"

def get_VLMmodel(model_args, training_args, bnb_model_from_pretrained_args, data_args):
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    attn_implementation = "flash_attention_2"
    assert model_args.vision_tower is not None
    
    if 'dap' in training_args.mode or 'l2p' in training_args.mode:
        assert training_args.lora_enable == False, "no lora in pFedPG and feddat"
    if training_args.mode == 'feddat':
        assert training_args.gradient_accumulation_steps == 1
    
    if "ia3" in training_args.mode:
        assert training_args.ia3_enable == True
    
    # load tokenizer
    # for llava
    if model_args.model_type == "mpt":
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    elif model_args.model_type == 'llama': 
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    
    # for bunny
    elif (
        model_args.model_type == 'phi-1.5' or model_args.model_type == 'phi-2'
            or model_args.model_type == 'qwen1.5-1.8b' or model_args.model_type == 'minicpm'):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    elif model_args.model_type == 'llama3-8b':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            token=ACCESS_TOKEN
        )
    elif model_args.model_type == 'stablelm-2':
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=True,
            trust_remote_code=True
        )

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token
    
    if model_args.model_type == 'llama3-8b':
        tokenizer.pad_token = tokenizer.eos_token
        
    if training_args.is_eval:
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    
    if 'llava' in model_args.model_name_or_path.lower():
        
        #############################################################################
        if training_args.mode in ['L2P2', 'L2P2_FedAvg', 'L2P2_FedPer']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaForL2PIA3CausalLM2.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                pool_size = training_args.pool_size,
                prompt_top_k = training_args.prompt_top_k,
                **bnb_model_from_pretrained_args
            )
            print('load L2P-IA3')
        elif training_args.mode in ['L2P_T2', 'L2P_T2_FedAvg', 'L2P_T2_FedPer']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaForL2PTIA3CausalLM2.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                pool_size = training_args.pool_size,
                prompt_top_k = training_args.prompt_top_k,
                **bnb_model_from_pretrained_args
            )
            print('load L2P_T-IA3')
        elif training_args.mode in ['L2P2_FedDAT', 'L2P2_Ditto']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaForL2PIA3DualCausalLM2.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                pool_size = training_args.pool_size,
                prompt_top_k = training_args.prompt_top_k,
                **bnb_model_from_pretrained_args
            )
            print('load L2P_Dual-IA3')
        elif training_args.mode in ['L2P_T2_FedDAT', 'L2P_T2_Ditto', 'L2P_T2_fedours']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaForL2PTIA3DualCausalLM2.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                pool_size = training_args.pool_size,
                prompt_top_k = training_args.prompt_top_k,
                **bnb_model_from_pretrained_args
            )
            print('load L2P_T_Dual-IA3')
        
        #############################################################################
        elif training_args.mode in ['CodaPrompt', 'CodaPrompt_FedAvg', 'CodaPrompt_FedPer']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaForCodaIA3CausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                pool_size = training_args.pool_size,
                prompt_top_k = training_args.prompt_top_k,
                **bnb_model_from_pretrained_args
            )
            print('load CodaPrompt-IA3')
        elif training_args.mode in ['CodaPrompt_T', 'CodaPrompt_T_FedAvg', 'CodaPrompt_T_FedPer']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaForCodaIA3TCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                pool_size = training_args.pool_size,
                prompt_top_k = training_args.prompt_top_k,
                **bnb_model_from_pretrained_args
            )
            print('load CodaPrompt_T-IA3')
        elif training_args.mode in ['CodaPrompt_FedDAT', 'CodaPrompt_Ditto']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaForCodaIA3DualCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                pool_size = training_args.pool_size,
                prompt_top_k = training_args.prompt_top_k,
                **bnb_model_from_pretrained_args
            )
            print('load CodaPrompt_Dual-IA3')
        elif training_args.mode in ['CodaPrompt_T_FedDAT', 'CodaPrompt_T_Ditto']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaForCodaIA3TDualCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                pool_size = training_args.pool_size,
                prompt_top_k = training_args.prompt_top_k,
                **bnb_model_from_pretrained_args
            )
            print('load CodaPrompt_T_Dual-IA3')
        
        #############################################################################
        elif training_args.mode in ['DAP', 'DAP_FedAvg', 'DAP_FedPer']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaDAPForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                key_embed_size=training_args.key_embed_size,
                generator_hidden_feature=training_args.generator_hidden_feature,
                **bnb_model_from_pretrained_args
            )
            print('load DAP-IA3')
        elif training_args.mode in ['DAP_T', 'DAP_T_FedAvg', 'DAP_T_FedPer']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaDAPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                key_embed_size=training_args.key_embed_size,
                generator_hidden_feature=training_args.generator_hidden_feature,
                **bnb_model_from_pretrained_args
            )
            print('load DAP_T-IA3')
        elif training_args.mode in ['DAP_FedDAT', 'DAP_Ditto']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaDAPDualForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                key_embed_size=training_args.key_embed_size,
                generator_hidden_feature=training_args.generator_hidden_feature,
                **bnb_model_from_pretrained_args
            )
        elif training_args.mode in ['DAP_T_FedDAT', 'DAP_T_Ditto']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaDAPTDualForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                key_embed_size=training_args.key_embed_size,
                generator_hidden_feature=training_args.generator_hidden_feature,
                **bnb_model_from_pretrained_args
            )
        
        #############################################################################
        elif training_args.mode in ['EvoPrompt', 'EvoPrompt_FedAvg', 'EvoPrompt_FedPer']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaEVOIA3ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            print('load EvoPrompt-IA3')
        elif training_args.mode in ['EvoPrompt_T', 'EvoPrompt_T_FedAvg', 'EvoPrompt_T_FedPer']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaEVOTIA3ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            print('load EvoPrompt_T-IA3')
        elif training_args.mode in ['EvoPrompt_FedDAT', 'EvoPrompt_Ditto']:
            assert model_args.model_type != 'mpt'
            pass
        elif training_args.mode in ['EvoPrompt_T_FedDAT', 'EvoPrompt_T_Ditto']:
            assert model_args.model_type != 'mpt'
            pass
        
        #############################################################################
        elif training_args.mode in ['LAE', 'LAE_FedAvg', 'LAE_FedPer']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaForLAEIA3CausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            print('load LAE-IA3')
        elif training_args.mode in ['LAE_FedDAT', 'LAE_Ditto']:
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaForDualLAEIA3CausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            print('load Dual LAE-IA3')
        #############################################################################
        elif training_args.mode == 'ours_generator':
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaOURSGENIA3ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                generator_output_size=training_args.generator_output_size,
                generator_hidden_dim=training_args.generator_hidden_dim,
                **bnb_model_from_pretrained_args
            )
            print('load ours generator')
        elif training_args.mode == 'ours_generator2':
            assert model_args.model_type != 'mpt'
            model = LlavaLlamaOURSGENIA3ForCausalLM2.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                generator_hidden_dim=training_args.generator_hidden_dim,
                **bnb_model_from_pretrained_args
            )
            print('load ours generator2')
        ################################################################################
        elif 'ditto' == training_args.mode or 'feddat' == training_args.mode or 'fedours' == training_args.mode:
            model = LlavaLlamaFordittoCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        elif 'fedsim' == training_args.mode:
            model = FEDSIMLlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
    
    elif 'bunny' in model_args.model_name_or_path.lower():
        if model_args.model_type == 'phi-1.5' or model_args.model_type == 'phi-2':
            model = BunnyPhiForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_type == 'stablelm-2':
            model = BunnyStableLMForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_type == 'qwen1.5-1.8b':
            model = BunnyQwen2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_type == 'minicpm':
            model = BunnyMiniCPMForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        elif model_args.model_type == 'llama3-8b':
            model = BunnyLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                token = ACCESS_TOKEN,
                **bnb_model_from_pretrained_args
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")    

    model.config.use_cache = False
    model.model.requires_grad_(False)

    # FIXME
    if training_args.bits >= 16:
        model = model.to(training_args.device)
    
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        
        target_modules = ['k_proj', 'v_proj']
        
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=target_modules,#find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        if training_args.mode in ['fedsim', 'apfl', 'ditto', 'fedours'] or training_args.mode =='feddat':
            from models.duallora.dualloramodel import DualLoraModel
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALLORA'] = DualLoraModel
            lora_config.peft_type = 'DUALLORA'
        
        # rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    
    elif training_args.ia3_enable:
        from peft import IA3Config, get_peft_model
        ia3_config = IA3Config(
            target_modules=["k_proj", "v_proj", "down_proj"], 
            feedforward_modules=["down_proj"],
            task_type="CAUSAL_LM",
        )
        
        # create pool
        if training_args.mode in ['fedsim', 'apfl', 'ditto', 'fedours'] or training_args.mode =='feddat':
            from models.dual_ia3.dual_ia3_model import DualIA3Model
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALIA3'] = DualIA3Model
            ia3_config.peft_type = 'DUALIA3'
        
        elif 'L2P' in training_args.mode or 'DAP' in training_args.mode or 'CodaPrompt' in training_args.mode:
            from models.empty_ia3.empty_ia3_model import EmptyIA3Model
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['EMPTYIA3'] = EmptyIA3Model
            ia3_config.peft_type = 'EMPTYIA3'
        
        elif training_args.mode in ['LAE', 'LAE_FedAvg', 'LAE_FedPer']:
            from models.lae_ia3.lae_ia3_model import LAEIA3Model
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['LAEIA3'] = LAEIA3Model
            ia3_config.peft_type = 'LAEIA3'
        
        elif training_args.mode in ['LAE_FedDAT', 'LAE_Ditto']:
            from models.lae_ia3_dual.dual_lae_ia3_model import DualLAEIA3Model
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['DUALLAEIA3'] = DualLAEIA3Model
            ia3_config.peft_type = 'DUALLAEIA3'
        
        elif training_args.mode in ['EvoPrompt']:
            from models.evo_ia3.evoia3model import EVOIA3Model
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['EVOIA3'] = EVOIA3Model
            ia3_config.peft_type = 'EVOIA3'
            ia3_config.generator_output_size = 1024
            ia3_config.generator_hidden_feature = training_args.generator_hidden_feature
        elif training_args.mode in ['EvoPrompt_T']:
            from models.evo_ia3.evoia3model import EVOIA3Model
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['EVOIA3'] = EVOIA3Model
            ia3_config.peft_type = 'EVOIA3'
            ia3_config.generator_output_size = 768
            ia3_config.generator_hidden_feature = training_args.generator_hidden_feature

        elif training_args.mode in ['ours_generator']:
            from models.ours_ia3.ours_ia3_model import DualEVOIA3Model
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['OURSGEN'] = DualEVOIA3Model
            ia3_config.peft_type = 'OURSGEN'
            ia3_config.generator_output_size = training_args.generator_output_size
            ia3_config.generator_hidden_feature = training_args.generator_hidden_feature
        elif training_args.mode in ['ours_generator2', 'ours_generator3', 'ours_generator4']:
            from models.ours_ia3_2.ours_ia3_model2 import DualEVOIA3Model2
            from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING
            PEFT_TYPE_TO_MODEL_MAPPING['OURSGEN2'] = DualEVOIA3Model2
            ia3_config.peft_type = 'OURSGEN2'
        
        model = get_peft_model(model, ia3_config)
        model = model.to(device=training_args.device, dtype=compute_dtype)

    if 'llava' in model_args.model_name_or_path.lower():
        if model_args.version in conversation_lib_llava.conv_templates:
            conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates[model_args.version]
        else:
            conversation_lib_llava.default_conversation = conversation_lib_llava.conv_templates["vicuna_v1"]
            
    elif 'bunny' in model_args.model_name_or_path.lower():
        if model_args.version in conversation_lib_bunny.conv_templates:
            conversation_lib_bunny.default_conversation = conversation_lib_bunny.conv_templates[model_args.version]
        else:
            conversation_lib_bunny.default_conversation = conversation_lib_bunny.conv_templates["default"]

    # load vision tower
    # if model_args.vision_tower is not None:
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        # fsdp=training_args.fsdp
    )

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    # vision_tower.requires_grad_(True)
    if training_args.mode == 'ours_generator' or 'L2P' in training_args.mode or 'CodaPrompt' in training_args.mode or 'DAP' in training_args.mode or 'EvoPrompt' in training_args.mode:
        vision_tower.select_feature = 'cls_patch'
    
        if '_T' in training_args.mode:
            model.base_model.model.clip_encoder = CLIPModel.from_pretrained(model_args.vision_tower).to(device=training_args.device, dtype=compute_dtype)
            
            model.base_model.model.clipprocessor = CLIPProcessor.from_pretrained("/home/vision/thkim/FederatedCL/models/clip_models/clipprocessor/")

            model.base_model.model.clip_encoder.requires_grad_(False)

    data_args.image_processor = vision_tower.image_processor
    
    data_args.is_multimodal = True

    model.config.image_aspect_ratio = "pad" #data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length
    
    if 'L2P' in training_args.mode or 'CodaPrompt' in training_args.mode or 'DAP' in training_args.mode or 'EvoPrompt' in training_args.mode or 'LAE' in training_args.mode:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
        model.lm_head.requires_grad_(False)
        for n, p in model.named_parameters():
            if 'lang_prompt' in n :
                p.requires_grad_(True)
        
        if 'FedDAT' in training_args.mode or 'Ditto' in training_args.mode or 'fedours' in training_args.mode:
            model.set_state(training_args.set_state)
    
    elif training_args.mode in [ 'fedsim', 'ditto', 'apfl', 'feddat', 'fedours']:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
        
        model.set_state(training_args.set_state)
        model.activate_all()
        model.lm_head.requires_grad_(False)
    
    elif 'ia3' in training_args.mode:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
        model.lm_head.requires_grad_(False)
        for n, p in model.named_parameters():
            if 'lang_prompt' in n :
                p.requires_grad_(True)
                
    elif training_args.mode == 'ours_generator' or training_args.mode == 'ours_generator2' or training_args.mode == 'ours_generator3' or training_args.mode == 'ours_generator4':
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
        model.lm_head.requires_grad_(False)
        for n, p in model.named_parameters():
            if 'lang_prompt' in n :
                p.requires_grad_(True)
        model.set_state(training_args.set_state)
        model.activate_all()
        
        if training_args.mode == 'ours_generator2':
            for layer in model.base_model.model.model.layers:
                layer.lang_prompt_downsample_1.oproj.weight.data.fill_(0)
                layer.lang_prompt_downsample_2.oproj.weight.data.fill_(0)
        
    else:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False
        model.lm_head.requires_grad_(False)
    
    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    
    if 'llava' in model_args.model_name_or_path.lower():
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer)or isinstance(module, torch.nn.LayerNorm):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    total_count = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(n, p.shape)
            total_count += p.numel()
    print(total_count)
    return model, tokenizer, data_args

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

def get_decay_parameter_names(model):
    """
    Get all parameter names that weight decay will be applied to

    Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
    apply to those modules since this function only filter out instance of nn.LayerNorm
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters



# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

from torch import nn

def load_deepspeed(state_dict, module: nn.Module, prefix="", strict=True):
    import deepspeed
    # because zero3 puts placeholders in model params, this context
    # manager gathers (unpartitions) the params of the current layer, then loads from
    # the state dict and then re-partitions them again
    with deepspeed.zero.GatheredParameters(list(module.parameters(recurse=False)), modifier_rank=0):
        if deepspeed.comm.get_rank() == 0:
            module._load_from_state_dict(state_dict, prefix, {}, strict, [], [], [])
            # module.load_state_dict(state_dict, strict=strict)

    for name, child in module._modules.items():
        if child is not None:
            load_deepspeed(state_dict, child, prefix + name + ".")

import random
from federated_methods.fedours import fedours_ema_distill_create_trainer
def get_task_vectors(model, tokenizer, train_datalists, training_args, data_args, global_state_dict, make_supervised_data_module):
    random.seed(training_args.seed)
    client_task_vectors = []
    for client_id in range(len(train_datalists)):
        datalist = train_datalists[client_id][0]['datalist']
        
        sub_datalist = random.sample(datalist, 4*20)
        
        data_module = make_supervised_data_module(client_data=sub_datalist, # sub_dataset
                                                tokenizer=tokenizer,
                                                data_args=copy.deepcopy(data_args))
    
        extra_state_dict_dict = {}
        extra_state_dict_dict['client_id']=0
        extra_state_dict_dict['curr_round']=0
        extra_state_dict_dict['fisher_freq'] = 1
        trainer = fedours_ema_distill_create_trainer(model, tokenizer, training_args, data_module, extra_state_dict_dict)

        results = trainer.train()
        
        task_vector = trainer.task_vector
        
        client_task_vectors.append(task_vector)
        
        trainer.deepspeed.empty_partition_cache()
        del trainer
        
        with torch.no_grad():
            model.load_state_dict(global_state_dict, strict=False)
    
    extra_state_dict_dict['fisher_freq']=5
    return client_task_vectors