from transformers import CLIPTextModel
import torch

# Load the models
cliptext_model = CLIPTextModel.from_pretrained("/home/vision/thkim/FederatedCL/models/clip_models/text_encoder/").cuda()
state_dict = torch.load("/home/vision/thkim/FederatedCL/models/clip_models/longclip-B.pt", map_location='cpu')

# Create a new state dict for the transformed keys
new_state_dict = {}

# Define key mapping
key_mapping = {
    "token_embedding.weight": "text_model.embeddings.token_embedding.weight",
    "positional_embedding": "text_model.embeddings.position_embedding.weight",
    "transformer.resblocks.": "text_model.encoder.layers.",
    "ln_final.": "text_model.final_layer_norm.",
    "attn.in_proj_": "self_attn.in_proj_",
    "attn.out_proj": "self_attn.out_proj",
    "mlp.c_fc": "mlp.fc1",
    "mlp.c_proj": "mlp.fc2",
    "ln_1": "layer_norm1",
    "ln_2": "layer_norm2"
}

# Transform keys
for old_key, value in state_dict.items():
    if old_key.startswith(("visual.", "text_projection", "logit_scale", "positional_embedding_res")):
        continue  # Skip visual and other irrelevant keys

    new_key = old_key
    for old_pattern, new_pattern in key_mapping.items():
        if old_pattern in new_key:
            new_key = new_key.replace(old_pattern, new_pattern)
    # Handle special cases for attention layers
    if "in_proj_weight" in new_key:
        q, k, v = torch.chunk(value, 3)
        new_state_dict[new_key.replace("in_proj_weight", "q_proj.weight")] = q
        new_state_dict[new_key.replace("in_proj_weight", "k_proj.weight")] = k
        new_state_dict[new_key.replace("in_proj_weight", "v_proj.weight")] = v
    elif "in_proj_bias" in new_key:
        q, k, v = torch.chunk(value, 3)
        new_state_dict[new_key.replace("in_proj_bias", "q_proj.bias")] = q
        new_state_dict[new_key.replace("in_proj_bias", "k_proj.bias")] = k
        new_state_dict[new_key.replace("in_proj_bias", "v_proj.bias")] = v
    else:
        new_state_dict[new_key] = value

# Check if all required keys are present
required_keys = [
    'text_model.embeddings.token_embedding.weight',
    'text_model.embeddings.position_embedding.weight',
    'text_model.final_layer_norm.weight',
    'text_model.final_layer_norm.bias'
]

for i in range(12):  # Assuming 12 layers
    layer_prefix = f'text_model.encoder.layers.{i}.'
    required_keys.extend([
        f'{layer_prefix}self_attn.k_proj.weight',
        f'{layer_prefix}self_attn.k_proj.bias',
        f'{layer_prefix}self_attn.v_proj.weight',
        f'{layer_prefix}self_attn.v_proj.bias',
        f'{layer_prefix}self_attn.q_proj.weight',
        f'{layer_prefix}self_attn.q_proj.bias',
        f'{layer_prefix}self_attn.out_proj.weight',
        f'{layer_prefix}self_attn.out_proj.bias',
        f'{layer_prefix}layer_norm1.weight',
        f'{layer_prefix}layer_norm1.bias',
        f'{layer_prefix}mlp.fc1.weight',
        f'{layer_prefix}mlp.fc1.bias',
        f'{layer_prefix}mlp.fc2.weight',
        f'{layer_prefix}mlp.fc2.bias',
        f'{layer_prefix}layer_norm2.weight',
        f'{layer_prefix}layer_norm2.bias',
    ])

missing_keys = set(required_keys) - set(new_state_dict.keys())
unexpected_keys = set(new_state_dict.keys()) - set(required_keys)

if missing_keys:
    print("Missing keys:", missing_keys)
if unexpected_keys:
    print("Unexpected keys:", unexpected_keys)

torch.save(new_state_dict, 'longclip_B_text.pt')

# Load the new state dict
cliptext_model.text_model.embeddings.position_embedding = torch.nn.Embedding(248, 768).cuda()

# Loading with strict=False allows loading even if some keys mismatch
cliptext_model.load_state_dict(new_state_dict, strict=True)

print("Model loaded successfully!")
