import torch

MODEL_BLOCK_DICT = {
    'resnet18': [['initial'],
                 ['group1.blocks.block0'],
                 ['group1.blocks.block1'],
                 ['group2.blocks.block0'],
                 ['group2.blocks.block1'],
                 ['group3.blocks.block0'],
                 ['group3.blocks.block1'],
                 ['group4.blocks.block0'],
                 ['group4.blocks.block1'],
                 ['pool', 'fc']],
    
    'resnet34': [['initial'],
                 ['group1.blocks.block0'],
                 ['group1.blocks.block1'],
                 ['group1.blocks.block2'],
                 ['group2.blocks.block0'],
                 ['group2.blocks.block1'],
                 ['group2.blocks.block2'],
                 ['group2.blocks.block3'],
                 ['group3.blocks.block0'],
                 ['group3.blocks.block1'],
                 ['group3.blocks.block2'],
                 ['group3.blocks.block3'],
                 ['group3.blocks.block4'],
                 ['group3.blocks.block5'],
                 ['group4.blocks.block0'],
                 ['group4.blocks.block1'],
                 ['group4.blocks.block2'],
                 ['pool', 'fc']],
    
    'resnet20': [['initial'],
                 ['group1.blocks.block0'],
                 ['group1.blocks.block1'],
                 ['group1.blocks.block2'],
                 ['group2.blocks.block0'],
                 ['group2.blocks.block1'],
                 ['group2.blocks.block2'],
                 ['group3.blocks.block0'],
                 ['group3.blocks.block1'],
                 ['group3.blocks.block2'],
                 ['pool', 'fc']],
    
    'resnet32': [['initial'],
                 ['group1.blocks.block0'],
                 ['group1.blocks.block1'],
                 ['group1.blocks.block2'],
                 ['group1.blocks.block3'],
                 ['group1.blocks.block4'],
                 ['group2.blocks.block0'],
                 ['group2.blocks.block1'],
                 ['group2.blocks.block2'],
                 ['group2.blocks.block3'],
                 ['group2.blocks.block4'],
                 ['group3.blocks.block0'],
                 ['group3.blocks.block1'],
                 ['group3.blocks.block2'],
                 ['group3.blocks.block3'],
                 ['group3.blocks.block4'],
                 ['pool', 'fc']],
    
    'resnet44': [['initial'],
                 ['group1.blocks.block0'],
                 ['group1.blocks.block1'],
                 ['group1.blocks.block2'],
                 ['group1.blocks.block3'],
                 ['group1.blocks.block4'],
                 ['group1.blocks.block5'],
                 ['group1.blocks.block6'],
                 ['group2.blocks.block0'],
                 ['group2.blocks.block1'],
                 ['group2.blocks.block2'],
                 ['group2.blocks.block3'],
                 ['group2.blocks.block4'],
                 ['group2.blocks.block5'],
                 ['group2.blocks.block6'],
                 ['group3.blocks.block0'],
                 ['group3.blocks.block1'],
                 ['group3.blocks.block2'],
                 ['group3.blocks.block3'],
                 ['group3.blocks.block4'],
                 ['group3.blocks.block5'],
                 ['group3.blocks.block6'],
                 ['pool', 'fc']],
    
    'resnet56': [['initial'],
                 ['group1.blocks.block0'],
                 ['group1.blocks.block1'],
                 ['group1.blocks.block2'],
                 ['group1.blocks.block3'],
                 ['group1.blocks.block4'],
                 ['group1.blocks.block5'],
                 ['group1.blocks.block6'],
                 ['group1.blocks.block7'],
                 ['group1.blocks.block8'],
                 ['group2.blocks.block0'],
                 ['group2.blocks.block1'],
                 ['group2.blocks.block2'],
                 ['group2.blocks.block3'],
                 ['group2.blocks.block4'],
                 ['group2.blocks.block5'],
                 ['group2.blocks.block6'],
                 ['group2.blocks.block7'],
                 ['group2.blocks.block8'],
                 ['group3.blocks.block0'],
                 ['group3.blocks.block1'],
                 ['group3.blocks.block2'],
                 ['group3.blocks.block3'],
                 ['group3.blocks.block4'],
                 ['group3.blocks.block5'],
                 ['group3.blocks.block6'],
                 ['group3.blocks.block7'],
                 ['group3.blocks.block8'],
                 ['pool', 'fc']],
    
    'vit': [['patch_embed', 'pos_drop', 'patch_drop', 'norm_pre'], 
            ['blocks.0'], 
            ['blocks.1'], 
            ['blocks.2'], 
            ['blocks.3'], 
            ['blocks.4'], 
            ['blocks.5'], 
            ['blocks.6'], 
            ['blocks.7'], 
            ['blocks.8'], 
            ['blocks.9'], 
            ['blocks.10'], 
            ['blocks.11'], 
            ['norm', 'fc_norm', 'head_drop', 'head']]
}

REMIND_MODEL_BLOCK_DICT = {
    'resnet18': [['model_G.initial'],
                 ['model_G.group1.blocks.block0'],
                 ['model_G.group1.blocks.block1'],
                 ['model_G.group2.blocks.block0'],
                 ['model_G.group2.blocks.block1'],
                 ['model_G.group3.blocks.block0'],
                 ['model_G.group3.blocks.block1'],
                 ['model_F.group4.blocks.block0'],
                 ['model_F.group4.blocks.block1'],
                 ['model_F.pool', 'fc']],
    
    'resnet32': [['model_G.initial'],
                 ['model_G.group1.blocks.block0'],
                 ['model_G.group1.blocks.block1'],
                 ['model_G.group1.blocks.block2'],
                 ['model_G.group1.blocks.block3'],
                 ['model_G.group1.blocks.block4'],
                 ['model_G.group2.blocks.block0'],
                 ['model_G.group2.blocks.block1'],
                 ['model_G.group2.blocks.block2'],
                 ['model_G.group2.blocks.block3'],
                 ['model_G.group2.blocks.block4'],
                 ['model_G.group3.blocks.block0'],
                 ['model_G.group3.blocks.block1'],
                 ['model_G.group3.blocks.block2'],
                 ['model_F.group3.blocks.block3'],
                 ['model_F.group3.blocks.block4'],
                 ['model_F.pool','fc']],
    
    'vit': [['patch_embed', 'pos_drop', 'patch_drop', 'norm_pre'], 
            ['blocks.0'], 
            ['blocks.1'], 
            ['blocks.2'], 
            ['blocks.3'], 
            ['blocks.4'], 
            ['blocks.5'], 
            ['blocks.6'], 
            ['blocks.7'], 
            ['blocks.8'], 
            ['blocks.9'], 
            ['blocks.10'], 
            ['blocks.11'], 
            ['norm', 'fc_norm', 'head_drop', 'head']]
}

MEMO_MODEL_BLOCK_DICT = {
    'resnet18': [['backbone.initial'],
                 ['backbone.group1.blocks.block0'],
                 ['backbone.group1.blocks.block1'],
                 ['backbone.group2.blocks.block0'],
                 ['backbone.group2.blocks.block1'],
                 ['backbone.group3.blocks.block0'],
                 ['backbone.group3.blocks.block1'],
                 ['AdaptiveExtractors.0.group4.blocks.block0'],
                 ['AdaptiveExtractors.0.group4.blocks.block1'],
                 ['AdaptiveExtractors.0.pool'], ['fc']],
    
    'resnet32': [['backbone.initial'],
                 ['backbone.group1.blocks.block0'],
                 ['backbone.group1.blocks.block1'],
                 ['backbone.group1.blocks.block2'],
                 ['backbone.group1.blocks.block3'],
                 ['backbone.group1.blocks.block4'],
                 ['backbone.group2.blocks.block0'],
                 ['backbone.group2.blocks.block1'],
                 ['backbone.group2.blocks.block2'],
                 ['backbone.group2.blocks.block3'],
                 ['backbone.group2.blocks.block4'],
                 ['AdaptiveExtractors.0.group3.blocks.block0'],
                 ['AdaptiveExtractors.0.group3.blocks.block1'],
                 ['AdaptiveExtractors.0.group3.blocks.block2'],
                 ['AdaptiveExtractors.0.group3.blocks.block3'],
                 ['AdaptiveExtractors.0.group3.blocks.block4'],
                 ['AdaptiveExtractors.0.pool'], ['fc']],
    
    'vit': [['patch_embed', 'pos_drop', 'patch_drop', 'norm_pre'], 
            ['blocks.0'], 
            ['blocks.1'], 
            ['blocks.2'], 
            ['blocks.3'], 
            ['blocks.4'], 
            ['blocks.5'], 
            ['blocks.6'], 
            ['blocks.7'], 
            ['blocks.8'], 
            ['blocks.9'], 
            ['blocks.10'], 
            ['blocks.11'], 
            ['norm', 'fc_norm', 'head_drop', 'head']]
}


def get_blockwise_flops(flops_dict, model_name, method=None):

    if method=="memo":
        assert model_name in MEMO_MODEL_BLOCK_DICT.keys()
        block_list = MEMO_MODEL_BLOCK_DICT[model_name]
    elif method=="remind":
        assert model_name in REMIND_MODEL_BLOCK_DICT.keys()
        block_list = REMIND_MODEL_BLOCK_DICT[model_name]
    else:
        assert model_name in MODEL_BLOCK_DICT.keys()
        block_list = MODEL_BLOCK_DICT[model_name]
    
    forward_flops = []
    backward_flops = []
    G_forward_flops = []
    G_backward_flops = []
    F_forward_flops = []
    F_backward_flops = []
    G_forward, G_backward, F_forward, F_backward = [], [], [], []
    
    for block in block_list:
        forward_flops.append(sum([flops_dict[layer]['forward_flops']/10e9 for layer in block]))
        backward_flops.append(sum([flops_dict[layer]['backward_flops']/10e9 for layer in block]))
    
        if method=="remind":
            for layer in block:
                if "model_G" in layer:
                    G_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    G_backward.append(flops_dict[layer]['backward_flops']/10e9)
                else:
                    F_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    F_backward.append(flops_dict[layer]['backward_flops']/10e9)
                    
            G_forward_flops.append(sum(G_forward))
            G_backward_flops.append(sum(G_backward))
            F_forward_flops.append(sum(F_forward))
            F_backward_flops.append(sum(F_backward))
            
        elif method=="memo":
            for layer in block:
                if "backbone" in layer:
                    G_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    G_backward.append(flops_dict[layer]['backward_flops']/10e9)
                elif "AdaptiveExtractors" in layer or "fc" in layer:
                    F_forward.append(flops_dict[layer]['forward_flops']/10e9)
                    F_backward.append(flops_dict[layer]['backward_flops']/10e9)
  
            G_forward_flops.append(sum(G_forward))
            G_backward_flops.append(sum(G_backward))
            F_forward_flops.append(sum(F_forward))
            F_backward_flops.append(sum(F_backward))


    return forward_flops, backward_flops, G_forward_flops, G_backward_flops, F_forward_flops, F_backward_flops