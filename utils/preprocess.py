import sys
sys.path.append("..")

import re
import numpy as np
from collections import OrderedDict
from model import resnet_56, resnet_56_sparse
import torch

def prune_resnet(args, state_dict):
    thre = args.thre
    num_layers = int(args.student_model.split('_')[1])
    n = (num_layers - 2) // 6
    layers = np.arange(0, 3*n ,n)
 
    mask_block = []
    for name, weight in state_dict.items():
        if 'mask' in name:
            mask_block.append(weight.item())

    pruned_num = sum(m <= thre for m in mask_block)
    pruned_blocks = [int(m) for m in np.argwhere(np.array(mask_block) <= thre)]

    old_block = 0
    layer = 'layer1'
    layer_num = int(layer[-1])
    new_block = 0
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if 'layer' in key:
            if key.split('.')[0] != layer:
                layer = key.split('.')[0]
                layer_num = int(layer[-1])
                new_block = 0

            if key.split('.')[1] != old_block:
                old_block = key.split('.')[1]

            if mask_block[layers[layer_num-1] + int(old_block)] == 0:
                if layer_num != 1 and old_block == '0' and 'mask' in key:
                    new_block = 1
                continue

            new_key = re.sub(r'\.\d+\.', '.{}.'.format(new_block), key, 1)
            if 'mask' in new_key: new_block += 1

            new_state_dict[new_key] = state_dict[key]

        else:
            new_state_dict[key] = state_dict[key]

    model = resnet_56_sparse(has_mask=mask_block).to(args.gpus[0])

    print('\n---- After Prune ----\n')
    print(f"Pruned / Total: {pruned_num} / {len(mask_block)}")
    print("Pruned blocks", pruned_blocks)

    save_dir = f'{args.job_dir}/pruned.pt'
    print(f'Saving pruned model to {save_dir}...')
    
    save_state_dict = {}
    save_state_dict['state_dict_s'] = new_state_dict
    save_state_dict['mask'] = mask_block
    torch.save(save_state_dict, save_dir)

    if not args.random:
        model.load_state_dict(new_state_dict)

    return model

