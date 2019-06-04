import sys
sys.path.append("..")

import re
import numpy as np
from collections import OrderedDict
from importlib import import_module
from model import *
resnet_56_sparse, densenet_40_sparse, vgg_16_bn_sparse, googlenet_sparse
import torch

def resnet(args, state_dict):
    n = (56 - 2) // 6
    layers = np.arange(0, 3*n ,n)
 
    mask_block = []
    for name, weight in state_dict.items():
        if 'mask' in name:
            mask_block.append(weight.item())

    pruned_num = sum(m <= 0.0 for m in mask_block)
    pruned_blocks = [int(m) for m in np.argwhere(np.array(mask_block) <= 0.0)]

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

    print(f"Pruned / Total: {pruned_num} / {len(mask_block)}")
    print("Pruned blocks", pruned_blocks)

    save_dir = f'{args.job_dir}/{args.arch}_pruned_{pruned_num}.pt'
    print(f'=> Saving pruned model to {save_dir}')
    
    save_state_dict = {}
    save_state_dict['state_dict_s'] = new_state_dict
    save_state_dict['mask'] = mask_block
    torch.save(save_state_dict, save_dir)

    if not args.random:
        model.load_state_dict(new_state_dict)

    return model

def densenet(args, state_dict):
    pruned_num = 0
    total = 0
    filters_list = []
    indexes_list = []
    indexes_dense = []
    indexes_trans = []
    for name, weight in state_dict.items():
        if 'mask' in name:
            weight_copy = weight.clone()
            indexes = np.array(weight_copy.gt(0.0).cpu())
            selected_index = np.squeeze(np.argwhere(indexes))

            if selected_index.size == 1:
                selected_index = np.resize(selected_index, (1,))
            filters = sum(indexes)

            size = weight_copy.size()[0]

            if (size - filters) == size:
                selected_index = [0]
                filters = 1

            if 'dense' in name:
                indexes_dense.append(selected_index)
            
            if 'trans' in name:
                indexes_trans.append(selected_index)

            indexes_list.append(selected_index)
            filters_list.append(filters)

            pruned_num += size - filters
            total += size

            state_dict[name] = weight_copy[selected_index]

    i = 0
    j = 0
    for name, weight in state_dict.items():
        if 'conv1' in name:
            if 'dense' in name:
                state_dict[name] = state_dict[name][:, indexes_dense[i], :, :]
                i += 1
            if 'trans' in name:
                state_dict[name] = state_dict[name][:, indexes_trans[j], :, :]
                j += 1


    model = densenet_40_sparse(filters=filters_list, indexes=indexes_list).to(args.gpus[0])
    model.load_state_dict(state_dict)

    print(f"Pruned / Total: {pruned_num} / {total}")
    save_dir = f'{args.job_dir}/{args.arch}_pruned_{pruned_num}.pt'
    save_state_dict = {}
    save_state_dict['state_dict_s'] = state_dict
    save_state_dict['filters'] = filters_list
    save_state_dict['indexes'] = indexes_list
    torch.save(save_state_dict, save_dir)
    print(f'=> Saving model to {save_dir}')

    return model

def vgg(args, state_dict):
    pruned_num = 0
    total = 0
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
    indexes_list = [np.arange(3)]

    for name, weight in state_dict.items():
        if 'mask' in name:
            weight_copy = weight.clone()
            indexes = np.array(weight_copy.gt(0.0).cpu())  # index to retrain
            selected_index = np.squeeze(np.argwhere(indexes))
            if selected_index.size == 1:
                selected_index = np.resize(selected_index, (1,))
            filters = sum(indexes)
            size = weight_copy.size()[0]

            if (size - filters) == size:
                selected_index = [0]
                filters = 1

            if 'features' in name:
                # get the name of conv in state_dict
                idx = int(re.findall(r"\d+", name.split('.')[1])[0])
                cfg[idx] = filters
                state_dict['features.conv{}.weight'.format(idx)] = state_dict['features.conv{}.weight'.format(idx)][selected_index, :, :, :]
                state_dict['features.conv{}.weight'.format(idx)] = state_dict['features.conv{}.weight'.format(idx)][:, indexes_list[-1], :, :]
                state_dict['features.conv{}.bias'.format(idx)] = state_dict['features.conv{}.bias'.format(idx)][selected_index]
                state_dict['features.norm{}.weight'.format(idx)] = state_dict['features.norm{}.weight'.format(idx)][selected_index]
                state_dict['features.norm{}.bias'.format(idx)] = state_dict['features.norm{}.bias'.format(idx)][selected_index]
                state_dict['features.norm{}.running_mean'.format(idx)] = state_dict['features.norm{}.running_mean'.format(idx)][
                    selected_index]
                state_dict['features.norm{}.running_var'.format(idx)] = state_dict['features.norm{}.running_var'.format(idx)][
                    selected_index]

            elif 'classifier' in name:
                state_dict['classifier.linear1.weight'] = state_dict['classifier.linear1.weight'][selected_index, :]
                state_dict['classifier.linear1.weight'] = state_dict['classifier.linear1.weight'][:,indexes_list[-1]]
                state_dict['classifier.linear1.bias'] = state_dict['classifier.linear1.bias'][selected_index]
                state_dict['classifier.norm1.running_mean'] = state_dict['classifier.norm1.running_mean'][selected_index]
                state_dict['classifier.norm1.running_var'] = state_dict['classifier.norm1.running_var'][selected_index]
                state_dict['classifier.norm1.weight'] = state_dict['classifier.norm1.weight'][selected_index]
                state_dict['classifier.norm1.bias'] = state_dict['classifier.norm1.bias'][selected_index]
                cfg[-1] = filters
                state_dict['classifier.linear2.weight'] = state_dict['classifier.linear2.weight'][:, selected_index]

            indexes_list.append(selected_index)
            pruned_num += size - filters
            total += size

            # change the state dict of mask
            state_dict[name] = weight_copy[selected_index]

    model = vgg_16_bn_sparse(cfg=cfg).to(args.gpus[0])
    model.load_state_dict(state_dict)

    print(f"Pruned / Total: {pruned_num} / {total}")

    save_dir = f'{args.job_dir}/{args.arch}_pruned_{pruned_num}.pt'
    save_state_dict = {}
    save_state_dict['state_dict_s'] = state_dict
    save_state_dict['cfg'] = cfg
    torch.save(save_state_dict, save_dir)
    print(f'=> Saving model to {save_dir}')

    return model

def googlenet(args, state_dict):
    filters = [
        [64, 128, 32, 32],
        [128, 192, 96, 64],
        [192, 208, 48, 64],
        [160, 224, 64, 64],
        [128, 256, 64, 64],
        [112, 288, 64, 64],
        [256, 320, 128, 128],
        [256, 320, 128, 128],
        [384, 384, 128, 128]
    ]
    mask = []
    arr = []
    i = 0

    for name, weight in state_dict.items():
        if 'mask' in name:
            mask.append(weight.item())

    cfg = [int(m) for m in np.argwhere(np.array(mask)==0)]
    module_name = 'inception_a3'
    index = list(range(192)) 

    i = 0
    for name, param in state_dict.items():
        if 'branch1x1.0.weight' in name or 'branch3x3.0.weight' in name or 'branch5x5.0.weight' in name or 'branch_pool.1.weight' in name:
            # get module_name
            if name.split('.')[0] != module_name:
                start = 0
                index = []

                for j, v in enumerate(mask[i * 4:i * 4 + 4]):
                    if v != 0:
                        index.extend([i for i in range(start, start + filters[i][j])]) # select index
                    start += filters[i][j]
                    
                i += 1
                module_name = name.split('.')[0]
                
            state_dict[name] = param[:, index, :, :]

        elif 'linear.weight' in name:
            fc_index = []
            fc_start = 0
            for j, v in enumerate(mask[8*4:8*4+4]):
                if v != 0:
                    fc_index.extend([i for i in range(fc_start, fc_start + filters[8][j])])
                fc_start += filters[i][j]
            state_dict[name] = param[:, fc_index]

    pruned_num = sum(m == 0 for m in mask)
    model = googlenet_sparse(has_mask=mask).to(args.gpus[0])

    print('\n---- After Prune ----\n')
    
    model_dict = model.state_dict()
    pretrained = {k:v for k,v in state_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)

    save_state_dict = {}
    save_state_dict['state_dict_s'] = pretrained
    save_state_dict['mask'] = mask
    save_dir = f'{args.job_dir}/{args.arch}_pruned_{pruned_num}.pt'
    torch.save(save_state_dict, save_dir)
    print(f'=> Saving model to {save_dir}')
    
    return model
