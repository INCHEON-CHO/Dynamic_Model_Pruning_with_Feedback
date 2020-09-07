import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_weight_threshold(model, rate, args):
    importance_all = None
    for name, item in model.module.named_parameters():
        if len(item.size())==4 and 'mask' not in name:
            weights = item.data.view(-1).cpu()
            grads = item.grad.data.view(-1).cpu()

            if args.prune_imp == 'L1':
                importance = weights.abs().numpy()
            elif args.prune_imp == 'L2':
                importance = weights.pow(2).numpy()
            elif args.prune_imp == 'grad':
                importance = grads.abs().numpy()
            elif args.prune_imp == 'syn':
                importance = (weights * grads).abs().numpy()
            

            if importance_all is None:
                importance_all = importance
            else:
                importance_all = np.append(importance_all, importance)

    threshold = np.sort(importance_all)[int(len(importance_all) * rate)]
    return threshold


def weight_prune(model, threshold, args):
    state = model.state_dict()
    for name, item in model.named_parameters():
        if 'weight' in name:
            key = name.replace('weight', 'mask')
            if key in state.keys():
                if args.prune_imp == 'L1':
                    mat = item.data.abs()
                elif args.prune_imp == 'L2':
                    mat = item.data.pow(2)
                elif args.prune_imp == 'grad':
                    mat = item.grad.data.abs()
                elif args.prune_imp == 'syn':
                    mat = (item.data * item.grad.data).abs()
                state[key].data.copy_(torch.gt(mat, threshold).float())


def get_filter_mask(model, rate, args):
    importance_all = None
    for name, item in model.module.named_parameters():
        if len(item.size())==4 and 'weight' in name:
            filters = item.data.view(item.size(0), -1).cpu()
            weight_len = filters.size(1)
            if args.prune_imp =='L1':
                importance = filters.abs().sum(dim=1).numpy() / weight_len
            elif args.prune_imp == 'L2':
                importance = filters.pow(2).sum(dim=1).numpy() / weight_len
        
            if importance_all is None:
                importance_all = importance
            else:
                importance_all = np.append(importance_all, importance)

    threshold = np.sort(importance)[int(len(importance) * rate)]
    #threshold = np.percentile(importance, rate)
    filter_mask = np.greater(importance, threshold)
    return filter_mask


def filter_prune(model, filter_mask):
    idx = 0
    for name, item in model.module.named_parameters():
        if len(item.size())==4 and 'mask' in name:
            for i in range(item.size(0)):
                item.data[i,:,:,:] = 1 if filter_mask[idx] else 0
                idx+=1


def cal_sparsity(model):
    mask_nonzeros = 0
    mask_length = 0
    total_weights = 0

    for name, item in model.module.named_parameters():
        if 'mask' in name:
            flatten = item.data.view(-1)
            np_flatten = flatten.cpu().numpy()

            mask_nonzeros += np.count_nonzero(np_flatten)
            mask_length += item.numel()

        if 'weight' in name or 'bias' in name:
            total_weights += item.numel()

    num_zero = mask_length - mask_nonzeros
    sparsity = (num_zero / total_weights) * 100
    return total_weights, num_zero, sparsity
