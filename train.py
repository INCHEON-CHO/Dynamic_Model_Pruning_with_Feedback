'''
Modified from https://github.com/jack-willturner/DeepCompression-PyTorch

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Tao Lin, Sebastian U.stich, Luis Barba, Daniil Dmitriev, Martin Jaggi
    Dynamic Pruning with Feedback. ICLR2020
'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils  import *
from tqdm   import tqdm

parser = argparse.ArgumentParser(description='PyTorch DPF Training')
parser.add_argument('--data', default='cifar10', type=str, help='cifar10, cifar100, imagenet')
parser.add_argument('--cutout', default=False, type=bool, help='using regularization with cutout')
parser.add_argument('--model', default='resnet18', type=str, help='resnet9/18/34/50, resnet20/32/44/56/110/1202, wrn_40_2/_16_2/_40_1')
parser.add_argument('--datapath', default='/dataset/CIFAR', type=str)
parser.add_argument('--expname', default="resnetT2", type=str, help='checkpoint name')
parser.add_argument('--GPU', default='0,1', type=str,help='GPU to use')

### training specific args
parser.add_argument('--epochs', default=300, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.2)
parser.add_argument('--nesterov', dest='nesterov', default=False, action='store_true', help='use neterov momentum')
parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay')
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--do-DPF', dest='do_DPF', default=False, action='store_true', help='use DPF')
parser.add_argument('--target-sparsity', default=0.5, type=float)
parser.add_argument('--frequency', default=1, type=int, help='pruning frequency')

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]=args.GPU

if 'resnet' in args.model:
    '''cifar : 20, 32, 44, 56, 110, 1202'''
    '''imagenet : 9, 18, 34, 50, 101, 152'''
    model = make_ResNet(args.model[6:], args.data)
if 'wrn' in args.model:
    # wrn will be modified for all data
    _, depth, widen_factor = args.model.split('_') 
    model = WideResNet(int(depth), int(widen_factor), args.data)

if torch.cuda.is_available():
    model = model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
model.to(device)

if args.data == 'imagenet':
    trainloader, testloader = get_imagenet_loaders(args.datapath, args.batch_size)
elif args.data == 'cifar10':
    trainloader, testloader = get_cifar10_loaders(args.datapath, args.batch_size, args.cutout)
elif args.data == 'cifar100':
    trainloader, testloader = get_cifar100_loaders(args.datapath, args.batch_size, args.cutout)

optimizer = optim.SGD([w for name, w in model.named_parameters() if not 'mask' in name],
                        lr=args.lr,
                        momentum=0.9,
                        weight_decay=args.weight_decay,
                        nesterov=args.nesterov
                        )
criterion = nn.CrossEntropyLoss()

acc_list = []
remain_percent = []

for epoch in range(args.epochs):
    adjust_learning_rate(optimizer, epoch, args.lr, args.lr_decay)
    print("Epoch : {}, lr : {}".format(epoch, optimizer.param_groups[0]['lr']))
    print('===> [ Training ]')

    if (epoch+1)%args.frequency == 0 and args.do_DPF:
        sparsity = args.target_sparsity - args.target_sparsity * (1 - (epoch+1)/args.epochs)**3
        model = sparsify(model, sparsity*100)

    train(model, epoch, trainloader, criterion, optimizer)
    top1, top5 = validate(model, epoch, testloader, criterion)
    optimizer.step()

    acc_list.append([top1, top5])

    remain_percent.append(number_of_zeros(model))

    if epoch+1  == args.epochs:
        state = {
            'net': model.state_dict(),
            'masks': [w for name, w in model.named_parameters() if 'mask' in name],
            'epoch': epoch,
            'valid_acc': acc_list,
            'remain' : remain_percent
        }
        save_checkpoint(state, args.expname)
