'''
Modified from https://github.com/jack-willturner/DeepCompression-PyTorch

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] Tao Lin, Sebastian U.stich, Luis Barba, Daniil Dmitriev, Martin Jaggi
    Dynamic Pruning with Feedback. ICLR2020
'''
import time
import random
import pathlib
from os.path import isfile
import copy

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torch.autograd import Variable

import models
import config
import pruning
from utils import *
from data import DataLoader

# for ignore ImageNet PIL EXIF UserWarning and ignore transparent images
import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "(Palette )?images with Transparency", UserWarning)


def hyperparam():
    args = config.config()
    return args

def main(args):
    global arch_name
    if args.cuda and not torch.cuda.is_available():
        raise Exception('No GPU found, please run without --cuda')

    # set model name
    arch_name = set_arch_name(args)
    print('\n=> creating model \'{}\''.format(arch_name))

    if not args.prune:  # base  
        model, image_size = models.__dict__[args.arch](data=args.dataset, num_layers=args.layers,
                                                        width_mult=args.width_mult,
                                                        depth_mult=args.depth_mult,
                                                        model_mult=args.model_mult)

    elif args.prune:    # for pruning
        pruner = pruning.__dict__[args.pruner]
        model, image_size = pruning.models.__dict__[args.arch](data=args.dataset, num_layers=args.layers,
                                                   width_mult=args.width_mult,
                                                   depth_mult=args.depth_mult,
                                                   model_mult=args.model_mult,
                                                   mnn=pruner.mnn)
    
    assert model is not None, 'Unavailable model parameters!! exit...\n'

    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                            momentum=args.momentum, weight_decay=args.weight_decay,
                            nesterov=args.nesterov)
    scheduler = set_scheduler(optimizer, args)
    
    # set multi-gpu
    if args.cuda:
        torch.cuda.set_device(args.gpuids[0])
        with torch.cuda.device(args.gpuids[0]):
            model = model.cuda()
            criterion = criterion.cuda()
        model = nn.DataParallel(model, device_ids=args.gpuids,
                                output_device=args.gpuids[0])
        cudnn.benchmark = True
        
        # for distillation

    # Dataset loading
    print('==> Load data..')
    start_time = time.time()
    train_loader, val_loader = DataLoader(args.batch_size, args.dataset,
                                          args.workers, args.datapath, image_size,
                                          args.cuda)
    elapsed_time = time.time() - start_time
    print('===> Data loading time: {:,}m {:.2f}s'.format(
        int(elapsed_time//60), elapsed_time%60))
    print('===> Data loaded..')

    # load a pre-trained model
    if args.load is not None:
        ckpt_file = pathlib.Path('checkpoint') / arch_name / args.dataset / args.load
        assert isfile(ckpt_file), '==> no checkpoint found \"{}\"'.format(args.load)

        print('==> Loading Checkpoint \'{}\''.format(args.load))
        # check pruning or quantization or transfer
        strict = False if args.prune else True
        # load a checkpoint
        checkpoint = load_model(model, ckpt_file, main_gpu=args.gpuids[0], use_cuda=args.cuda, strict=strict)
        print('==> Loaded Checkpoint \'{}\''.format(args.load))


    # for training
    if args.run_type == 'train':
        # init parameters
        start_epoch = 0
        global iterations
        iterations = 0
        best_acc1 = 0.0
        train_time = 0.0
        validate_time = 0.0

        for epoch in range(start_epoch, args.epochs):
            print('\n==> {}/{} training'.format(
                    arch_name, args.dataset))
            print('==> Epoch: {}, lr = {}'.format(
                epoch, optimizer.param_groups[0]["lr"]))

            # train for one epoch
            print('===> [ Training ]')
            start_time = time.time()
            acc1_train, acc5_train = train(args, train_loader,
                epoch=epoch, model=model,
                criterion=criterion, optimizer=optimizer)

            elapsed_time = time.time() - start_time
            train_time += elapsed_time
            print('====> {:.2f} seconds to train this epoch\n'.format(
                elapsed_time))

            # evaluate on validation set
            print('===> [ Validation ]')
            start_time = time.time()
            acc1_valid, acc5_valid = validate(args, val_loader,
                epoch=epoch, model=model, criterion=criterion)
            elapsed_time = time.time() - start_time
            validate_time += elapsed_time
            print('====> {:.2f} seconds to validate this epoch'.format(
                elapsed_time))
        
            # learning rate schduling
            scheduler.step()

            acc1_train = round(acc1_train.item(), 4)
            acc5_train = round(acc5_train.item(), 4)
            acc1_valid = round(acc1_valid.item(), 4)
            acc5_valid = round(acc5_valid.item(), 4)

            # remember best Acc@1 and save checkpoint and summary csv file
            state = model.state_dict()
            summary = [epoch, acc1_train, acc5_train, acc1_valid, acc5_valid]

            is_best = acc1_valid > best_acc1
            best_acc1 = max(acc1_valid, best_acc1)
            if is_best:
                save_model(arch_name, args.dataset, state, args.save)
            save_summary(arch_name, args.dataset, args.save.split('.pth')[0], summary)

            # for pruning
            if args.prune:
                num_total, num_zero, sparsity = pruning.cal_sparsity(model)
                print('\n====> sparsity: {:.2f}% || num_zero/num_total: {}/{}'.format(sparsity, num_zero, num_total))

            # end of one epoch
            print()

        # calculate the total training time 
        avg_train_time = train_time / (args.epochs - start_epoch)
        avg_valid_time = validate_time / (args.epochs - start_epoch)
        total_train_time = train_time + validate_time
        print('====> average training time each epoch: {:,}m {:.2f}s'.format(
            int(avg_train_time//60), avg_train_time%60))
        print('====> average validation time each epoch: {:,}m {:.2f}s'.format(
            int(avg_valid_time//60), avg_valid_time%60))
        print('====> training time: {}h {}m {:.2f}s'.format(
            int(train_time//3600), int((train_time%3600)//60), train_time%60))
        print('====> validation time: {}h {}m {:.2f}s'.format(
            int(validate_time//3600), int((validate_time%3600)//60), validate_time%60))
        print('====> total training time: {}h {}m {:.2f}s'.format(
            int(total_train_time//3600), int((total_train_time%3600)//60), total_train_time%60))
        
        return best_acc1
    
    elif args.run_type == 'evaluate':   # for evaluation
        # for evaluation on validation set
        print('\n===> [ Evaluation ]')
        
        # main evaluation
        start_time = time.time()
        acc1, acc5 = validate(args, val_loader, None, model, criterion)
        elapsed_time = time.time() - start_time
        print('====> {:.2f} seconds to evaluate this model\n'.format(
            elapsed_time))
        
        acc1 = round(acc1.item(), 4)
        acc5 = round(acc5.item(), 4)

        # save the result

        ckpt_name = '{}-{}-{}'.format(arch_name, args.dataset, args.load[:-4])
        save_eval([ckpt_name, acc1, acc5])

        if args.prune:
            _,_,sparsity = pruning.cal_sparsity(model)
            print('Sparsity : {}'.format(sparsity))
        return acc1
    else:
        assert False, 'Unkown --run-type! It should be \{train, evaluate\}.'
    

def train(args, train_loader, epoch, model, criterion, optimizer, **kwargs):
    r"""Train model each epoch
    """
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time,
                             losses, top1, top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    #with torch.autograd.set_detect_anomaly(True):
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            target = target.cuda(non_blocking=True)
        
        # for pruning
        if args.prune:
            if (globals()['iterations']+1) % args.prune_freq==0 and (epoch+1) <= args.milestones[1]:
                target_sparsity = args.prune_rate - args.prune_rate * (1 - (globals()['iterations'])/(args.milestones[1] * len(train_loader)))**3
                if args.prune_type == 'structured':
                    filter_mask = pruning.get_filter_mask(model, rate, args)
                    pruning.filter_prune(model, filter_mask)
                elif args.prune_type == 'unstructured':
                    threshold = pruning.get_weight_threshold(model, target_sparsity, args)
                    pruning.weight_prune(model, threshold, args)


        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)

        if i % args.print_freq == 0:
            progress.print(i)

        end = time.time()

        # end of one mini-batch
        globals()['iterations'] += 1

    print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


def validate(args, val_loader, epoch, model, criterion):
    r"""Validate model each epoch and evaluation
    """
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.cuda:
                target = target.cuda(non_blocking=True)

            # compute output
            #if args.ensemble:
            #    output = [k(input) for k in model]
            #    loss = torch.mean(torch.stack([criterion(k, target) for k in output]))
            #    output = torch.stack([F.softmax(k, dim=1) for k in output]).mean(dim=0)
            #else:
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)

            if i % args.print_freq == 0:
                progress.print(i)

            end = time.time()

        print('====> Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg


if __name__ == '__main__':
    start_time = time.time()
    args = hyperparam()
    main(args)
    elapsed_time = time.time() - start_time
    print('====> total time: {}h {}m {:.2f}s'.format(
        int(elapsed_time//3600), int((elapsed_time%3600)//60), elapsed_time%60))
