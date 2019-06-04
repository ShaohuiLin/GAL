import os
import re
import time
import utils.common as utils
from importlib import import_module
from tensorboardX import SummaryWriter
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from utils.options import args
import pdb
from model import *

device = torch.device(f"cuda:{args.gpus[0]}")
ckpt = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')

def main():
    start_epoch = 0
    best_prec1, best_prec5 = 0.0, 0.0


    # Data loading
    print('=> Preparing data..')
    loader = import_module('data.' + args.dataset).Data(args)

    # Create model
    print('=> Building model...')
    criterion = nn.CrossEntropyLoss()

    # Fine tune from a checkpoint
    refine = args.refine
    assert refine is not None, 'refine is required'
    checkpoint = torch.load(refine, map_location=device)
        
    if args.pruned:
        state_dict = checkpoint['state_dict_s']
        if args.arch == 'vgg':
            cfg = checkpoint['cfg']
            model = vgg_16_bn_sparse(cfg = cfg).to(device)
        # pruned = sum([1 for m in mask if mask == 0])
        # print(f"Pruned / Total: {pruned} / {len(mask)}")
        elif args.arch =='resnet':
            mask = checkpoint['mask']
            model = resnet_56_sparse(has_mask = mask).to(device)

        elif args.arch == 'densenet':
            filters = checkpoint['filters']
            indexes = checkpoint['indexes']
            model = densenet_40_sparse(filters = filters, indexes = indexes).to(device)
        elif args.arch =='googlenet':
            mask = checkpoint['mask']
            model = googlenet_sparse(has_mask = mask).to(device)
        model.load_state_dict(state_dict)
    else:
        model = import_module('utils.preprocess').__dict__[f'{args.arch}'](args, checkpoint['state_dict_s'])

    print_logger.info(f"Simply test after pruning...")
    test_prec1, test_prec5 = test(args, loader.loader_test, model, criterion, writer_test)
    
    if args.test_only:
        return 

    if args.keep_grad:
        for name, weight in model.named_parameters():
            if 'mask' in name:
                weight.requires_grad = False

    train_param = [param for name, param in model.named_parameters() if 'mask' not in name]

    optimizer = optim.SGD(train_param, lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_decay_step, gamma=0.1)

    resume = args.resume
    if resume:
        print('=> Loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=device)
        start_epoch = checkpoint['epoch']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print('=> Continue from epoch {}...'.format(start_epoch))


    for epoch in range(start_epoch, args.num_epochs):
        scheduler.step(epoch)

        train(args, loader.loader_train, model, criterion, optimizer, writer_train, epoch)
        test_prec1, test_prec5 = test(args, loader.loader_test, model, criterion, writer_test, epoch)

        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)

        state = {
            'state_dict_s': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1
        }

        ckpt.save_model(state, epoch + 1, is_best)

    print_logger.info(f"=> Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")

def train(args, loader_train, model, criterion, optimizer, writer_train, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.train()
    num_iterations = len(loader_train)

    for i, (inputs, targets) in enumerate(loader_train, 1):

        inputs = inputs.to(args.gpus[0])
        targets = targets.to(args.gpus[0])

        logits = model(inputs)
        loss = criterion(logits, targets)

        prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))

        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(args, loader_test, model, criterion, writer_test, epoch=0):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model.eval()
    num_iterations = len(loader_test)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))

        print_logger.info(f'* Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')

    if not args.test_only:
        writer_test.add_scalar('test_top1', top1.avg, epoch)

    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
