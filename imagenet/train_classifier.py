import argparse
import json
import os
from datetime import datetime
from os.path import join
import random
import shutil
import warnings
import pathlib
import time

import repackage
repackage.up()

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from imagenet.dataloader import (get_imagenet_dls, get_cf_imagenet_dls,
                                 get_cue_conflict_dls, get_in9_dls)
from imagenet.models import InvariantEnsemble
from utils import eval_bg_gap, eval_shape_bias

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

best_acc1_overall = 0

def main(args):

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1_overall
    global writer
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training,
            # rank needs to be the
            # global rank among all the
            # processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)

    # save path
    if not args.resume:
        time_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
        model_path = join('.', 'imagenet', 'experiments',
                            f'classifier_{time_str}_{args.name}')
        pathlib.Path(model_path).mkdir(parents=True, exist_ok=True)
        # dump current args in this folder
        with open(join(model_path, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    else:
        model_path = join(*pathlib.PurePosixPath(args.resume).parts[:-1])

    # create model
    print(f"=> using Multi Head Ensemble, arch: {args.arch}, pretrained: {args.pretrained}")
    model = InvariantEnsemble(args.arch, args.pretrained)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')

    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per DistributedDataParallel
            # we need to divide the batch size ourselves basedo nt the toal number of GPUS
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1_overall = checkpoint['best_acc1_overall']
            if args.gpu is not None:
                # best_acc1_overall may be from a checkpoint from a different GPU
                best_acc1_overall = best_acc1_overall.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    

    ### dataloaders
    train_loader, val_loader, train_sampler = get_imagenet_dls(args.distributed, args.batch_size, args.workers)
    cf_train_loader, cf_val_loader, cf_train_sampler = get_cf_imagenet_dls(args.cf_data, args.cf_ratio, len(train_loader), args.distributed, args.batch_size, args.workers)
    dl_shape_bias = get_cue_conflict_dls(args.batch_size, args.workers)
    dls_in9 = get_in9_dls(args.distributed, args.batch_size, args.workers, ['mixed_rand', 'mixed_same'])

    
    # eval before training
    if not args.resume:
        metrics = validate(model, val_loader, cf_val_loader,
                               dl_shape_bias, dls_in9, args)
        if args.evaluate: return
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            writer = SummaryWriter(logdir=model_path.replace('experiments', 'runs'))
            update_tb(writer, 0, metrics)
        elif args.resume and not args.evaluate:
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                writer = SummaryWriter(logdir=model_path.replace('experiments', 'runs'))

    # training loop
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)
            cf_train_sampler.set_epoch(epoch)

        cf_train_loader.dataset.resample()
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, cf_train_loader, model, criterion,
              optimizer, epoch, args)

        # evaluate on validation set
        metrics = validate(model, val_loader, cf_val_loader,
                               dl_shape_bias, dls_in9, args)

        # remember best acc@1 and save checkpoint
        acc1_overall = metrics['acc1/0_overall']
        is_best = acc1_overall > best_acc1_overall
        best_acc1_overall = max(acc1_overall, best_acc1_overall)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            print('saving checkpoint')
            save_checkpoint(
                state={
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1_overall': best_acc1_overall,
                    'optimizer': optimizer.state_dict(),
                },
                is_best=is_best,
                model_path=model_path,
            )
            update_tb(writer, epoch + 1, metrics)


def update_tb(writer, epoch, metrics):
    for k, v in metrics.items():
        writer.add_scalar(k, v, epoch)


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def set_bn_train(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.train()


def train(train_loader, cf_train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    cf_losses = AverageMeter('CF_Loss', ':.4e')
    top1_real = AverageMeter('Real Acc@1', ':6.2f')
    top5_real = AverageMeter('Real Acc@5', ':6.2f')
    top1_shape = AverageMeter('Shape Acc@1', ':6.2f')
    top5_shape = AverageMeter('Shape Acc@5', ':6.2f')
    top1_texture = AverageMeter('Texture Acc@1', ':6.2f')
    top5_texture = AverageMeter('Texture Acc@5', ':6.2f')
    top1_bg = AverageMeter('Bg Acc@1', ':6.2f')
    top5_bg = AverageMeter('Bg Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                             [batch_time, data_time, losses,
                              top1_real, top5_real, top1_shape, top5_shape,
                              top1_texture, top5_texture, top1_bg, top5_bg],
                             prefix=f"Epoch: [{epoch}]")


    # switch to train mode
    model.train()
    model.apply(set_bn_eval)

    end = time.time()
    for i, (data, data_cf) in enumerate(zip(train_loader, cf_train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None or torch.cuda.is_available():
            data = {k: v.cuda(args.gpu, non_blocking=True) for k, v in data.items()}
            data_cf = {k: v.cuda(args.gpu, non_blocking=True) for k, v in data_cf.items()}

        # compute output
        out = model(data['ims'])
        loss = criterion(out['avg_preds'], data['labels'])

        # compute gradient
        loss.backward()

        # compute output for counterfactuals
        out_cf = model(data_cf['ims'])
        loss_cf = criterion(out_cf['shape_preds'], data_cf['shape_labels'])
        loss_cf += criterion(out_cf['texture_preds'], data_cf['texture_labels'])
        loss_cf += criterion(out_cf['bg_preds'], data_cf['bg_labels'])

        # compute gradient
        loss_cf.backward()

        # measure accuracy and record loss
        sz = len(data['ims'])
        acc1, acc5 = accuracy(out['avg_preds'], data['labels'], topk=(1, 5))
        losses.update(loss.item(), data['ims'].size(0))
        cf_losses.update(loss_cf.item(), data['ims'].size(0))
        top1_real.update(acc1[0], sz)
        top5_real.update(acc5[0], sz)
        acc1, acc5 = accuracy(out_cf['shape_preds'], data_cf['shape_labels'], topk=(1, 5))
        top1_shape.update(acc1[0], sz)
        top5_shape.update(acc5[0], sz)
        acc1, acc5 = accuracy(out_cf['texture_preds'], data_cf['texture_labels'], topk=(1, 5))
        top1_texture.update(acc1[0], sz)
        top5_texture.update(acc5[0], sz)
        acc1, acc5 = accuracy(out_cf['bg_preds'], data_cf['bg_labels'], topk=(1, 5))
        top1_bg.update(acc1[0], sz)
        top5_bg.update(acc5[0], sz)

        # Step
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logging
        if i % args.print_freq == 0:
            progress.display(i)


def validate(model, val_loader, cf_val_loader, dl_shape_bias, dls_in9, args):
    real_accs = validate_imagenet(val_loader, model, args)
    cf_accs = validate_counterfactual(cf_val_loader, model, args)
    shapes_biases = validate_shape_bias(model, dl_shape_bias)
    in_9_accs = validate_in_9(dls_in9, model)
    val_res = {**real_accs, **cf_accs, **shapes_biases, **in_9_accs}

    # Sync up
    if args.multiprocessing_distributed:
        metrics = {}
        for k, v in val_res.items():
            metrics[k] = v.detach().to(args.gpu)
            dist.all_reduce(metrics[k], dist.ReduceOp.SUM, async_op=False)
            metrics[k] = metrics[k] / args.world_size
    else:
        metrics = val_res

    return metrics


def validate_imagenet(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Real Acc@1', ':6.2f')
    top5 = AverageMeter('Real Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            if args.gpu is not None or torch.cuda.is_available():
                data = {k: v.cuda(args.gpu, non_blocking=True) for k, v in data.items()}

            # compute output
            out = model(data['ims'].cuda())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(out['avg_preds'], data['labels'], topk=(1, 5))
            top1.update(acc1[0], data['labels'].size(0))
            top5.update(acc5[0], data['labels'].size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # logging
            if i % args.print_freq == 0:
                progress.display(i)

    print(f'* Real: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')

    return {'acc1/1_real': top1.avg,
            'acc5/1_real': top5.avg}


def validate_counterfactual(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1_shape = AverageMeter('Shape Acc@1', ':6.2f')
    top5_shape = AverageMeter('Shape Acc@5', ':6.2f')
    top1_texture = AverageMeter('Texture Acc@1', ':6.2f')
    top5_texture = AverageMeter('Texture Acc@5', ':6.2f')
    top1_bg = AverageMeter('Bg Acc@1', ':6.2f')
    top5_bg = AverageMeter('Bg Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader),
                             [batch_time, top1_shape, top5_shape, top1_texture, top5_texture, top1_bg, top5_bg],
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            if args.gpu is not None or torch.cuda.is_available():
                data = {k: v.cuda(args.gpu, non_blocking=True) for k, v in data.items()}

            # compute output
            out = model(data['ims'])

            # measure accuracy and record loss
            sz = len(data['ims'])
            acc1, acc5 = accuracy(out['shape_preds'], data['shape_labels'], topk=(1, 5))
            top1_shape.update(acc1[0], sz)
            top5_shape.update(acc5[0], sz)
            acc1, acc5 = accuracy(out['texture_preds'], data['texture_labels'], topk=(1, 5))
            top1_texture.update(acc1[0], sz)
            top5_texture.update(acc5[0], sz)
            acc1, acc5 = accuracy(out['bg_preds'], data['bg_labels'], topk=(1, 5))
            top1_bg.update(acc1[0], sz)
            top5_bg.update(acc5[0], sz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # logging
            if i % args.print_freq == 0:
                progress.display(i)

    print(f'* Shape: Acc@1 {top1_shape.avg:.3f} Acc@5 {top5_shape.avg:.3f}')
    print(f'* Texture: Acc@1 {top1_texture.avg:.3f} Acc@5 {top5_texture.avg:.3f}')
    print(f'* BG: Acc@1 {top1_bg.avg:.3f} Acc@5 {top5_bg.avg:.3f}')

    return {'acc1/2_shape': top1_shape.avg,
            'acc1/3_texture': top1_texture.avg,
            'acc1/4_bg': top1_bg.avg,
            'acc5/2_shape': top5_shape.avg,
            'acc5/3_texture': top5_texture.avg,
            'acc5/4_bg': top5_bg.avg}


def validate_shape_bias(model, dl):
    model.eval()

    res = {}
    backbone = model.module.backbone

    shape_bias = eval_shape_bias(model.module.m_shape, backbone, dl)
    print(f"Shape Classifier: shape bias {shape_bias}")
    res['shape_biases/0_m_shape_bias'] = shape_bias

    shape_bias = eval_shape_bias(model.module.m_texture, backbone, dl)
    print(f"Texture Classifier: shape bias {shape_bias}")
    res['shape_biases/1_m_texture_bias'] = shape_bias

    shape_bias = eval_shape_bias(model.module.m_bg, backbone, dl)
    print(f"Background Expert: shape bias {shape_bias}")
    res['shape_biases/2_m_bg_bias'] = shape_bias

    return res


def validate_in_9(dls_in9, model):
    model.eval()

    map_to_in9 = {}
    with open('utils/eval_bg_gap/in_to_in9.json', 'r') as f:
        map_to_in9.update(json.load(f))

    # Evaluate model
    res = {}
    for k, dl in dls_in9.items():
        acc1 = eval_bg_gap(dl, model, map_to_in9)
        for pred_k, acc in acc1.items():
            print(f'* {k}_{pred_k}: Acc@1 {acc:3.2f}')
            res[f'in_9_acc1_{k}/{pred_k}'] = acc

    # BG gap for the prediction without the background
    res['in_9_gaps/bg_gap'] = res['in_9_acc1_mixed_same/shape_texture'] - res['in_9_acc1_mixed_rand/shape_texture']
    return res


def save_checkpoint(state, is_best, model_path):
    filename = join(model_path, 'checkpoint.pth')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(model_path, 'model_best.pth'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        # return fmtstr.format(**self.__dict__)
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**{'name': self.name, 'avg': self.avg})


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(' | '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 15))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--data', default='data/ImageNet', 
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
    parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=45, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    # distributed stuff
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:8888', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')

    # my arguments
    parser.add_argument('--cf_data', required=True, type=str,
                        help='Path to the counterfactual dataset.')
    parser.add_argument('--name', default='', type=str,
                        help='name of the experiment')
    parser.add_argument('--cf_ratio', default=1.0, type=float,
                        help='Ratio of CF/Real data')

    args = parser.parse_args()
    print(args)
    main(args)
