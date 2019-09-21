""" -------------------------------------------------------------------------------------
Training on Imagenet Script

Originally from: https://github.com/pytorch/examples/tree/master/imagenet

Example run command:
     python train_imagenet.py --data <Path to Imagenet Data> --gpu 0 --batch-size 16 --print-freq 100
 ------------------------------------------------------------------------------------ """
import argparse
import os
import random
import shutil
import time
import warnings
import pickle
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models


# model_names = sorted(name for name in models.__dict__
#     if name.islower() and not name.startswith("__")
#     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
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
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

best_acc1 = 0


def main(model):
    args = parser.parse_args()

    print("Input Arguments:")
    for a_idx, arg in enumerate(vars(args)):
        print("\t[{}] {}: {}".format(a_idx, arg, getattr(args, arg)))

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
        main_worker(model, args.gpu, ngpus_per_node, args)


def main_worker(model, gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    results_store_dir = os.path.join(
        './results/imagenet_classification/',
        model.__class__.__name__ + datetime.now().strftime("_%Y%m%d_%H%M%S")
    )
    if not os.path.exists(results_store_dir):
        os.makedirs(results_store_dir)

    # Train Summary File
    summary_file = os.path.join(results_store_dir, 'summary.txt')
    f = open(summary_file, 'w+')

    # Write the training setting
    f.write("Input Arguments:\n")
    for a_idx, arg in enumerate(vars(args)):
        f.write("\t[{}] {}: {}\n".format(a_idx, arg, getattr(args, arg)))
    f.write("{}\n".format('-' * 80))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    # if args.pretrained:
    #     print("=> using pre-trained model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch](pretrained=True)
    # else:
    #     print("=> creating model '{}'".format(args.arch))
    #     model = models.__dict__[args.arch]()

    # model.cuda()
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # # DataParallel will divide and allocate batch_size to all available GPUs
        # if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        #     model.features = torch.nn.DataParallel(model.features)
        #     model.cuda()
        # else:
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
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    print(">>> Setting up Data loaders {}".format('.' * 80))
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    f.write("Epoch, train_loss, train_accTop1, train_accTop5, val_loss val_accTop1, val_accTop5\n")

    # Evaluate performance on Validation set before Training - This for for models that start
    # with pre-trained models
    val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, args)
    f.write("[{}, np.nan, np.nan, np.nan, {:0.4f}, {:0.4f}, {:0.4f}],\n".format(
        0,
        val_loss, val_acc1, val_acc5
    ))

    print(">>> Start Training {} ".format('.'*80))
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train_loss, train_acc1, train_acc5 = \
            train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        val_loss, val_acc1, val_acc5 = validate(val_loader, model, criterion, args)

        f.write("[{}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}, {:0.4f}],\n".format(
            epoch,
            train_loss, train_acc1, train_acc5,
            val_loss, val_acc1, val_acc5
        ))

        # remember best acc@1 and save checkpoint
        is_best = val_acc1 > best_acc1
        best_acc1 = max(val_acc1, best_acc1)

        # if not args.multiprocessing_distributed or \
        #         (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

        if is_best:
            save_checkpoint(
                state={
                    'epoch': epoch + 1,
                    # 'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_acc1': best_acc1,
                    'optimizer': optimizer.state_dict(),
                },
                is_best=is_best,
                filename=os.path.join(results_store_dir, 'best_accuracy.pth')
            )
    f.close()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    epoch_start_time = end
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    print("Epoch Complete Duration {}".format(time.time() - epoch_start_time))

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, args):
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
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,  os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))


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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
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
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def check_requires_grad(model):
    print("Parameters that will be trained (requires_grads):")
    for c_idx, child in enumerate(model.children()):
        print("Layer {}. Type: {}".format(c_idx, child.__class__.__name__))

        if child.__class__.__name__ == 'Sequential':
            for gc_idx, grand_child in enumerate(child.children()):
                print("\tGrandChild {}. Type: {}".format(gc_idx, grand_child.__class__.__name__))

                for p_idx, param in enumerate(grand_child.parameters()):
                    if param.requires_grad:
                        print("\t\t[{}]. Shape {} Requires Grad = {}".format(
                            p_idx, param.shape, param.requires_grad))
        else:
            for p_idx, param in enumerate(child.parameters()):
                if param.requires_grad:
                    print("\t[{}]. Shape {} Requires Grad = {}".format(p_idx, param.shape, param.requires_grad))
    import pdb
    pdb.set_trace()


def embed_resnet50(model_to_embed, pretrained=True):
    """

    :param pretrained:
    :param model_to_embed:
    :return:
    """

    base_model = torchvision_models.resnet50(pretrained=pretrained)

    if not pretrained:
        # Load the edge Extraction of the original model
        resnet50_edge_detect_layer = torchvision_models.resnet50(pretrained=True).conv1
        base_model.features[0].weight.data.copy_(resnet50_edge_detect_layer.weight.data)

    # Replace the first edge extraction layer of the contour integration model with the one from base resnet.
    model_to_embed.conv1 = base_model.conv1

    # Replace the base models edge extraction layer with edge extraction + contour integration model
    base_model.conv1 = model_to_embed

    if pretrained:
        # Only Train the Contour Integration Layer
        for c_idx, child in enumerate(base_model.children()):
            if c_idx >= 1:
                for p_idx, param in enumerate(child.parameters()):
                    param.requires_grad = False

    # Set the requires gradient parameter of the first edge extraction layer as False
    base_model.conv1.conv1.weight.requires_grad = False

    return base_model


def embed_alexnet(model_to_embed, pretrained=True):
    """

    :param pretrained:
    :param model_to_embed:
    :return:
    """
    base_model = torchvision_models.alexnet(pretrained=pretrained)

    if not pretrained:
        # Load the edge Extraction of the original model
        alexnet_edge_detect_layer = torchvision_models.alexnet(pretrained=True).features[0]
        base_model.features[0].weight.data.copy_(alexnet_edge_detect_layer.weight.data)

    # Replace the first edge extraction layer of the contour integration model with the one from base resnet.
    model_to_embed.conv1 = base_model.features[0]
    model_to_embed.conv1.bias = None   # Original Alexnet used a bias in the first layer. Turn it off.

    # Replace the edge extraction layer with edge extraction + contour integration model
    base_model.features[0] = model_to_embed

    if pretrained:
        # Only Train the Contour Integration Layer
        for c_idx, child in enumerate(base_model.children()):
            if c_idx >= 1:
                for p_idx, param in enumerate(child.parameters()):
                    param.requires_grad = False
            elif c_idx == 0:
                for gc_idx, grand_child in enumerate(child.children()):
                    if gc_idx != 0:
                        for p_idx, param in enumerate(grand_child.parameters()):
                            param.requires_grad = False

    # Set the requires gradient parameter of the first edge extraction layer as False
    base_model.features[0].conv1.weight.requires_grad = False

    return base_model


if __name__ == '__main__':

    # -----------------------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------------------
    print(">>> Building the model {}".format('.' * 80))

    # Contour Integration model
    import models.piech_models
    cont_int_model = models.piech_models.CurrentSubtractiveInhibition(use_class_head=False)

    # # # Control Model
    # # import models.control_models
    # # cont_int_model = models.control_models.CmMatchParameters(use_class_head=False)
    #
    net = embed_resnet50(cont_int_model)
    # net = embed_alexnet(cont_int_model, pretrained=False)

    # Train_imagenet stores the not the whole state of everything. Not just the weights.
    # this is similar to how resume option is used in the train imagenet script
    print("Loading model weights")

    saved_model = \
        './results/imagenet_classification/' \
        'Resnet50_20190907_162401_pretrained_with_contour_integration/best_accuracy.pth'

    checkpoint = torch.load(saved_model)
    net.load_state_dict(checkpoint['state_dict'])

    # Allow all layers to be trained:
    for c_idx, child in enumerate(net.children()):
        for p_idx, param in enumerate(child.parameters()):
            param.requires_grad = True

    # net = torchvision_models.resnet50(pretrained=True)

    print(">>> Starting main script {}".format('.'*80))
    check_requires_grad(net)
    main(net)
