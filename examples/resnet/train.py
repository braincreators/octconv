import time

import torch
import torch.nn as nn
from configargparse import ArgumentParser
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from examples.resnet.models import oct_resnet20, oct_resnet50, oct_resnet101, oct_resnet152
from examples.utils import AverageMeter, ProgressMeter, save_checkpoint, adjust_learning_rate, accuracy

models = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'oct_resnet20': oct_resnet20,
    'oct_resnet50': oct_resnet50,
    'oct_resnet101': oct_resnet101,
    'oct_resnet152': oct_resnet152
}


def get_model(arch):
    return models[arch]()


def make_data_loader(root, batch_size, workers=4, is_train=True, download=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if is_train:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        dataset = ImageNet(root=root, split='train', download=download, transform=transform)

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=workers,
                            pin_memory=True)
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        dataset = ImageNet(root=root, split='val', download=download, transform=transform)

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=workers,
                            pin_memory=True)

    return loader


def train(train_loader, model, criterion, optimizer, epoch, gpu=None, print_freq=10):
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
    for i, (input_data, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if gpu is not None:
            input_data = input_data.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

        # compute output
        output = model(input_data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input_data.size(0))
        top1.update(acc1[0], input_data.size(0))
        top5.update(acc5[0], input_data.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.print(i)


def validate(val_loader, model, criterion, gpu, print_freq=10):
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
        for i, (input_data, target) in enumerate(val_loader):
            if gpu is not None:
                input_data = input_data.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

            # compute output
            output = model(input_data)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input_data.size(0))
            top1.update(acc1[0], input_data.size(0))
            top5.update(acc5[0], input_data.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def main():
    parser = ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-c', '--config', is_config_file=True, help='training configuration file')
    parser.add_argument('--root', required=True, help='path to ImageNet dataset')
    parser.add_argument('-a', '--arch', default='resnet18', choices=models.keys(),
                        help='model architecture: {} (default: resnet18)'.format(' | '.join(models.keys())))
    parser.add_argument('-j', '--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', dest='start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=256, type=int,
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', dest='lr', default=0.1, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, dest='weight_decay',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('-p', '--print-freq', default=10, type=int, dest='print_freq',
                        help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')

    args = parser.parse_args()

    train_loader = make_data_loader(root=args.root,
                                    batch_size=args.batch_size,
                                    workers=args.workers,
                                    download=True,
                                    is_train=True)

    val_loader = make_data_loader(root=args.root,
                                  batch_size=args.batch_size,
                                  workers=args.workers,
                                  download=True,
                                  is_train=False)

    model = get_model(args.arch)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    best_acc1 = 0

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args.gpu, args.print_freq)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args.gpu, args.print_freq)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


if __name__ == '__main__':
    main()
