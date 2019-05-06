import datetime
import os
import time
import pprint
import logging

import torch
import torch.distributed as dist
import torch.utils.data
from configargparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet, CIFAR10
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

import benchmarks.utils as utils
from benchmarks.models.resnets import oct_resnet20, oct_resnet50, oct_resnet101, oct_resnet152

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


def get_model(arch, **kwargs):
    return models[arch](**kwargs)


def make_data_loader(root,
                     batch_size,
                     dataset='imagenet',
                     workers=4,
                     is_train=True,
                     download=False,
                     distributed=False):
    if dataset == 'imagenet':
        loader = _make_data_loader_imagenet(root=root, batch_size=batch_size, workers=workers,
                                            is_train=is_train, download=download, distributed=distributed)
    elif dataset == 'cifar10':
        loader = _make_data_loader_cifar10(root=root, batch_size=batch_size, workers=workers,
                                           is_train=is_train, download=download, distributed=distributed)
    else:
        raise ValueError('Invalid dataset name')

    return loader


def _make_data_loader_imagenet(root,
                               batch_size,
                               workers=4,
                               is_train=True,
                               download=False,
                               distributed=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    logger = logging.getLogger('octconv')

    if is_train:
        logger.info("Loading ImageNet training data")

        st = time.time()
        scale = (0.08, 1.0)

        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        dataset = ImageNet(root=root, split='train', download=download, transform=transform)

        logger.info("Took", time.time() - st)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=workers,
                            sampler=sampler,
                            pin_memory=True)
    else:
        logger.info("Loading ImageNet validation data")

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        dataset = ImageNet(root=root, split='val', download=download, transform=transform)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=workers,
                            sampler=sampler,
                            pin_memory=True)

    return loader


def _make_data_loader_cifar10(root,
                              batch_size,
                              workers=4,
                              is_train=True,
                              download=False,
                              distributed=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    logger = logging.getLogger('octconv')

    if is_train:
        logger.info('Loading CIFAR10 Training')

        dataset = CIFAR10(root=root, train=True,
                          download=download, transform=transform)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.RandomSampler(dataset)

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=sampler,
                                             num_workers=workers)
    else:
        logger.info('Loading CIFAR10 Test')

        dataset = CIFAR10(root=root, train=False,
                          download=download, transform=transform)

        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             sampler=sampler,
                                             num_workers=workers)

    return loader


def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = utils.get_world_size()

    if world_size < 2:
        return loss_dict

    with torch.no_grad():
        loss_names = []
        all_losses = []

        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])

        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)

        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size

        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}

    return reduced_losses


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    meters = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    logger = logging.getLogger("octconv")

    for image, target in meters.log_every(data_loader, logger, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)

        loss = criterion(output, target)

        loss_dict = {'loss': loss}
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]

        meters.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        meters.meters['acc1'].update(acc1.item(), n=batch_size)
        meters.meters['acc5'].update(acc5.item(), n=batch_size)


def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    logger = logging.getLogger('octconv')
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, logger, 100, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    logger.info(' * Acc@1 {top1.global_avg:.3f} Acc@5 '
                '{top5.global_avg:.3f}'.format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg


def main(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        utils.synchronize()

    logger = utils.setup_logger("octconv", args.output_dir, utils.get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("Arguments: {}".format(pprint.pformat(args.__dict__)))

    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True

    # Data loading code
    logger.info("Loading data")

    data_loader = make_data_loader(root=args.root,
                                   batch_size=args.batch_size,
                                   dataset=args.dataset,
                                   workers=args.workers,
                                   is_train=True,
                                   download=args.download,
                                   distributed=args.distributed)

    data_loader_test = make_data_loader(root=args.root,
                                        batch_size=args.batch_size,
                                        dataset=args.dataset,
                                        workers=args.workers,
                                        is_train=False,
                                        download=args.download,
                                        distributed=args.distributed)

    if args.dataset == 'imagenet':
        num_classes = 1000
    elif args.dataset == 'cifar10':
        num_classes = 10
    else:
        raise ValueError('Invalid dataset')

    logger.info("Creating model")
    model = get_model(args.arch, alpha=args.alpha, num_classes=num_classes)
    model.to(device)

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    logger.info(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
        model_without_ddp = model.module

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    if args.test_only:
        evaluate(model, criterion, data_loader_test, device=device)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):

        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        lr_scheduler.step()
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq)
        evaluate(model, criterion, data_loader_test, device=device)
        utils.synchronize()

        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = ArgumentParser(description='ImageNet training')

    parser.add_argument('-c', '--config', is_config_file=True, help='config file')
    parser.add_argument('--root', required=True, help='dataset')
    parser.add_argument('--dataset', default='imagenet', help='dataset name')
    parser.add_argument('--download', action='store_true', default=False, help='download ImageNet')
    parser.add_argument('--arch', default='resnet18', help='model')
    parser.add_argument('--alpha', default=0.125, type=float, help='OctConv alpha parameter')
    parser.add_argument('--device', default='cuda', help='GPU device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=30, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=10, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='./output', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--test-only", dest="test_only", action="store_true", default=False, help="Only test the model")

    # distributed training parameters
    parser.add_argument("--local_rank", type=int, default=0)

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
