import datetime
import os
import time
from pprint import pprint

import torch
import torch.utils.data
from configargparse import ArgumentParser
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageNet
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

import examples.utils as utils
from examples.resnet.models import oct_resnet20, oct_resnet50, oct_resnet101, oct_resnet152

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


def make_data_loader(root, batch_size, workers=4, is_train=True, download=False, distributed=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if is_train:
        print("Loading training data")

        st = time.time()
        scale = (0.08, 1.0)

        transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        dataset = ImageNet(root=root, split='train', download=download, transform=transform)

        print("Took", time.time() - st)

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
        print("Loading validation data")

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


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)


def evaluate(model, criterion, data_loader, device):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
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

    print(' * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    return metric_logger.acc1.global_avg


def main(args):
    utils.init_distributed_mode(args)
    print("Arguments:")
    pprint(args.__dict__)

    device = torch.device(args.device)

    torch.backends.cudnn.benchmark = True

    # Data loading code
    print("Loading data")

    data_loader = make_data_loader(root=args.root,
                                   batch_size=args.batch_size,
                                   workers=args.workers,
                                   is_train=True,
                                   download=args.download,
                                   distributed=args.distributed)

    data_loader_test = make_data_loader(root=args.root,
                                        batch_size=args.batch_size,
                                        workers=args.workers,
                                        is_train=False,
                                        download=args.download,
                                        distributed=args.distributed)

    print("Creating model")
    model = get_model(args.arch)
    model.to(device)

    print(model)

    if args.distributed:
        model = torch.nn.utils.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
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

    print("Start training")
    start_time = time.time()
    for epoch in range(args.epochs):

        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        lr_scheduler.step()
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args.print_freq)
        evaluate(model, criterion, data_loader_test, device=device)
        if args.output_dir:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'args': args},
                os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = ArgumentParser(description='ImageNet training')

    parser.add_argument('-c', '--config', is_config_file=True, help='config file')
    parser.add_argument('--root', required=True, help='dataset')
    parser.add_argument('--download', action='store_true', default=False, help='download ImageNet')
    parser.add_argument('--arch', default='resnet18', help='model')
    parser.add_argument('--device', default='cuda', help='device')
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
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--test-only", dest="test_only", action="store_true", default=False, help="Only test the model")

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)
