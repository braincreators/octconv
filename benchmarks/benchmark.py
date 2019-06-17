import time

import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50

from benchmarks.models.resnets import oct_resnet50
from octconv import OctConv2d


@torch.no_grad()
def benchmark_conv():
    x = torch.rand(1, 3, 224, 224)

    conv1 = nn.Conv2d(3, 64, 3)
    conv2 = OctConv2d(3, 64, 3, alpha=(0., 0.5))

    if torch.cuda.is_available():
        x = x.cuda()
        conv1 = conv1.cuda()
        conv2 = conv2.cuda()

    t0 = time.time()
    conv1(x)
    t1 = time.time()
    conv2(x)
    t2 = time.time()

    conv_time = t1 - t0
    octconv_time = t2 - t1

    print("Conv2D:", conv_time)
    print("OctConv2D:", octconv_time)
    print("ratio:", conv_time / octconv_time * 100)


@torch.no_grad()
def benchmark_resnet50():
    x = torch.rand(1, 3, 224, 224)

    model1 = resnet50()
    model2 = oct_resnet50()

    if torch.cuda.is_available():
        x = x.cuda()
        model1 = model1.cuda()
        model2 = model2.cuda()

    t0 = time.time()
    model1(x)
    t1 = time.time()
    model2(x)
    t2 = time.time()

    conv_time = t1 - t0
    octconv_time = t2 - t1

    print("ResNet50:", conv_time)
    print("OctResNet50:", octconv_time)
    print("ratio:", conv_time / octconv_time * 100)


if __name__ == '__main__':
    benchmark_conv()
    print("*" * 30)
    benchmark_resnet50()
