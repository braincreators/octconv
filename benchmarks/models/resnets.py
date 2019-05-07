import torch.nn as nn

from benchmarks.models.layers import OctConvBn, OctConvBnAct


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, alpha=0.5, norm_layer=None,
                 first_block=False, last_block=False):

        super(Bottleneck, self).__init__()

        assert not (first_block and last_block), "mutually exclusive options"

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = OctConvBnAct(inplanes, width, kernel_size=1, norm_layer=norm_layer,
                                  alpha=alpha if not first_block else (0., alpha))
        self.conv2 = OctConvBnAct(width, width, kernel_size=3, stride=stride, padding=1,
                                  norm_layer=norm_layer, alpha=alpha)
        self.conv3 = OctConvBn(width, planes * self.expansion, kernel_size=1, norm_layer=norm_layer,
                               alpha=alpha if not last_block else (alpha, 0.))

        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity_h = x[0] if type(x) is tuple else x
        identity_l = x[1] if type(x) is tuple else None

        x_h, x_l = self.conv1(x)
        x_h, x_l = self.conv2((x_h, x_l))
        out = self.conv3((x_h, x_l))

        x_h, x_l = out if isinstance(out, tuple) else (out, None)

        if self.downsample is not None:
            identity = self.downsample(x)
            identity_h, identity_l = identity if isinstance(identity, tuple) else (identity, None)

        x_h += identity_h
        x_l = x_l + identity_l if identity_l is not None else None

        x_h = self.relu(x_h)
        x_l = self.relu(x_l) if x_l is not None else None

        return x_h, x_l


class OctResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, groups=1, width_per_group=64, norm_layer=None, alpha=0.5):
        super(OctResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.alpha = alpha
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, first_layer=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, norm_layer=norm_layer)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, norm_layer=norm_layer, last_layer=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None, first_layer=False, last_layer=False):

        assert not (first_layer and last_layer), "mutually exclusive options"

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if last_layer:
                downsample = nn.Sequential(
                    OctConvBn(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                              alpha=(self.alpha, 0.))
                )
            else:
                downsample = nn.Sequential(
                    OctConvBn(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride,
                              alpha=self.alpha if not first_layer else (0., self.alpha))
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample,
                            groups=self.groups, base_width=self.base_width,
                            alpha=self.alpha, norm_layer=norm_layer,
                            first_block=first_layer, last_block=last_layer))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, norm_layer=norm_layer,
                                alpha=self.alpha if not last_layer else 0.,
                                last_block=last_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_h, x_l = self.layer1(x)
        x_h, x_l = self.layer2((x_h, x_l))
        x_h, x_l = self.layer3((x_h, x_l))
        x_h, x_l = self.layer4((x_h, x_l))

        x = self.avgpool(x_h)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def _oct_resnet(inplanes, planes, **kwargs):
    model = OctResNet(inplanes, planes, **kwargs)
    return model


def oct_resnet50(**kwargs):
    """Constructs a OctResNet-50 model."""
    return _oct_resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def oct_resnet101(**kwargs):
    """Constructs a OctResNet-101 model."""
    return _oct_resnet(Bottleneck, [3, 4, 23, 3], **kwargs)


def oct_resnet152(**kwargs):
    """Constructs a OctResNet-152 model."""
    return _oct_resnet(Bottleneck, [3, 8, 36, 3], **kwargs)


if __name__ == '__main__':
    import torch

    with torch.no_grad():
        x = torch.rand(1, 3, 224, 224)

        model = oct_resnet50()
        out = model(x)
        print(out.shape)

        model = oct_resnet101()
        out = model(x)
        print(out.shape)

        model = oct_resnet152()
        out = model(x)
        print(out.shape)
