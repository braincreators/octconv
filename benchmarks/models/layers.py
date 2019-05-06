import torch.nn as nn
from octconv import OctConv2d


class OctConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=0,
                 bias=False, norm_layer=None):

        super(OctConvBn, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = OctConv2d(in_channels, out_channels, kernel_size=kernel_size,
                              alpha=alpha, stride=stride, padding=padding, bias=bias)

        alpha_out = self.conv.alpha_out

        self.bn_h = None if alpha_out == 1 else norm_layer(self.conv.out_channels['high'])
        self.bn_l = None if alpha_out == 0 else norm_layer(self.conv.out_channels['low'])

    def forward(self, x):
        out = self.conv(x)

        x_h, x_l = out if isinstance(out, tuple) else (out, None)

        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None

        return x_h, x_l


class OctConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha=0.5, stride=1, padding=0,
                 bias=False, norm_layer=None, activation_layer=None):

        super(OctConvBnAct, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation_layer is None:
            activation_layer = nn.ReLU(inplace=True)

        self.conv = OctConv2d(in_channels, out_channels, kernel_size=kernel_size,
                              alpha=alpha, stride=stride, padding=padding, bias=bias)

        alpha_out = self.conv.alpha_out

        self.bn_h = None if alpha_out == 1 else norm_layer(self.conv.out_channels['high'])
        self.bn_l = None if alpha_out == 0 else norm_layer(self.conv.out_channels['low'])

        self.act = activation_layer

    def forward(self, x):
        out = self.conv(x)

        x_h, x_l = out if isinstance(out, tuple) else (out, None)

        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None

        return x_h, x_l


if __name__ == '__main__':
    pass
