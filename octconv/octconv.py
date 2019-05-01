import torch.nn as nn
import torch.nn.functional as F


class OctConv2d(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 alpha=0.5,
                 bias=False):

        super(OctConv2d, self).__init__()

        assert isinstance(in_channels, int) and in_channels > 0
        assert isinstance(out_channels, int) and out_channels > 0
        assert isinstance(kernel_size, int) and kernel_size > 0
        assert stride in {1, 2}, "Only strides of 1 and 2 are currently supported"

        if isinstance(alpha, tuple):
            assert len(alpha) == 2
            assert all([0 <= a <= 1 for a in alpha]), "Alphas must be in interval [0, 1]"
            self.alpha_in, self.alpha_out = alpha
        else:
            assert 0 <= alpha <= 1, "Alpha must be in interval [0, 1]"
            self.alpha_in = alpha
            self.alpha_out = alpha

        # in_channels
        in_ch_hf = int((1 - self.alpha_in) * in_channels)
        self.in_channels = {
            'high': in_ch_hf,
            'low': in_channels - in_ch_hf
        }

        # out_channels
        out_ch_hf = int((1 - self.alpha_out) * out_channels)
        self.out_channels = {
            'high': out_ch_hf,
            'low': out_channels - out_ch_hf
        }

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.conv_h2h = nn.Conv2d(in_channels=self.in_channels['high'],
                                  out_channels=self.out_channels['high'],
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  bias=bias) \
            if not (self.alpha_in == 1 or self.alpha_out == 1) else None

        self.conv_h2l = nn.Conv2d(in_channels=self.in_channels['high'],
                                  out_channels=self.out_channels['low'],
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  bias=bias) \
            if not (self.alpha_in == 1 or self.alpha_out == 0) else None

        self.conv_l2h = nn.Conv2d(in_channels=self.in_channels['low'],
                                  out_channels=self.out_channels['high'],
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  bias=bias) \
            if not (self.alpha_in == 0 or self.alpha_out == 1) else None

        self.conv_l2l = nn.Conv2d(in_channels=self.in_channels['low'],
                                  out_channels=self.out_channels['low'],
                                  kernel_size=kernel_size,
                                  padding=padding,
                                  bias=bias) \
            if not (self.alpha_in == 0 or self.alpha_out == 0) else None

    def forward(self, x):
        x_h, x_l = x if isinstance(x, tuple) else (x, None)

        self._check_inputs(x_h, x_l)

        x_l2l, x_l2h = None, None

        # High -> High
        x_h = self.pool(x_h) if self.stride == 2 else x_h
        x_h2h = self.conv_h2h(x_h) if self.out_channels['high'] > 0 else None

        # High -> Low
        x_h2l = self.pool(x_h) if self.out_channels['low'] > 0 else x_h
        x_h2l = self.conv_h2l(x_h2l) if self.out_channels['low'] > 0 else None

        if x_l is not None:
            # Low -> Low
            x_l2l = self.pool(x_l) if (self.out_channels['low'] > 0 and self.stride == 2) else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.out_channels['low'] > 0 else None

            # Low -> High
            x_l2h = self.conv_l2h(x_l) if self.out_channels['high'] > 0 else None
            x_l2h = F.interpolate(x_l2h, size=x_h2h.shape[-2:]) \
                if (self.out_channels['high'] > 0 and self.stride == 1) else x_l2h

        x_h = x_h2h + x_l2h if x_l2h is not None else x_h2h
        x_l = x_l2l + x_h2l if x_l2l is not None else x_h2l

        output = (x_h, x_l)

        return output[0] if output[1] is None else output

    def _check_inputs(self, x_h, x_l):
        assert x_h.dim() == 4

        if x_l is not None:
            assert x_l.dim() == 4

        if self.in_channels['high'] > 0:
            assert x_h.shape[1] == self.in_channels['high']

        if self.in_channels['low'] > 0:
            assert x_l.shape[1] == self.in_channels['low']

    def __repr__(self):
        s = """{}(in_channels=(low: {}, high: {}), out_channels=(low: {}, high: {}), 
          kernel_size=({kernel}, {kernel}), stride=({stride}, {stride}), 
          padding={}, alphas=({}, {}), bias={})""".format(
            self._get_name(), self.in_channels['low'], self.in_channels['high'],
            self.out_channels['low'], self.out_channels['high'],
            self.padding, self.alpha_in, self.alpha_out, self.bias,
            kernel=self.kernel_size, stride=self.stride)

        return s
