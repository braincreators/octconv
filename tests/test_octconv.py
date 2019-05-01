import pytest
import torch
from octconv import OctConv2d


@torch.no_grad()
def test_forward_single_input_stride1():
    x = torch.rand(2, 3, 200, 200)  # (b, c, h, w)
    conv = OctConv2d(in_channels=3, out_channels=10, kernel_size=3, alpha=(0., 0.5), padding=1)
    out_h, out_l = conv(x)

    shape_h = tuple(out_h.shape)
    shape_l = tuple(out_l.shape)

    assert shape_h == (2, 5, 200, 200)
    assert shape_l == (2, 5, 100, 100)


@torch.no_grad()
def test_forward_single_input_stride2():
    x = torch.rand(2, 3, 200, 200)  # (b, c, h, w)
    conv = OctConv2d(in_channels=3, out_channels=10, kernel_size=3, stride=2, alpha=(0., 0.5), padding=1)
    out_h, out_l = conv(x)

    shape_h = tuple(out_h.shape)
    shape_l = tuple(out_l.shape)

    assert shape_h == (2, 5, 100, 100)
    assert shape_l == (2, 5, 50, 50)


@torch.no_grad()
def test_forward_split_input():
    x_h = torch.rand(2, 2, 200, 200)  # (b, c, h, w)
    x_l = torch.rand(2, 3, 100, 100)  # (b, c, h, w)
    conv = OctConv2d(in_channels=5, out_channels=10, kernel_size=3, alpha=(0.5, 0.5), padding=1)
    out_h, out_l = conv((x_h, x_l))

    shape_h = tuple(out_h.shape)
    shape_l = tuple(out_l.shape)

    assert shape_h == (2, 5, 200, 200)
    assert shape_l == (2, 5, 100, 100)


@torch.no_grad()
def test_forward_wrong_shapes():
    x_h = torch.rand(2, 3, 200, 200)  # (b, c, h, w)
    x_l = torch.rand(2, 3, 100, 100)  # (b, c, h, w)
    conv = OctConv2d(in_channels=5, out_channels=10, kernel_size=3, alpha=(0.5, 0.5), padding=1)

    with pytest.raises(AssertionError):
        _ = conv((x_h, x_l))


@torch.no_grad()
def test_forward_cascade():
    x = torch.rand(2, 3, 200, 200)  # (b, c, h, w)
    conv1 = OctConv2d(in_channels=3, out_channels=10, kernel_size=3, alpha=(0., 0.5), padding=1)
    conv2 = OctConv2d(in_channels=10, out_channels=20, kernel_size=7, alpha=(0.5, 0.8), padding=3)
    conv3 = OctConv2d(in_channels=20, out_channels=1, kernel_size=3, alpha=(0.8, 0.), padding=1)

    out = conv3(conv2(conv1(x)))

    shape = tuple(out.shape)
    assert shape == (2, 1, 200, 200)


if __name__ == '__main__':
    test_forward_single_input_stride2()
