import time
import torch
import torch.nn as nn
from octconv import OctConv2d


with torch.no_grad():
    x = torch.rand(1, 3, 224, 224)

    conv1 = nn.Conv2d(3, 64, 3)
    conv2 = OctConv2d(3, 64, 3, alpha=(0., 0.5))

    if torch.cuda.is_available():
        x = x.cuda()
        conv1 = conv1.cuda()
        conv2 = conv2.cuda()

    t0 = time.time()
    out = conv1(x)
    t1 = time.time()
    out2 = conv2(x)
    t2 = time.time()

    conv_time = t1 - t0
    octconv_time = t2 - t1

    print("Conv2D:", conv_time)
    print("OctConv2D:", octconv_time)
    print("ratio:", conv_time / octconv_time * 100)
