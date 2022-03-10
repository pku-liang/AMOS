import torch
import numpy as np
import torch.nn as nn
import math
from torch import optim
import torch.nn.functional as F
import argparse

class MobileNetv1(nn.Module):
    def __init__(self):
        super(MobileNetv1, self).__init__()
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup, track_running_stats=False),
                nn.ReLU(inplace=True)
            )
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp, track_running_stats=False),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup, track_running_stats=False),
                nn.ReLU(inplace=True),
            )
        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


example_text = """
    example:
        python mobile_v1e.py --batch 1 --enable_cudnn --number 5 --repeats 5
        python mobile_v2.py --batch 1 --number 8 --repeats 8
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--enable_cudnn', action='store_true')
    parser.add_argument('--number', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=10)

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False
    
    batch_size = args.batch

    # The following code is used for profiling with dummy data input on CUDA 0
    model = MobileNetv1().cuda().half()
    model.eval()

    img_tensor = torch.rand([batch_size, 3, 224, 224], dtype=torch.float16).cuda()
    
    # warm up
    out = model(img_tensor)

    number = args.number
    repeats = args.repeats
    for i in range(repeats):
        time_record = []
        for j in range(number):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            output = model(img_tensor)

            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
        if i == repeats - 1:
            print("Average inference latency for mobile_v1 fp16", np.mean(time_record))
    print("batch = ", batch_size)
