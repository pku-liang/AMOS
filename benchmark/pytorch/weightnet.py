import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

class WeightNet(nn.Module):
    # https://github.com/megvii-model/WeightNet/blob/669b5f4c0c46fd30cd0fedf5e5a63161e9e94bcc/weightnet.py

    def __init__(self, inp=24, oup=216, ksize=1, stride=1):
        super().__init__()

        self.M = 2
        self.G = 2

        self.pad = ksize // 2
        inp_gap = inp // 16
        self.inp = inp
        self.oup = oup
        self.ksize = ksize
        self.stride = stride

        self.wn_fc1 = nn.Conv2d(inp_gap, self.M*oup, kernel_size=ksize, stride=stride, padding=0, groups=1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.wn_fc2 = nn.Conv2d(self.M*oup, oup*inp*ksize*ksize, kernel_size=ksize, stride=stride, padding=0, groups=self.G*oup, bias=False)


    def forward(self, x, x_gap):
        x_w = self.wn_fc1(x_gap)
        x_w = self.sigmoid(x_w)
        x_w = self.wn_fc2(x_w)
        x_w = x_w.reshape(self.oup, self.inp, self.ksize, self.ksize)
        x = F.conv2d(x, weight=x_w, stride=self.stride, padding=self.pad)
        return x

example_text = """
    example:
        python weightnet.py --enable_cudnn --number 10 --repeats 5
        python weightnet.py --number 10 --repeats 10
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="base_maker",
        description="template maker",
        epilog=example_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--enable_cudnn', action='store_true')
    parser.add_argument('--number', type=int, default=10)
    parser.add_argument('--repeats', type=int, default=10)

    args = parser.parse_args()

    if args.enable_cudnn:
        assert torch.backends.cudnn.is_available()
        torch.backends.cudnn.enabled = True
    else:
        torch.backends.cudnn.enabled = False
    
    batch_size = 1 # Only target batch_size = 1

    model = WeightNet().cuda().half()
    model.eval()

    x = torch.rand([batch_size, 24, 1, 1], dtype=torch.float16).cuda()
    x_gap = torch.rand([batch_size, 1, 1, 1], dtype=torch.float16).cuda()

    # warm up
    out = model(x, x_gap)

    number = args.number
    repeats = args.repeats
    for i in range(repeats):
        time_record = []
        for j in range(number):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            output = model(x, x_gap)

            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
        if i == repeats - 1:
            print("Average inference latency for WeightNet fp16", np.mean(time_record))
    print("batch = ", batch_size)
