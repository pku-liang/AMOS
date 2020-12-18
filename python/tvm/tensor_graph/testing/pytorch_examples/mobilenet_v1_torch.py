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

def train_perf(device=0):
    model = MobileNetv1().cuda("cuda:" + str(device))
    model.train()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch_size, 3, 224, 224]).astype(dtype)
    img_tensor = torch.tensor(img).cuda("cuda:" + str(device))
    label_tensor = torch.empty(batch_size, dtype=torch.long).random_(1000).cuda("cuda:" + str(device))
    model(img_tensor)
    number = 10
    repeats = 10

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.002)

    for i in range(number):
        time_record = []
        for j in range(repeats):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            optimizer.zero_grad()
            lltm_output = model(img_tensor)
            loss = criterion(lltm_output, label_tensor)
            loss.backward()
            optimizer.step()

            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
        print("Average training latency", np.mean(time_record))
        print("Median training latency", np.median(time_record))
    print("batch = ", batch_size)


def inference_perf(device=0):
    model = MobileNetv1().cuda("cuda:" + str(device))
    model.eval()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch_size, 3, 224, 224]).astype(dtype)
    img_tensor = torch.tensor(img).cuda("cuda:" + str(device))
    model(img_tensor)
    number = 10
    repeats = 10
    
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
        print("Average inference latency", np.mean(time_record))
        print("Median inference latency", np.median(time_record))
    print("batch = ", batch_size)

if __name__ == "__main__":
    # The following code is used for profiling with dummy data input on CUDA 0
    print(torch.backends.cudnn.is_available())
    torch.backends.cudnn.enabled = True
    batch_size = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train", action="store_true")

    args = parser.parse_args()

    if args.train:
        train_perf(args.device)
    else:
        inference_perf(args.device)

# Inference     AVG,        MED
# batch = 1:  5.1404 ms,  5.1026 ms
# CUDNN:      5.1272 ms,  5.1220 ms

# Training
# batch = 1:   11.2906 ms,  11.1114 ms
# batch = 16:  35.5023 ms,  34.5283 ms
# batch = 32:  66.1076 ms,  66.0224 ms
# batch = 64: 129.2857 ms, 129.1059 ms

# cuDNN training
# batch = 1:   12.2147 ms, 11.6393 ms
# batch = 16:  18.2949 ms, 18.4172 ms
# batch = 32:  32.9708 ms, 32.9405 ms
# batch = 64:  62.4251 ms, 62.3493 ms