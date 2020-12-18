import torch
import numpy as np
import torch.nn as nn
import math
import argparse
import torch.nn.functional as F
from torch import optim


def shuffle_channels(x, groups):
    """shuffle channels of a 4-D Tensor"""
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0
    channels_per_group = channels // groups
    # split into groups
    x = x.view(batch_size, groups, channels_per_group,
               height, width)
    # transpose 1, 2 axis
    x = x.transpose(1, 2).contiguous()
    # reshape into orignal
    x = x.view(batch_size, channels, height, width)
    return x

class ShuffleNetUnitA(nn.Module):
    """ShuffleNet unit for stride=1"""
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitA, self).__init__()
        assert in_channels == out_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                        1, groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels, track_running_stats=False)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=1,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels, track_running_stats=False)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        out = F.relu(x + out)
        return out

class ShuffleNetUnitB(nn.Module):
    """ShuffleNet unit for stride=2"""
    def __init__(self, in_channels, out_channels, groups=3):
        super(ShuffleNetUnitB, self).__init__()
        out_channels -= in_channels
        assert out_channels % 4 == 0
        bottleneck_channels = out_channels // 4
        self.groups = groups
        self.group_conv1 = nn.Conv2d(in_channels, bottleneck_channels,
                                     1, groups=groups, stride=1)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels, track_running_stats=False)
        self.depthwise_conv3 = nn.Conv2d(bottleneck_channels,
                                         bottleneck_channels,
                                         3, padding=1, stride=2,
                                         groups=bottleneck_channels)
        self.bn4 = nn.BatchNorm2d(bottleneck_channels, track_running_stats=False)
        self.group_conv5 = nn.Conv2d(bottleneck_channels, out_channels,
                                     1, stride=1, groups=groups)
        self.bn6 = nn.BatchNorm2d(out_channels, track_running_stats=False)

    def forward(self, x):
        out = self.group_conv1(x)
        out = F.relu(self.bn2(out))
        out = shuffle_channels(out, groups=self.groups)
        out = self.depthwise_conv3(out)
        out = self.bn4(out)
        out = self.group_conv5(out)
        out = self.bn6(out)
        x = F.avg_pool2d(x, 3, stride=2, padding=1)
        out = F.relu(torch.cat([x, out], dim=1))
        return out

class ShuffleNet(nn.Module):
    """ShuffleNet for groups=3"""
    def __init__(self, groups=3, in_channels=3, num_classes=1000):
        super(ShuffleNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 24, 3, stride=2, padding=1)
        stage2_seq = [ShuffleNetUnitB(24, 240, groups=3)] + \
            [ShuffleNetUnitA(240, 240, groups=3) for i in range(3)]
        self.stage2 = nn.Sequential(*stage2_seq)
        stage3_seq = [ShuffleNetUnitB(240, 480, groups=3)] + \
            [ShuffleNetUnitA(480, 480, groups=3) for i in range(7)]
        self.stage3 = nn.Sequential(*stage3_seq)
        stage4_seq = [ShuffleNetUnitB(480, 960, groups=3)] + \
                     [ShuffleNetUnitA(960, 960, groups=3) for i in range(3)]
        self.stage4 = nn.Sequential(*stage4_seq)
        self.fc = nn.Linear(960, num_classes)

    def forward(self, x):
        net = self.conv1(x)
        net = F.avg_pool2d(net, 3, stride=2, padding=1)
        net = self.stage2(net)
        net = self.stage3(net)
        net = self.stage4(net)
        net = F.avg_pool2d(net, 7)
        net = net.view(net.size(0), -1)
        net = self.fc(net)
        logits = F.softmax(net)
        return logits

def train_perf(device=0):
    model = ShuffleNet().cuda("cuda:" + str(device))
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
    model = ShuffleNet().cuda("cuda:" + str(device))
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
    torch.backends.cudnn.enabled = False
    batch_size = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train", action="store_true")

    args = parser.parse_args()

    if args.train:
        train_perf(args.device)
    else:
        inference_perf(args.device)

# Inference:     AVG,         MED
# batch = 1:   15.8537 ms,  15.8541 ms
# cuDNN:       11.0424 ms,  10.9711 ms 

# Train
# batch = 1:   58.8737 ms,   56.7214 ms
# batch = 16: 139.0346 ms,  131.4253 ms
# batch = 32: 207.1157 ms,  208.8940 ms
# batch = 64: 372.4677 ms,  374.2950 ms

# Training cuDNN
# batch = 1:   32.3326 ms,  31.4808 ms
# batch = 16:  30.6563 ms,  30.3892 ms
# batch = 32:  34.6896 ms,  33.6399 ms
# batch = 64:  49.7290 ms,  49.5022 ms