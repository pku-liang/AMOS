# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse

from torchvision import datasets
import torchvision.transforms as transforms

class ConvLayer(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=256):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=1, padding=0)

    def forward(self, x):
        conved = self.conv(x)
        features = F.relu(conved)
        return features

class PrimaryCaps(nn.Module):
    
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32):
        super(PrimaryCaps, self).__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=9, stride=2, padding=0)
            for _ in range(num_capsules)])
    
    def forward(self, x):
        # get batch size of inputs
        batch_size = x.size(0)
        u = [capsule(x).view(batch_size, 32 * 6 * 6, 1) for capsule in self.capsules]
        u = torch.cat(u, dim=-1)
        u_squash = self.squash(u)
        return u_squash
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)    
        return output_tensor
    

def softmax(input_tensor, dim=2):
    transposed_input = input_tensor.transpose(dim, len(input_tensor.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input_tensor.size()) - 1)


# dynamic routing
def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    for iteration in range(routing_iterations):
        c_ij = softmax(b_ij, dim=2)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
        v_j = squash(s_j)
        if iteration < routing_iterations - 1:
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            b_ij = b_ij + a_ij
    return v_j

TRAIN_ON_GPU = torch.cuda.is_available()

if(TRAIN_ON_GPU):
    print('Training on GPU!')
else:
    print('Only CPU available')


class DigitCaps(nn.Module):
    
    def __init__(self, num_capsules=10, previous_layer_nodes=32*6*6, 
                 in_channels=8, out_channels=16):
        super(DigitCaps, self).__init__()
        self.num_capsules = num_capsules
        self.previous_layer_nodes = previous_layer_nodes
        self.in_channels = in_channels

        self.W = nn.Parameter(torch.randn(num_capsules, previous_layer_nodes, 
                                          in_channels, out_channels))

    def forward(self, u):
        u = u[None, :, :, None, :]
        W = self.W[:, None, :, :, :]
        u_hat = torch.matmul(u, W)
        b_ij = torch.zeros_like(u_hat)
        # if TRAIN_ON_GPU:
        #     b_ij = b_ij.cuda()

        v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)

        return v_j
    
    
    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)    
        return output_tensor


class CapsuleNetwork(nn.Module):
    
    def __init__(self):
        super(CapsuleNetwork, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
                
    def forward(self, images):
        primary_caps_output = self.primary_capsules(self.conv_layer(images))
        caps_output = self.digit_capsules(primary_caps_output).squeeze().transpose(0,1)
        # squeeze can will delete all 1 in dims, which is unexpected
        if batch == 1:
            caps_output = caps_output.reshape(batch, 10, 16)
        return caps_output
    

capsule_net = CapsuleNetwork()

print(capsule_net)

if TRAIN_ON_GPU:
    capsule_net = capsule_net.cuda()


class CapsuleLoss(nn.Module):
    
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, x, labels, images=None, reconstructions=None):
        batch_size = x.size(0)
        v_c = torch.sqrt((x**2).sum(dim=2))       
        left = F.relu(0.9 - v_c).view(batch_size, -1)
        right = F.relu(v_c - 0.1).view(batch_size, -1)
        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()
        return margin_loss


criterion = CapsuleLoss()

optimizer = optim.Adam(capsule_net.parameters())


def train_perf(device=0):
    in_channel = 1
    model = CapsuleNetwork().cuda("cuda:" + str(device))
    model.train()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, in_channel, 28, 28]).astype(dtype)
    img_tensor = torch.tensor(img).cuda("cuda:" + str(device))
    label_tensor = torch.rand([batch, 10]).cuda("cuda:" + str(device))
    model(img_tensor)
    number = 10
    repeats = 10

    criterion = CapsuleLoss()

    optimizer = optim.SGD(model.parameters(), lr=0.002)

    for i in range(number):
        time_record = []
        for j in range(repeats):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()

            optimizer.zero_grad()
            caps_output = model(img_tensor)
            images, reconstructions = 0, 0
            loss = criterion(caps_output, label_tensor, images, reconstructions)
            loss.backward()
            optimizer.step()

            end.record()
            torch.cuda.synchronize()
            total = start.elapsed_time(end)
            time_record.append(total)
        print("Average training latency", np.mean(time_record))
        print("Median training latency", np.median(time_record))
    print("batch = ", batch)


def inference_perf(device=0):
    in_channel = 1
    model = CapsuleNetwork().cuda("cuda:" + str(device))
    model.eval()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch, in_channel, 28, 28]).astype(dtype)
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
    print("batch = ", batch)


if __name__ == "__main__":
    print(torch.backends.cudnn.is_available())
    torch.backends.cudnn.enabled = True
    batch = 64
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train", action="store_true")

    args = parser.parse_args()

    if args.train:
        train_perf(args.device)
    else:
        inference_perf(args.device)

# Profile for inference:
#                 Average, Median
# batch = 1:   2.3226 ms,  2.3212 ms
# cuDNN:       2.4476 ms,  2.4453 ms

# Profile for training:
# batch = 1:   6.8429 ms,  6.6831 ms
# batch = 16: 34.9058 ms, 34.8554 ms
# batch = 32: 67.6183 ms, 67.5679 ms
# batch = 64: 134.0412ms, 133.8435ms

# cuDNN
# batch = 1:   7.2370 ms,  7.1752 ms
# batch = 16: 12.4554 ms, 12.4093 ms
# batch = 32: 18.4121 ms, 18.4115 ms
# batch = 64: 30.1592 ms, 30.1281 ms
