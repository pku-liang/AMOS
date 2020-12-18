import math
import torch
import torch.nn.functional as F
import time
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import numpy as np
import argparse

torch.manual_seed(42)

import torch.nn as nn


class MILSTM_Cell(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=1024):
        super(MILSTM_Cell, self).__init__()

        self.hidden_size = hidden_size
        # lstm weights
        self.weight_fh = nn.Linear(hidden_size, hidden_size)
        self.weight_ih = nn.Linear(hidden_size, hidden_size)
        self.weight_zh = nn.Linear(hidden_size, hidden_size)
        self.weight_oh = nn.Linear(hidden_size, hidden_size)
        self.weight_fx = nn.Linear(input_size, hidden_size)
        self.weight_ix = nn.Linear(input_size, hidden_size)
        self.weight_zx = nn.Linear(input_size, hidden_size)
        self.weight_ox = nn.Linear(input_size, hidden_size)
        # alphas and betas
        self.alpha_f = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_f1 = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_f2 = nn.Parameter(torch.ones(1,hidden_size))
        
        self.alpha_i = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_i1 = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_i2 = nn.Parameter(torch.ones(1,hidden_size))
        
        self.alpha_o = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_o1 = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_o2 = nn.Parameter(torch.ones(1,hidden_size))
        
        self.alpha_z = nn.Parameter(torch.ones(1,hidden_size))
        self.alpha_z = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_z1 = nn.Parameter(torch.ones(1,hidden_size))
        self.beta_z2 = nn.Parameter(torch.ones(1,hidden_size))


    def forward(self, inp, h_0, c_0):
        # inp : [batch, 28*28]
        # gates : [batch, hidden_size]

        # forget gate
        f_g = torch.sigmoid(self.alpha_f * self.weight_fx(inp) * self.weight_fh(h_0) +
                       (self.beta_f1 * self.weight_fx(inp)) + (self.beta_f2 * self.weight_fh(h_0)))
        # input gate
        i_g = torch.sigmoid(self.alpha_i * self.weight_ix(inp) * self.weight_ih(h_0) +
                       (self.beta_i1 * self.weight_ix(inp)) + (self.beta_i2 * self.weight_ih(h_0)))
        # output gate
        o_g = torch.sigmoid(self.alpha_o * self.weight_ox(inp) * self.weight_oh(h_0) +
                       (self.beta_o1 * self.weight_ox(inp)) + (self.beta_o2 * self.weight_oh(h_0)))
        # block input
        z_t = torch.tanh(self.alpha_z * self.weight_zx(inp) * self.weight_zh(h_0) +
                    (self.beta_z1 * self.weight_zx(inp)) + (self.beta_z2 * self.weight_zh(h_0)))
        # current cell state
        cx = f_g * c_0 + i_g * z_t
        # hidden state
        hx = o_g * torch.tanh(cx)

        return hx, cx

class MILSTM(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=1024, n_class=10):
        super(MILSTM, self).__init__()
        self.n_class = n_class
        self.hidden_size = hidden_size
        self.lstm = MILSTM_Cell(input_size=input_size, hidden_size=hidden_size)
        self.classifier = nn.Linear(hidden_size, n_class)
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            zeroh = Variable(torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device))
            zeroc = Variable(torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device))
            self.h = zeroh
            self.c = zeroc
        new_h, new_c = self.lstm(x, self.h, self.c)
        self.h, self.c = Variable(new_h), Variable(new_c)
        out = self.classifier(new_h)
        return out

def MINST_train():
    # This profiling is inaccurate and deprecated!
    learning_rate = 1e-3
    num_epoches = 3

    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transforms.ToTensor(), download=False)

    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transforms.ToTensor(), download=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MILSTM()  # 图片大小是28x28
    use_gpu = torch.cuda.is_available()
    assert use_gpu == True
    if use_gpu:
        model = model.cuda()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_time_record = []
    infer_time_record = []
    # 开始训练
    for epoch in range(num_epoches):
        print('epoch {}'.format(epoch + 1))
        print('*' * 10)
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):
            img, label = data
            b, c, h, w = img.size()
            assert c == 1, 'channel must be 1'
            if b != batch_size:
                print("discarded for training")
                continue
            img = img.squeeze(1)
            img = img.view(batch_size, 28*28)
            if use_gpu:
                img = Variable(img).cuda()
                #print(img.size())
                label = Variable(label).cuda()
            else:
                img = Variable(img)
                label = Variable(label)
            
            start_train_time = time.time()
            # forward
            out = model(img)
            # compute loss
            # lable: torch.Size(batch_size)
            loss = criterion(out, label)
            
            # back-prop
            optimizer.zero_grad()
            loss.backward()

            train_time_record.append(time.time() - start_train_time)

            optimizer.step()

            running_loss += loss.item() * label.size(0)
            _, pred = torch.max(out, 1)
            num_correct = (pred == label).sum()
            running_acc += num_correct.item()

            if i % 30000 == 0:
                print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                    epoch + 1, num_epoches, running_loss / (batch_size * i),
                    running_acc / (batch_size * i)))
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
                train_dataset))))
        with torch.no_grad():
            #model.eval()
            eval_loss = 0.
            eval_acc = 0.
            for data in test_loader:
                img, label = data
                b, c, h, w = img.size()
                assert c == 1, 'channel must be 1'
                if b != batch_size:
                    print("discarded for testing")
                    continue
                img = img.squeeze(1)
                img = img.view(batch_size, 28*28)
                if use_gpu:
                    img = Variable(img, volatile=True).cuda()
                    label = Variable(label, volatile=True).cuda()
                else:
                    img = Variable(img, volatile=True)
                    label = Variable(label, volatile=True)
                
                start_infer_time = time.time()
                # only forward
                out = model(img)

                infer_time_record.append(time.time() - start_infer_time)

                loss = criterion(out, label)
                eval_loss += loss.item() * label.size(0)
                _, pred = torch.max(out, 1)
                num_correct = (pred == label).sum()
                eval_acc += num_correct.item()
            print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
                test_dataset)), eval_acc / (len(test_dataset))))
    print("train time median:", np.median(train_time_record), "max", np.max(train_time_record))
    print("infer time meidan:", np.median(infer_time_record), "max", np.max(infer_time_record))


def train_perf(device=0):
    model = MILSTM().cuda("cuda:" + str(device))
    model.train()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch_size, 28*28]).astype(dtype)
    img_tensor = torch.tensor(img).cuda("cuda:" + str(device))
    label_tensor = torch.empty(batch_size, dtype=torch.long).random_(10).cuda("cuda:" + str(device))
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
            images, reconstructions = 0, 0
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
    model = MILSTM().cuda("cuda:" + str(device))
    model.eval()
    dtype = "float32"
    img = np.random.uniform(-1, 1, [batch_size, 28*28]).astype(dtype)
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
    # If you want to train with MNIST data, you can call MINST_train()
    # The following code is used for profiling with dummy data input on CUDA 0
    print(torch.backends.cudnn.is_available())
    torch.backends.cudnn.enabled = True
    batch_size = 64
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train", action="store_true")

    args = parser.parse_args()

    if args.train:
        train_perf(args.device)
    else:
        inference_perf(args.device)

# Inference:       AVG,      MED
# batch = 1:    1.6901 ms  1.6824 ms
# cuDNN:        1.6742 ms, 1.6676 ms

# Training:
# batch = 1:    4.8209 ms, 4.6960 ms
# batch = 16:   5.1443 ms, 5.0918 ms
# batch = 32:   5.3921 ms, 5.2705 ms
# batch = 64:   5.4895 ms, 5.4789 ms

# cuDNN training:
# batch = 1:    5.0158 ms, 4.9914 ms
# batch = 16:   5.1001 ms, 5.1010 ms
# batch = 32:   5.3670 ms, 5.3325 ms
# batch = 64:   5.4030 ms, 5.4415 ms

# ON SCC with one GPU
# RESULT of function MINST_train()
# batch_size = 64
# Finish 3 epoch
# Test Loss: 0.081103, Acc: 0.973700
# train time median: INACCURATE 5.347 ms
# infer time meidan: INACCURATE 1.604 ms