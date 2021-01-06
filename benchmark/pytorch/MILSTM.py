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


if __name__ == "__main__":
    print(torch.backends.cudnn.is_available())
    torch.backends.cudnn.enabled = True
    batch_size = 1
    input_size = 512

    model = MILSTM(input_size=input_size).cuda().half()
    model.eval()

    img_tensor = torch.rand([batch_size, input_size], dtype=torch.float16).cuda()

    # warm up
    out = model(img_tensor)

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
        if i == repeats - 1:
            print("Average inference latency for MILSTM fp16", np.mean(time_record))
    print("batch = ", batch_size)
