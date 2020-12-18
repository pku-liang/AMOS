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


class SCRNNCell(nn.Module):
    def __init__(self, input_size = 28*28, num_units = 128, context_units = 64, alpha = 0.5):
        super(SCRNNCell, self).__init__()
        self._input_size = input_size
        self._num_units = num_units
        self._context_units = context_units
        self._alpha = alpha
        self.B = nn.Parameter(torch.empty(input_size, context_units))
        self.V = nn.Parameter(torch.empty(context_units, num_units))
        self.U = nn.Parameter(torch.empty(num_units, num_units))
        self.fc = nn.Linear(context_units + input_size + num_units, num_units, bias=False)
        self.reset_parameters()

    def forward(self, inputs, state_h, state_c):
        # inputs: [batch, 28*28]
        # state_h : [batch, 128]
        # state_c : [batch, 64]
        # self.B : [28*28, 64]
        # (inputs @ self.B) : [batch, 64]
        # context_state : [batch, 64] -> next state_c
        
        # concated: [batch, 64+28*28+128]
        # FC layer
        # hidden_state : [batch, 128]
        # self.U: [128, 128]
        # self.V: [64, 128]
        # new_h: [batch, 128] -> next state_h

        context_state = (1 - self._alpha) * (inputs @ self.B) + self._alpha * state_c
        concated = torch.cat([context_state, inputs, state_h], dim=1)
        hidden_state = torch.sigmoid(self.fc(concated))
        new_h = hidden_state @ self.U + context_state @ self.V
        return new_h, context_state

    def reset_parameters(self):
        for weight in self.parameters():
            nn.init.xavier_uniform_(weight, gain=1.0)


class SCRNN(nn.Module):
    def __init__(self, num_units=128, context_units=64, n_class=10):
        super(SCRNN, self).__init__()
        self.num_units = num_units
        self.context_units = context_units
        self.lstm = SCRNNCell(num_units=num_units, context_units=context_units)
        self.classifier = nn.Linear(num_units, n_class)
        self.h = None
        self.c = None

    def forward(self, x):
        if self.h is None:
            zeroh = Variable(torch.zeros(batch_size, self.num_units, dtype=x.dtype, device=x.device))
            zeroc = Variable(torch.zeros(batch_size, self.context_units, dtype=x.dtype, device=x.device))
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

    model = SCRNN()  # 图片大小是28x28
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
    model = SCRNN().cuda("cuda:" + str(device))
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
    model = SCRNN().cuda("cuda:" + str(device))
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

# Inference:     AVG,      MED
# batch = 1:  0.45107 ms, 0.44952 ms
# cuDNN:      0.42915 ms, 0.42445 ms

# Training:
# batch = 1:  1.1935 ms, 1.1914 ms
# batch = 16: 1.3181 ms, 1.3235 ms
# batch = 32: 1.4581 ms, 1.3758 ms
# batch = 64: 1.4698 ms, 1.4699 ms

# cuDNN training:
# batch = 1:  1.2553 ms, 1.2554 ms
# batch = 16: 1.3650 ms, 1.3614 ms
# batch = 32: 1.3404 ms, 1.3409 ms
# batch = 64: 1.3334 ms, 1.3379 ms

# ON SCC with one GPU
# RESULT of function MINST_train()
# batch_size = 64
# Finish 3 epoch
# Test Loss: 0.142585, Acc: 0.95520
# train time median: INACCURATE 1.271 ms
# infer time meidan: INACCURATE 0.3655 ms