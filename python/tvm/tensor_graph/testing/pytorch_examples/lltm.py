'''
Source code partially from https://github.com/pytorch/extension-cpp/blob/master/python/lltm.py
Tutorial at https://pytorch.org/tutorials/advanced/cpp_extension.html#motivation-and-example
'''
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


class LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.Tensor(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.Tensor(1, 3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_cell = state
        X = torch.cat([old_h, input], dim=1)

        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(3, dim=1)

        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])

        # Compute the new cell state.
        new_cell = old_cell + candidate_cell * input_gate
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate

        return new_h, new_cell

class RnnLLTM(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_class):
        super(RnnLLTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = LLTM(in_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_class)
        self.hx = None        

    def forward(self, x):
        if self.hx is None:
            zeros = Variable(torch.zeros(batch_size, self.hidden_dim,dtype=x.dtype, device=x.device))
            self.hx = (zeros, zeros)
        new_h, new_c = self.lstm(x, self.hx)
        self.hx = (Variable(new_h), Variable(new_c))
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

    model = RnnLLTM(28*28, 128, 10)  # 图片大小是28x28
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
    model = RnnLLTM(28*28, 128, 10).cuda("cuda:" + str(device))
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
    model = RnnLLTM(28*28, 128, 10).cuda("cuda:" + str(device))
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
    batch_size = 64
    print(torch.backends.cudnn.is_available())
    torch.backends.cudnn.enabled = True
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--train", action="store_true")

    args = parser.parse_args()

    if args.train:
        train_perf(args.device)
    else:
        inference_perf(args.device)


# Accurate profiling:
# inference:
# batch_size = 1:  AVG 0.36775 ms, MED 0.36489 ms
# CUDNN enabled:   AVG 0.34417 ms, MED 0.34200 ms

# training:
#                        AVG,    MED
# batch_size = 1:  1.07535 ms   1.06907 ms
# batch_size = 16: 1.18323 ms  1.19298 ms
# batch_size = 32: 1.23516 ms  1.23029 ms
# batch_size = 64: 1.39452 ms  1.38880 ms

# CUDNN enabled
# batch_size = 1:   1.20473 ms   1.20933 ms
# batch_size = 16:  1.33939 ms   1.24672 ms
# batch_size = 32:  1.36828 ms   1.37678 ms
# batch_size = 64:  1.44004 ms   1.43976 ms


# ON SCC with one GPU
# RESULT of function MINST_train()
# batch_size = 1, epoch = 3
# Test Loss: 0.098778, Acc: 0.972000
# Inaccurate profiling:
# train time median: 1.14 ms INACCURATE
# infer time meidan: 0.293 ms INACCURATE
