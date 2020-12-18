import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import time
random_cnt = 0

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        global random_cnt
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=False)
        # [6, 1, 5, 5]
        np.random.seed(random_cnt)
        random_cnt += 1
        weight1 = np.random.uniform(-1, 1, [6, 1, 5, 5]).astype("float32")
        self.conv1.weight = torch.nn.Parameter(torch.tensor(weight1))

        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=False)
        # [16, 6, 5, 5]
        np.random.seed(random_cnt)
        random_cnt += 1
        weight2 = np.random.uniform(-1, 1, [16, 6, 5, 5]).astype("float32")
        self.conv3.weight = torch.nn.Parameter(torch.tensor(weight2))

        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0, bias=False)
        # [120, 16, 5, 5]
        np.random.seed(random_cnt)
        random_cnt += 1
        weight3 = np.random.uniform(-1, 1, [120, 16, 5, 5]).astype("float32")
        self.conv5.weight = torch.nn.Parameter(torch.tensor(weight3))

        self.fc6 = nn.Linear(120, 84, bias=False)
        # [84, 120]
        np.random.seed(random_cnt)
        random_cnt += 1
        weight4 = np.random.uniform(-1, 1, [84, 120]).astype("float32")
        self.fc6.weight = torch.nn.Parameter(torch.tensor(weight4))

        self.output = nn.Linear(84, 10, bias=False)
        # [10, 84]
        np.random.seed(random_cnt)
        random_cnt += 1
        weight5 = np.random.uniform(-1, 1, [10, 84]).astype("float32")
        self.output.weight = torch.nn.Parameter(torch.tensor(weight5))
        
        self.relu = nn.ReLU()
        print("random_cnt", random_cnt)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.relu(x)
        x = self.s2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.s4(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = x.mean(-1).mean(-1)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.output(x)
        return x


class LeNet5Repeat(nn.Module):
    def __init__(self):
        super(LeNet5Repeat, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, bias=False)
        self.conv1_1 = nn.Conv2d(6, 6, kernel_size=3, padding=1, bias=False)
        self.conv1_2 = nn.Conv2d(6, 6, kernel_size=3, padding=1, bias=False)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, bias=False)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv5 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0, bias=False)

        self.fc6 = nn.Linear(120, 84, bias=False)
        self.output = nn.Linear(84, 10, bias=False)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.relu(x)
        x = self.s2(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.s4(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = x.mean(-1).mean(-1)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.output(x)
        return x

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

    model = LeNet5()
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
            # This code snippet should be useful somewhere!
            label_torch = torch.tensor(np.zeros([batch_size, 10]).astype(dtype))
            label_torch.scatter_(1, label.unsqueeze(0).T, 1.0)
            label_numpy = label_torch.numpy()
            # print(label_numpy)
            b, c, h, w = img.size()
            assert c == 1, 'channel must be 1'
            if b != batch_size:
                print("discarded for training")
                continue
            # img = img.squeeze(1)
            # img = img.view(batch_size, 28*28)
            img = torch.tensor(np.pad(img, ((0,0),(0,0), (2,2),(2,2)), 'constant')) # 28 -> 32
            if use_gpu:
                img = Variable(img).cuda() # [batch, 1, 28->32, 28->32]
                label = Variable(label).cuda() # torch.Size([4])
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
                # img = img.squeeze(1)
                # img = img.view(batch_size, 28*28)
                img = torch.tensor(np.pad(img, ((0,0),(0,0), (2,2),(2,2)), 'constant')) # 28 -> 32
                if use_gpu:
                    img = Variable(img).cuda()
                    label = Variable(label).cuda()
                else:
                    img = Variable(img)
                    label = Variable(label)
                
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

def lenet5():
    model = LeNet5()
    return model


def lenet5_repeat():
    model = LeNet5Repeat()
    return model


if __name__ == "__main__":
    batch_size = 1
    dtype = "float32"
    # MINST_train()
    # exit(0)
    np.random.seed(random_cnt)
    random_cnt += 1
    img = np.random.uniform(-1, 1, [batch_size, 1, 32, 32]).astype(dtype)
    img_tensor = torch.tensor(img).cuda()

    model = LeNet5().cuda()
    result = model(img_tensor)
    print(result)

# The following is reproducible!
# random_cnt 6
# tensor([[ -68.8599,  649.2622,  -82.8930,  277.4005,  226.7232, -588.3690,
#          -459.3513, -312.9294, -584.4176,   35.0095]], device='cuda:0',
#        grad_fn=<MmBackward>)

# batch_size = 4, MINST_train
# Finish 1 epoch, Loss: 6.781759, Acc: 0.890683
# Test Loss: 0.365122, Acc: 0.936400