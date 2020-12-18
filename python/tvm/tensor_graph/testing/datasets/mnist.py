import torch
import torchvision
from torchvision import transforms


def load_mnist_dataset(train_batch_size, test_batch_size=1):
    train_set = torchvision.datasets.MNIST(".", train=True, transform=transforms.Compose([transforms.ToTensor()]), download=True)
    test_set = torchvision.datasets.MNIST(".", train=False, transform=transforms.Compose([transforms.ToTensor()]), download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader
