import argparse
import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorchLab')
parser.add_argument('-d', '--datadir', required=True)

args = parser.parse_args()

print(args.datadir)
data_dir = args.datadir

# train_transform = transforms.Compose([transforms.ToTensor()])
input_size = 256
crop_size = 224

train_transform  = torchvision.transforms.Compose([torchvision.transforms.Resize((input_size, input_size)),
                        torchvision.transforms.CenterCrop(crop_size),
                        torchvision.transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                        ])
train_set = torchvision.datasets.ImageFolder(root=data_dir, transform=train_transform)
#print(vars(train_set))
print(len(train_set))

loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=10,
    num_workers=1,
    shuffle=False
)

mean = 0.
std = 0.
nb_samples = 0.
i = 0
for data, label in loader:
    try:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
        print(i, mean, std)
        i+=1
    except:
        continue

mean /= nb_samples
std /= nb_samples

# print(train_set.train_data.mean(axis=(0,1,2))/255)
# print(train_set.train_data.std(axis=(0,1,2))/255)

# elif args.dataset == "cifar100":
#     train_transform = transforms.Compose([transforms.ToTensor()])
#     train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
#     #print(vars(train_set))
#     print(train_set.train_data.shape)
#     print(np.mean(train_set.train_data, axis=(0,1,2))/255)
#     print(np.std(train_set.train_data, axis=(0,1,2))/255)

# elif args.dataset == "mnist":
#     train_transform = transforms.Compose([transforms.ToTensor()])
#     train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=train_transform)
#     #print(vars(train_set))
#     print(list(train_set.train_data.size()))
#     print(train_set.train_data.float().mean()/255)
#     print(train_set.train_data.float().std()/255)