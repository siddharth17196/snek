import os
import pickle
import torch
import torch.nn as nn
from pathlib import Path
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms

def savePickle(filename, data):
    with open(filename, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def loadPickle(filename):
    with open(filename, 'rb') as handle:
        b = pickle.load(handle)
    return b

def checkandcreatedir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def storemodel(model, name):
    root = str(Path(__file__).parent.parent)
    modeldir = root + "/models"
    checkandcreatedir(modeldir)
    filepath = modeldir + "/" + name
    savePickle(filepath, model)

def loadmodel(filename):
    root = str(Path(__file__).parent.parent)
    modeldir = root + "/models"
    filename = modeldir + "/" + filename
    try:
        model = loadPickle(filename)
        return model
    except:
        raise Exception("Model not found: " + filename )

def load_train_dataset(input_size, crop_size, batch_size):
    data_path = './datasets/train'
    trainTransform  = torchvision.transforms.Compose([
                        torchvision.transforms.Resize((input_size, input_size)),
                        torchvision.transforms.CenterCrop(crop_size),
                        torchvision.transforms.ToTensor(),
                        transforms.Normalize([0.0432, 0.0554, 0.0264], [0.8338, 0.8123, 0.7803]),
                    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=trainTransform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return train_loader, train_dataset

def load_test_dataset(input_size, crop_size, batch_size):
    data_path = './datasets/valid'
    trainTransform  = torchvision.transforms.Compose([
                        torchvision.transforms.Resize((input_size, input_size)),
                        torchvision.transforms.CenterCrop(crop_size),
                        torchvision.transforms.ToTensor(),
                        transforms.Normalize([0.0432, 0.0554, 0.0264], [0.8338, 0.8123, 0.7803]),
                    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=trainTransform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=True
    )
    return train_loader, train_dataset

input_size = 256  # DenseNet Characteristic
crop_size = 224
batch_size = 16

train_loader, train_dataset = load_train_dataset(input_size, crop_size, batch_size)
test_loader, test_dataset = load_test_dataset(input_size, crop_size, batch_size)

n_iters = 2000
num_epochs = n_iters / (len(train_dataset) / batch_size)
num_epochs = 20

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = crop_size*crop_size*3
output_dim = 100

model = LogisticRegressionModel(input_dim, output_dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()

learning_rate = 0.001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0
training_losses = []
for epoch in range(num_epochs):
    print('epoch', epoch)
    for i, (images, labels) in enumerate(train_loader):

        images = images.view(-1, 3*crop_size*crop_size).requires_grad_().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        training_losses.append(loss.item())
        optimizer.step()
        iter += 1
        if iter % 500 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.view(-1, 3*crop_size*crop_size).to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100 * correct.item() / total
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
            storemodel(training_losses, "losses"+str(epoch)+"_"+str(iter))
