import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision

def load_train_dataset():
    data_path = './datasets/train_resized'
    trainTransform  = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor()])
    train_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=trainTransform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=400,
        num_workers=0,
        shuffle=True
    )
    return train_loader, train_dataset

def load_test_dataset():
    data_path = './datasets/valid'
    testTransform  = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor()])
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=testTransform
    )
    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=400,
        num_workers=0,
        shuffle=True
    )
    return test_loader, test_dataset

train_loader, train_dataset = load_train_dataset()
test_loader, test_dataset = load_test_dataset()

batch_size = 50
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

input_dim = 16384
output_dim = 100

model = LogisticRegressionModel(input_dim, output_dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()


learning_rate = 0.001

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0
for epoch in range(num_epochs):
    print('epoch', epoch)
    for i, (images, labels) in enumerate(train_loader):

        images = images.view(-1, 128*128).requires_grad_().to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        iter += 1
        if iter % 500 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.view(-1, 128*128).to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)

                if torch.cuda.is_available():
                    correct += (predicted.cpu() == labels.cpu()).sum()
                else:
                    correct += (predicted == labels).sum()

            accuracy = 100 * correct.item() / total
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
