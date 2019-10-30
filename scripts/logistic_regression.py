import os
import torch
import pickle
import torchvision
from pathlib import Path
from torch.autograd import Variable

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
        batch_size=100,
        num_workers=0,
        shuffle=True
    )
    return train_loader, train_dataset

def load_test_dataset():
    data_path = './datasets/test'
    testTransform  = torchvision.transforms.Compose([torchvision.transforms.Grayscale(num_output_channels=1),
                                    torchvision.transforms.ToTensor()])
    test_dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=testTransform
    )
    test_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=100,
        num_workers=0,
        shuffle=True
    )
    return test_loader, test_dataset

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

if __name__ == "__main__":
    # Load data
    train_loader, train_dataset = load_train_dataset()
    test_loader, test_dataset = load_test_dataset()

    # System params
    batch_size = 100
    n_iters = 3000
    epochs = n_iters / (len(train_dataset) / batch_size)
    input_dim = 16384
    output_dim = 100
    lr_rate = 0.001
    print(epochs)

    # Load variables
    model = LogisticRegression(input_dim, output_dim).cuda()
    criterion = torch.nn.CrossEntropyLoss() # computes softmax and then the cross entropy
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_rate)

    iter = 0
    for epoch in range(int(epochs)):
        print("Epoch", epoch)
        for i, (images, labels) in enumerate(train_loader):
            print(i, end=",")
            images = Variable(images.view(-1, 128 * 128).cuda())
            labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            iter+=1
            if iter%500==0:
                # calculate Accuracy
                correct = 0
                total = 0
                for images, labels in test_loader:
                    images = Variable(images.view(-1, 128*128))
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total+= labels.size(0)
                    # for gpu, bring the predicted and labels back to cpu fro python operations to work
                    correct+= (predicted == labels).sum()
                accuracy = 100 * correct/total
                print("Iteration: {}. Loss: {}. Accuracy: {}.".format(iter, loss.item(), accuracy))
                storemodel(model, "epoch_" + str(epoch))
        print()
