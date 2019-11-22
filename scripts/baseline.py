import torch
import torchvision
import random
import matplotlib.pyplot as plt
import os
import copy
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--traindir', required=True)
parser.add_argument('-v', '--validdir', required=True)
parser.add_argument('-b', '--batchsize', required=False, default=32)

args = parser.parse_args()
train_data_dir = args.traindir
valid_data_dir = args.validdir

train_data = datasets.ImageFolder(train_data_dir)
num_classes = len(train_data.classes)
model_name = "densenet"  # resnet, vgg or densenet
input_size = 256  # DenseNet Characteristic
crop_size = 224
batch_size = args.batchsize
feature_extract = False

def load_train_dataset(root, batchsize):
    data_path = root
    trainTransform  = torchvision.transforms.Compose([torchvision.transforms.Resize((input_size, input_size)),
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
        batch_size=batchsize,
        num_workers=4,
        shuffle=True
    )
    return train_loader, train_dataset

image_datasets = {}
dataloaders_dict = {}
dataloaders_dict['train'], image_datasets['train'] = load_train_dataset(train_data_dir, batch_size)
dataloaders_dict['valid'], image_datasets['valid'] = load_train_dataset(valid_data_dir, batch_size)

class_names = image_datasets['train'].classes
# print(len(class_names))
# print(type(dataloaders_dict["train"]))
# for i, (images, labels) in enumerate(dataloaders_dict["train"]):
#     print(i)
# exit()

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# **Dataset Visualization:** To get an idea of which kind of images are we dealing with, we will visualize a random mini-batch of our training data.
def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.show()
    # plt.pause(0.001)

# Get a mini-batch of training data
mini_batch = 4
dataloaders_dict_visualize = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=mini_batch, shuffle=True, num_workers=4) for x in ['train']}
it = iter(dataloaders_dict_visualize['train'])
inputs, classes = next(it)

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)
# imshow(out)
# %% [markdown]
# # 3. Model Definition
# In this section we will define the training and the initialization of the network as well as its configuration.
# 
# This notebook enables the user to choose between three network architectures: DenseNet, VGG and ResNet. However, the training is performed on **DenseNet-121** (https://arxiv.org/pdf/1608.06993.pdf), for it is the current State Of The Art Network for image classification (it obtains a 82,8% of accuracy on the ImageNet benchmark).  It was presented in 2017 CVPR where it got Best Paper Award and has more than 2000 citations. It is jointly invented by Cornwell University, Tsinghua University and Facebook AI Research (FAIR).
# 
# An interesting review of this network can be found at: https://towardsdatascience.com/review-densenet-image-classification-b6631a8ef803
# 
# %% [markdown]
# **Training and Initialization Definition**

# %%
def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_fscore = 0.0
    
    loss_train_evo=[]
    acc_train_evo=[]
    fs_train_evo=[]
    
    loss_val_evo=[]
    acc_val_evo=[]
    fs_val_evo=[]
    
    total_train=round(47626/batch_size)

    for epoch in range(num_epochs):
        i = 0
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            print("phase:", phase)
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            fscore = []

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train

                with torch.set_grad_enabled(phase == 'train'):
                    """
                    if i==round(0.25*total_train):
                        print('Forward Passed 25%')
                    if i==round(0.5*total_train):
                        print('Forward Passed 50%')
                    if i==round(0.75*total_train):
                        print('Forward Passed 75%')
                    i = i + 1
                    """
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                labels_cpu = labels.cpu().numpy()
                predictions_cpu = preds.cpu().numpy()
                Fscore = f1_score(labels_cpu, predictions_cpu, average='macro')
                fscore.append(Fscore)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_fscore = np.average(np.array(fscore))
            
            print('{} Loss: {:.4f} Acc: {:.4f} F: {:.3f}'.format(phase, epoch_loss, epoch_acc, epoch_fscore))

            torch.save(model.state_dict(), "./model/model"+str(epoch))
            torch.save(optimizer.state_dict(), "./model/optimizer"+str(epoch))
            
            if phase == 'train':
                loss_train_evo.append(epoch_loss)
                epoch_acc = epoch_acc.cpu().numpy()
                acc_train_evo.append(epoch_acc)
                fs_train_evo.append(epoch_fscore)                
            else:
                loss_val_evo.append(epoch_loss)
                epoch_acc = epoch_acc.cpu().numpy()
                acc_val_evo.append(epoch_acc)
                fs_val_evo.append(epoch_fscore) 
                
            # deep copy the model
            if phase == 'valid' and epoch_fscore > best_fscore:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, loss_train_evo, acc_train_evo, fs_train_evo, loss_val_evo, acc_val_evo, fs_val_evo

# sets the .requires_grad attribute of the parameters in the model to False when we are feature extracting
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    model_ft = None
    input_size = 0

    if model_name == "resnet":  # ResNet-50
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":  # VGG-11
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "densenet":  # DenseNet-121
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    return model_ft, input_size

# %% [markdown]
# **Model Initialization and Configuration:** For this baseline, whe chose **Cross-Entropy Loss** as the training loss and the **ADAM optimizer** - a combination of Adaptive Gradient Algorithm (AdaGrad) and Root Mean Square Propagation (RMSProp) - with a static learning rate for backpropagation.

# %%
# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Send the model to GPU
model_ft = model_ft.to(device)

params_to_update = model_ft.parameters()
if feature_extract:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            # print("\t",name)
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            a=1 # print("\t",name)
            
# Optimizer
optimizer_ft = optim.Adam(params_to_update, lr=3e-4)

# Loss Funciton
criterion = nn.CrossEntropyLoss()

# **Training and Validation:** Network is trained for a 15 epochs on GCLOUD. The initial idea was to fine tune the model on this Jupyter Notebook. However, due to its wrong memory management, sometimes the training stopped after a few epochs and model did not end its training. Therefore, model has been directly run on GCLOUD terminal and then its metrics printed and hardcoded on this notebook to be able to visualize them. Although we are aware that this is not an elegant solution, we use it because it has shown to be effective. 
# *Here, a training sample of two epochs is followingly shown*.  

num_epochs = 20
model_ft, loss_train, acc_train, fs_train, loss_val, acc_val, fs_val = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)
# Save model
# torch.save(model_ft.state_dict(),'/mnt/disks/dades/model_baseline.pth')

# %% [markdown]
# **Metrics Evolution Visualization:** We consider the loss, the accuracy and the f1 score to be the three most relevant metrics for our task. In order to visualize their evolution, we define a function that compares its training values with its validation ones.  

# %%
def plot_metric(metric_train, metric_val, title):
    fig, (ax) = plt.subplots(1, 1)
    fig.suptitle(title)
    ax.set(xlabel='epoch')
    ax.plot(metric_train, label='Training')
    ax.plot(metric_val, label='Validation')
    ax.legend(loc='upper left')
    plt.savefig("metric.png")

"""
# Results for a GCLOUD Terminal training for 20 epochs

loss_train = [1.97, 1.47, 1.26, 1.08,  0.95,  0.82,  0.72,  0.62,  0.54,  0.48,  0.43,  0.38,  0.34,  0.32,  0.30,  0.26,  0.25,  0.23,  0.22,  0.21]
loss_val =  [1.58, 1.40, 1.30, 1.27, 1.28, 1.21, 1.34, 1.31, 1.38, 1.47, 1.46, 1.58, 1.57, 1.60, 1.71, 1.72, 1.71, 1.76, 1.78, 1.81]

acc_train = [0.45, 0.57, 0.63, 0.68, 0.71, 0.74, 0.77, 0.80, 0.82, 0.85, 0.86, 0.88, 0.89, 0.90, 0.90, 0.91, 0.92, 0.92, 0.93, 0.94]
acc_val = [0.54, 0.59, 0.62, 0.63, 0.64, 0.65, 0.64, 0.64, 0.64, 0.63, 0.63, 0.64, 0.64, 0.64, 0.63, 0.63, 0.63, 0.62, 0.63, 0.62]

fs_train = [0.30,  0.42,  0.48,  0.53,  0.57,  0.61,  0.65,  0.69,  0.72,  0.75,  0.77,  0.79,  0.82,  0.83,  0.83,  0.85,  0.86,  0.87,  0.88,  0.89]
fs_val = [0.38,  0.44,  0.46,  0.48,  0.49,  0.51,  0.49,  0.49,  0.48,  0.48,  0.49,  0.49,  0.48,  0.49,  0.48,  0.48,  0.50,  0.48,  0.48, 0.47]

"""

plot_metric(loss_train, loss_val, 'Loss')
plot_metric(acc_train, acc_val, 'Accuracy')
plot_metric(fs_train, fs_val, 'F-Score')

# **Model Loading:** We load our previously saved model to perform a test inference. This loading is necessary if we have not run the training on the Jupyter Notebook, for in this case we will not have the model saved in our current memory. 
# Load the pretrained model
# model_weights_path = '/mnt/disks/dades/model_baseline.pth'
# model_weights = torch.load(model_weights_path)
# model_ft.load_state_dict(model_weights)
# print('Model Loaded')

# # %% [markdown]
# # **Infer Test Data:** We infer to our model a random image from the test set and visualize it with its predicted class. We do not have the test set labels, for we cannot evaluate our model's behaviour on this particular image. However, an interesting option is to look for the predicted class in Google Images and see if the images shown are similar to the one infered to our network...

# # %%
# def get_item():
#     test_files = os.listdir('/mnt/disks/dades/test/')
#     idx = random.randint(0,len(test_files))
#     ima_dir = os.path.join('/mnt/disks/dades/test/',test_files[idx])
#     scaler = transforms.Resize((224, 224))
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#     to_tensor = transforms.ToTensor()
#     ima = Image.open(ima_dir)
#     plt.imshow(ima)
#     item = (normalize(to_tensor(scaler(ima))).unsqueeze(0)).to(device)
#     return item

# class_name = ['Thamnophis Proximus', 'Nerodia Sipedon', 'Opheodrys Vernalis', 'Crotalus Horridus', 'Crotalus Pyrrhus', 'Nerodia Rhombifer', 'Thamnophis Sirtalis', 'Natrix Natrix', 'Crotalus Adamanteus', 'Charina Bottae', 'Pituophis Catenifer', 'Lampropeltis Triangulum', 'Nerodia Erythrogaster', 'Thamnophis Marcianus', 'Lampropeltis Californiae', 'Crotalus Ruber', 'Rhinocheilus Lecontei', 'Opheodrys Aestivus', 'Thamnophis Ordinoides', 'Thamnophis Radix', 'Masticophis Flagellum', 'Pantherophis Vulpinus', 'Hierophis Viridiflavus', 'Feterodon Platirhinos', 'Pantherophis Emoryi', 'Regina Septemvittata', 'Haldea Striatula', 'Diadophis Punctatus', 'Nerodia Fasciata', 'Storeria Occipitomaculata', 'Crotalus Scutulatus', 'Storeria Dekayi', 'Crotalus Viridis', 'Boa Imperator', 'Pantherophis Obsoletus', 'Lichanura Trivirgata', 'Agkistrodon Contortrix', 'Thamnophis Elegans', 'Agkistrodon Piscivorus', 'Pantherophis Guttatus', 'Crotalus Atrox', 'Carphophism Amoenus', 'Coluber Constrictor', 'Pantherophis Spiloides', 'Pantherophis Alleghaniensis']
# item = get_item()
# model_ft.eval()
# output = model_ft(item)
# _, preds = torch.max(output, 1)
# class_id = preds.item()
# print("Predicted class: ", class_name[class_id])

# # We concluded that our baseline network performed decently on the task of snakes species identification with just a few epochs, reaching an maximum accuracy of 65% on the validation set. However, it turned to be highly **overfitted**, for it did not manage to generalize well. To try to prevent this overfitting, many actions could be taken such as applying a regularization, data augmentation, dropout and so on. Some of these features will be tackled in an advanced notebook named *Enhanced.ipynb*. 
