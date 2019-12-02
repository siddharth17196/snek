import torch
import torchvision
import random
import matplotlib.pyplot as plt
import os
import copy
import pickle
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torchvision import datasets, models, transforms
from PIL import Image

def load_train_dataset(root, batchsize, input_size, crop_size):
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
        crop_size = 224

    elif model_name == "vgg":  # VGG-11
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        crop_size = 224

    elif model_name == "densenet":  # DenseNet-121
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        crop_size = 224
    
    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained=use_pretrained, aux_logits=False)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299
        crop_size = 299
    
    elif model_name == "resnext":
        model_ft = models.resnext50_32x4d(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        crop_size = 224

    return model_ft, input_size, crop_size

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datadir', required=True)
parser.add_argument('-m', '--model', required=True)
parser.add_argument('-mn', '--modelname', required=False, default="models/")
parser.add_argument('-b', '--batchsize', required=False, default=32)

args = parser.parse_args()
data_dir = args.datadir
model_name = args.modelname
model_dir = args.model
batch_size = args.batchsize

data = datasets.ImageFolder(data_dir)
num_classes = len(data.classes)

feature_extract = False

torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.current_device())

model_ft, input_size, crop_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
loader, dataset = load_train_dataset("./datasets/valid", batch_size, input_size, crop_size)

model_weights = torch.load(model_dir)
print(model_weights)
model_ft.load_state_dict(model_weights)
print('Loaded model', model_dir)
model_ft = model_ft.to(device)

flag=1

y_pred=[]
y_test=[]

for i, (images, labels) in enumerate(loader):
	print(i)

	images = images.to(device)
	labels = labels.to(device)

	y_test.extend(labels)

	output = model_ft(images)
	_, preds = torch.max(output, 1)
	print(preds)
	# class_id = preds.item()
	y_pred.extend(preds)
	# print("Predicted class: ", y_pred, y_test)

f = open('preds_' + model_name + '.pkl', 'wb')
pickle.dump(y_pred, f)
f.close

f = open('tests_' + model_name + '.pkl', 'wb')
pickle.dump(y_test, f)
f.close

exit()

# def get_item(di):
# 	global flag
# 	ima_dir = os.path.join(di)
# 	scaler = transforms.Resize((224, 224))
# 	normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# 	to_tensor = transforms.ToTensor()
# 	try:
# 		ima = Image.open(ima_dir)
# 		plt.imshow(ima)
# 		item = (normalize(to_tensor(scaler(ima))).unsqueeze(0)).to(device)
# 		return item
# 	except:
# 		flag=0
# 		return 0

# s = './datasets/valid/'
# model_ft.eval()
# y_pred=[]
# y_test=[]
# for files in os.listdir('./datasets/valid/'):
# 	f = os.listdir(s+files)
# 	for i in range(len(f)):
# 		flag=1
# 		y_test.append(files)
# 		item = get_item(s+files+'/'+f[i])
# 		if flag==0:
# 			y_pred.append(files)
# 		else:
# 			output = model_ft(item)
# 			_, preds = torch.max(output, 1)
# 			class_id = preds.item()
# 			y_pred.append(class_names[class_id])
# 			print("Predicted class: ", class_names[class_id])

# f = open('predsmod11.pkl', 'wb')
# pickle.dump(y_pred, f)
# f.close

# f = open('testmod11.pkl', 'wb')
# pickle.dump(y_test, f)
# f.close
