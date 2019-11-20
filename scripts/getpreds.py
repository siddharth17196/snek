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
# from sklearn.metrics import f1_score
from PIL import Image


data_dir = "./datasets"
train_data = datasets.ImageFolder(data_dir + "/train")
num_classes = len(train_data.classes)
model_name = "densenet"  # resnet, vgg or densenet
input_size = 256  # DenseNet Characteristic
crop_size = 224
batch_size = 8
feature_extract = False

def load_train_dataset(root, batchsize):
	data_path = root
	trainTransform = torchvision.transforms.Compose([torchvision.transforms.Resize((input_size, input_size)),
													 torchvision.transforms.CenterCrop(crop_size),
													 torchvision.transforms.ToTensor(),
													 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
													 # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
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

loader, dataset = load_train_dataset("./datasets/valid", batch_size)

torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.current_device())

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

print(torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

print(torch.cuda.get_device_name(device))
torch.cuda.empty_cache()
model_weights = torch.load('model11_gray')
print(model_weights)
model_ft.load_state_dict(model_weights)
print('Model Loaded 1')

model_ft = model_ft.to(device)

print('*')

flag=1


def get_item(di):
	global flag
	ima_dir = os.path.join(di)
	scaler = transforms.Resize((224, 224))
	normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
	to_tensor = transforms.ToTensor()
	try:
		ima = Image.open(ima_dir)
		plt.imshow(ima)
		item = (normalize(to_tensor(scaler(ima))).unsqueeze(0)).to(device)
		return item
	except:
		flag=0
		return 0

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

f = open('predsmod11.pkl', 'wb')
pickle.dump(y_pred, f)
f.close

f = open('testmod11.pkl', 'wb')
pickle.dump(y_test, f)
f.close


exit()


s = './datasets/valid/'
# class_name = ['Thamnophis Proximus', 'Nerodia Sipedon', 'Opheodrys Vernalis', 'Crotalus Horridus', 'Crotalus Pyrrhus', 'Nerodia Rhombifer', 'Thamnophis Sirtalis', 'Natrix Natrix', 'Crotalus Adamanteus', 'Charina Bottae', 'Pituophis Catenifer', 'Lampropeltis Triangulum', 'Nerodia Erythrogaster', 'Thamnophis Marcianus', 'Lampropeltis Californiae', 'Crotalus Ruber', 'Rhinocheilus Lecontei', 'Opheodrys Aestivus', 'Thamnophis Ordinoides', 'Thamnophis Radix', 'Masticophis Flagellum', 'Pantherophis Vulpinus', 'Hierophis Viridiflavus', 'Feterodon Platirhinos', 'Pantherophis Emoryi', 'Regina Septemvittata', 'Haldea Striatula', 'Diadophis Punctatus', 'Nerodia Fasciata', 'Storeria Occipitomaculata', 'Crotalus Scutulatus', 'Storeria Dekayi', 'Crotalus Viridis', 'Boa Imperator', 'Pantherophis Obsoletus', 'Lichanura Trivirgata', 'Agkistrodon Contortrix', 'Thamnophis Elegans', 'Agkistrodon Piscivorus', 'Pantherophis Guttatus', 'Crotalus Atrox', 'Carphophism Amoenus', 'Coluber Constrictor', 'Pantherophis Spiloides', 'Pantherophis Alleghaniensis']
model_ft.eval()
y_pred=[]
y_test=[]
for files in os.listdir('./datasets/valid/'):
	f = os.listdir(s+files)
	for i in range(len(f)):
		flag=1
		y_test.append(files)
		item = get_item(s+files+'/'+f[i])
		if flag==0:
			y_pred.append(files)
		else:
			output = model_ft(item)
			_, preds = torch.max(output, 1)
			class_id = preds.item()
			y_pred.append(class_names[class_id])
			print("Predicted class: ", class_names[class_id])

f = open('predsmod11.pkl', 'wb')
pickle.dump(y_pred, f)
f.close

f = open('testmod11.pkl', 'wb')
pickle.dump(y_test, f)
f.close
