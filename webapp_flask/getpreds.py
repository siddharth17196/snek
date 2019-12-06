import torch
import torchvision
import random
import matplotlib.pyplot as plt
import os
import copy
import argparse
import pickle
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torchvision import datasets, models, transforms
from PIL import Image


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

def get_item(loc):
	
	#*****************************************************************************************************************************
	model_name = "densenet" 
	model_dir = "densenet_model" 
	num_classes = 45
	feature_extract = False
	torch.cuda.set_device(0)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model_ft, input_size, crop_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
	model_weights = torch.load(model_dir, map_location=device)
	model_ft.load_state_dict(model_weights)
	model_ft = model_ft.to(device)

	#*****************************************************************************************************************************
	ima_dir = loc
	print(ima_dir)
	scaler = transforms.Resize((224, 224))
	normalize = transforms.Normalize(mean=[0.0432, 0.0554, 0.0264],#[0.485, 0.456, 0.406],#[0.0432, 0.0554, 0.0264], [0.8338, 0.8123, 0.7803]
								 std= [0.8338, 0.8123, 0.7803])#[0.229, 0.224, 0.225])
	to_tensor = transforms.ToTensor()
	ima = Image.open(ima_dir)
	item = (normalize(to_tensor(scaler(ima))).unsqueeze(0)).to(device)
	class_name = ['Thamnophis Proximus', 'Nerodia Sipedon', 'Opheodrys Vernalis', 'Crotalus Horridus', 'Crotalus Pyrrhus', 'Nerodia Rhombifer', 'Thamnophis Sirtalis', 'Natrix Natrix', 'Crotalus Adamanteus', 'Charina Bottae', 'Pituophis Catenifer', 'Lampropeltis Triangulum', 'Nerodia Erythrogaster', 'Thamnophis Marcianus', 'Lampropeltis Californiae', 'Crotalus Ruber', 'Rhinocheilus Lecontei', 'Opheodrys Aestivus', 'Thamnophis Ordinoides', 'Thamnophis Radix', 'Masticophis Flagellum', 'Pantherophis Vulpinus', 'Hierophis Viridiflavus', 'Feterodon Platirhinos', 'Pantherophis Emoryi', 'Regina Septemvittata', 'Haldea Striatula', 'Diadophis Punctatus', 'Nerodia Fasciata', 'Storeria Occipitomaculata', 'Crotalus Scutulatus', 'Storeria Dekayi', 'Crotalus Viridis', 'Boa Imperator', 'Pantherophis Obsoletus', 'Lichanura Trivirgata', 'Agkistrodon Contortrix', 'Thamnophis Elegans', 'Agkistrodon Piscivorus', 'Pantherophis Guttatus', 'Crotalus Atrox', 'Carphophism Amoenus', 'Coluber Constrictor', 'Pantherophis Spiloides', 'Pantherophis Alleghaniensis']
	model_ft.eval()
	output = model_ft(item)
	_, preds = torch.max(output, 1)
	# print(preds)
	class_id = preds.item()
	# print(class_id)
	# print("Predicted class: ", class_name[class_id])
	return class_name[class_id]