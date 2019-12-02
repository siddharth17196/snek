import re
import os
import sys
import cv2
import time
import json
import pickle
import string
import random
import argparse
import progressbar
import numpy as np
import pandas as pd

from fastai.vision import *
from fastai.basic_data import *
from fastai.metrics import accuracy
from fastai.callbacks.hooks import num_features_model

import torch
import torchvision
from efficientnet_pytorch import EfficientNet

parser = argparse.ArgumentParser(description='PyTorchLab')
parser.add_argument('-m', '--modeldir', required=True)
parser.add_argument('-t', '--train', required=True)
parser.add_argument('-v', '--valid', required=True)
parser.add_argument('-b', '--batchsize', required=False, default=8)
parser.add_argument('-mn', '--modelname', required=False, default="b5")

args = parser.parse_args()

MODEL_PATH = args.modeldir
TRAIN_PATH = args.train
VALID_PATH = args.valid
BATCH_SIZE = int(args.batchsize)
MODEL_NAME = "efficientnet-" + args.modelname

src = (ImageList.from_folder(path=TRAIN_PATH).split_by_rand_pct(0.2).label_from_folder())
transforms = ([rotate(degrees=(-90,90), p=0.8)],[])
image_size = EfficientNet.get_image_size(MODEL_NAME)
data = (src.transform(transforms, size=image_size, resize_method=ResizeMethod.SQUISH).databunch(bs=BATCH_SIZE).normalize(imagenet_stats))

data.show_batch(3,figsize=(9,9))

"""
Load Model
"""
if torch.cuda.is_available():
    state_dict = torch.load(MODEL_PATH)
else:
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

model = EfficientNet.from_pretrained(MODEL_NAME)
model.add_module('_fc',nn.Linear(2048, data.c))

loss_func =LabelSmoothingCrossEntropy()
RMSprop = partial(torch.optim.RMSprop)
learn = Learner(data, model, loss_func=loss_func, opt_func=RMSprop, metrics=[accuracy, FBeta(beta=1, average='macro')])
learn.model.load_state_dict(state_dict['model'])

src_new = (ImageList.from_folder(path=TRAIN_PATH).split_by_rand_pct(0.0).label_from_folder())
str_name = str(src_new.items[0])

def load_train_dataset(root, batchsize):
    data_path = root
    trainTransform  = torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize([0.0432, 0.0554, 0.0264], [0.8338, 0.8123, 0.7803]),
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
    modeldir = root + "/metrics"
    checkandcreatedir(modeldir)
    filepath = modeldir + "/" + name
    savePickle(filepath, model)

def loadmodel(filename):
    root = str(Path(__file__).parent.parent)
    modeldir = root + "/metrics"
    filename = modeldir + "/" + filename
    try:
        model = loadPickle(filename)
        return model
    except:
        raise Exception("Model not found: " + filename )

image_datasets = {}
dataloaders_dict = {}
dataloaders_dict['train'], image_datasets['train'] = load_train_dataset(TRAIN_PATH, BATCH_SIZE)
dataloaders_dict['valid'], image_datasets['valid'] = load_train_dataset(VALID_PATH, BATCH_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_train = DeviceDataLoader(dataloaders_dict["train"], device)
data_train = DeviceDataLoader(dataloaders_dict["valid"], device)

metrics = learn.validate(data)
storemodel(metrics, "train_metrics")

metrics = learn.validate(dataloaders_dict["valid"])
storemodel(metrics, "valid_metrics")
