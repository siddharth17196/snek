import re
import os
import sys
import cv2
import time
import json
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

"""
Class def
"""

SZ = 360
class CustomDataset(Dataset):
    def __init__(self, j, aug=None):
        self.j = j
        if aug is not None: aug = get_aug(aug)
        self.aug = aug
    
    def __getitem__(self, idx):
        item = j2anno(self.j[idx])
        if self.aug: item = self.aug(**item)
        im, bbox = item['image'], np.array(item['bboxes'][0])
        im, bbox = self.normalize_im(im), self.normalize_bbox(bbox)
        
        return im.transpose(2,0,1).astype(np.float32), bbox.astype(np.float32)
    
    def __len__(self):
        return len(self.j)
    
    def normalize_im(self, ary):
        return ((ary / 255 - imagenet_stats[0]) / imagenet_stats[1])
    
    def normalize_bbox(self, bbox):
        return bbox / SZ

class SnakeDetector(nn.Module):
    def __init__(self, arch=models.resnet18):
        super().__init__() 
        self.cnn = create_body(arch)
        self.head = create_head(num_features_model(self.cnn) * 2, 4)
        
    def forward(self, im):
        x = self.cnn(im)
        x = self.head(x)
        return x.sigmoid_()

"""
Load dataset
"""

parser = argparse.ArgumentParser(description='PyTorchLab')
parser.add_argument('-m', '--modeldir', required=True)
parser.add_argument('-d', '--datadir', required=True)
parser.add_argument('-o', '--outdir', required=True)

args = parser.parse_args()

MODEL_PATH = args.modeldir
DATASET_PATH = args.datadir
OUTPUT_PATH = args.outdir

if OUTPUT_PATH[-1] != "/":
	OUTPUT_PATH += "/"

src = (ImageList.from_folder(path=DATASET_PATH).split_by_rand_pct(0.0).label_from_folder())

tfms = get_transforms(do_flip=True,flip_vert=False,max_rotate=10.0,max_zoom=1.1,max_lighting=0.2,max_warp=0.2,p_affine=0.75,p_lighting=0.75)

data = (src.transform(tfms, size=360, resize_method=ResizeMethod.SQUISH)
        .databunch(bs=32).normalize(imagenet_stats))

"""
Load Model
"""

learn = Learner(data, SnakeDetector(arch=models.resnet50), loss_func=torch.nn.L1Loss())

learn.split([learn.model.cnn[:6], learn.model.cnn[6:], learn.model.head])

if torch.cuda.is_available():
    state_dict = torch.load(MODEL_PATH)
else:
    state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))

learn.model.load_state_dict(state_dict['model'])

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)

src_new = (ImageList.from_folder(path=DATASET_PATH).split_by_rand_pct(0.0).label_from_folder())
str_name = str(src_new.items[0])

for filename in progressbar.progressbar(src_new.items):
    try:
        start = time.time()
        
        im = cv2.imread(f"{filename}", cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (360,360), interpolation = cv2.INTER_AREA)
        im_height, im_width, _ = im.shape
        
        orig_im = cv2.imread(f"{filename}", cv2.IMREAD_COLOR)
        orig_im_height, orig_im_width, _ = orig_im.shape
        to_pred = open_image(filename)
        
        _,_,bbox=learn.predict(to_pred)
        
        im_original = cv2.imread(f"{filename}", cv2.IMREAD_COLOR)
        im_original = cv2.cvtColor(im_original, cv2.COLOR_BGR2RGB)
        im_original.shape
        im_original_width = im_original.shape[1]
        im_original_height = im_original.shape[0]
        
        bbox_new = bbox
        bbox_new[0] = bbox_new[0]*im_original_width 
        bbox_new[2]= bbox_new[2]*im_original_width
        bbox_new[1] = bbox_new[1]*im_original_height
        bbox_new[3] = bbox_new[3]*im_original_height
        x_min, y_min, x_max, y_max = map(int, bbox_new)
        
        im_original = im_original[y_min:y_max,x_min:x_max]
        im_original = cv2.cvtColor(im_original,cv2.COLOR_BGR2RGB)
        filename_str = str(filename)
        
        to_save = filename_str.replace('train','cropped_images')
        to_save = to_save.split("/")
        file_name = "/".join(to_save[len(to_save)-2:])
        class_id = OUTPUT_PATH + to_save[-2]
        
        if not os.path.exists(class_id):
            os.makedirs(class_id)
        
        to_save = OUTPUT_PATH + file_name
        cv2.imwrite(to_save, im_original)
        # print("saved", to_save)
        # print('It took', time.time()-start, 'seconds.')
    except Exception as e:
        print(str(e))
