
'''
from https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb#scrollTo=Y8XOX0OPmQeB
 
'''

# %%

import torch
import torchvision

from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
# from tqdm.auto import tqdm

from pathlib import Path

from pycocotools.coco import COCO

# local
import dataset
# import customUnetModel

import importlib
importlib.reload(dataset)
# importlib.reload(customUnetModel)

# %%%%%%%%%%%%%%%%%%%
# Set up paths


# %%%%%%%%%%%%%%%%%%%%%
# Set up params and device
BATCH_SIZE = 64
IMAGE_SIZE = (128, 128)
COCO_PATH = '../../COCO'
list_natural_supcats = ['person','outdoor','animal','food']

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# %%%%%%%%%%%%%%%%%%%%
# Check donwloaded COCO dataset
# https://sachinruk.github.io/blog/pytorch/data/2021/08/21/coco-semantic-segmentation-data.html

# ## Get annotations
train_annotations = COCO(Path(COCO_PATH) / "annotations/instances_train2017.json")
valid_annotations = COCO(Path(COCO_PATH) / "annotations/instances_val2017.json")

## Get image IDs for train and valid set
# select a subset of superCategories
# 11 supercategories: person, vehicle, accessory, outdoor, animal, sports, kitchenware, food, furniture, electronic, appliance, kitchen, indoor
list_natural_supcats = ['person','outdoor','animal','food']
cat_ids = train_annotations.getCatIds(supNms=list_natural_supcats) # get categories for given supercategory names
train_img_ids = []
for cat in cat_ids:
    train_img_ids.extend(train_annotations.getImgIds(catIds=cat)) # extend can add multiple, append only one
    
train_img_ids = list(set(train_img_ids))
print(f"Number of training images: {len(train_img_ids)}") # 94291

valid_img_ids = []
for cat in cat_ids:
    valid_img_ids.extend(valid_annotations.getImgIds(catIds=cat))
    
valid_img_ids = list(set(valid_img_ids))
print(f"Number of validation images: {len(valid_img_ids)}") # 4000

print(f"Train-val split: {len(train_img_ids)/(len(train_img_ids)+len(valid_img_ids)):.2f}/{len(valid_img_ids)/(len(train_img_ids)+len(valid_img_ids)):.2f}")

# %%%%%%%%%%%%%%%%%%%%
#  Prepare dataset for segmentation
# https://sachinruk.github.io/blog/pytorch/data/2021/08/21/coco-semantic-segmentation-data.html

## subdir of images per split
img_path_per_split = {
    'train': Path(COCO_PATH)/'train2017',
    'val': Path(COCO_PATH)/'val017',
}

## all annotations per split
all_annotations_per_split = {
    'train': COCO(Path(COCO_PATH) / "annotations/instances_train2017.json"),
    'val': COCO(Path(COCO_PATH) / "annotations/instances_val2017.json")
}

## Select a subset of categories-IDs and extract their corresponding image-IDs
# get categories for given supercategory names
sel_cat_ids = all_annotations_per_split['train'].getCatIds(supNms=list_natural_supcats) # get categories for given supercategory names
# get image ids for those categories only
sel_img_ids_per_split = {}
for split in ['train','val']:
    split_img_ids = []
    for cat in sel_cat_ids:
        split_img_ids.extend(all_annotations_per_split[split].getImgIds(catIds=cat)) # extend can add multiple, append only one
    sel_img_ids_per_split[split] = list(set(split_img_ids)) # OJO! remove duplicates w set!

n_train_split = len(sel_img_ids_per_split['train'])
n_val_split = len(sel_img_ids_per_split['val'])
print(f"Number of training images: {n_train_split}") # 94291
print(f"Number of validation images: {n_val_split}") # 4000
print(f"Train-val split: {n_train_split/(n_train_split+n_val_split):.2f}/{n_val_split/(n_train_split+n_val_split):.2f}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create data loaders
train_data = dataset.customCOCO2017(
    True,
    all_annotations_per_split,  
    sel_img_ids_per_split, 
    sel_cat_ids, 
    img_path_per_split, 
    dataset.train_transform)

val_data = dataset.customCOCO2017(
    False,
    all_annotations_per_split, 
    sel_img_ids_per_split, 
    sel_cat_ids, 
    img_path_per_split, 
    dataset.train_transform)



# train_data = dataset.customCOCO2017(
#     train_annotations,  #---
#     train_img_ids, #-----
#     cat_ids, 
#     Path(COCO_PATH)  / "train2017", #----
#     dataset.train_transform)

# valid_data = dataset.customCOCO2017(
#     valid_annotations, 
#     valid_img_ids, 
#     cat_ids, 
#     Path(COCO_PATH)   / "val2017", 
#     dataset.train_transform)

# define dataloaders

# %%
# Feed it into a dataloader (batch size 8 here just for demo)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# View some examples
x, y = next(iter(train_dataloader))
print('Input shape:', x.shape)
print('Labels:', y)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Instantiate model



# %%%%%%%%%%%%%%%%%%%%%%%
# Prepare for training

# scheduler

# loss fn

# optimizer


# %%%%%%%%%%%%%%%%%%%%%%
# Training loop
# ideally: plot as we go? plot on test set too?


# %%%%%%%%%%%%%%%%%%
# Sampling / inference