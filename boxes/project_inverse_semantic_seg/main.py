
'''
from https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit2/02_class_conditioned_diffusion_model_example.ipynb#scrollTo=Y8XOX0OPmQeB
 
'''

# %%%%%%%%%%%%%%%%

import importlib
from pathlib import Path


import torch
import torchvision
import torchvision.transforms.functional as TF
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import matplotlib.colors as mcolors
import numpy as np
import customUnetModel

from tqdm.auto import tqdm

# local
import dataset
importlib.reload(dataset)
importlib.reload(customUnetModel)


# %%%%%%%%%%%%%%%%%%%%%
# Set up params and device
BATCH_SIZE = 64
# IMAGE_SIZE = (128, 128)---
COCO_PATH = '../../COCO'
list_natural_supcats = ['person','animal','food','outdoor']

N_EPOCHS = 10

map_supercat_str2ids = {k:v+1 for v,k in enumerate(list_natural_supcats)}
print(map_supercat_str2ids)


device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

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
# get categories for given supercategory names (unique?)
sel_cat_ids = all_annotations_per_split['train'].getCatIds(supNms=list_natural_supcats) # get categories for given supercategory names: we select a subset of 26 categories
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%###########
# Create datasets
train_data = dataset.customCOCO2017(
    True,
    all_annotations_per_split,  
    sel_img_ids_per_split, 
    sel_cat_ids, 
    img_path_per_split, 
    dataset.train_transform,
    map_supercat_str2ids)

val_data = dataset.customCOCO2017(
    False,
    all_annotations_per_split, 
    sel_img_ids_per_split, 
    sel_cat_ids, 
    img_path_per_split, 
    dataset.train_transform,
    map_supercat_str2ids)

# Create dataloaders
train_dataloader = DataLoader(
    train_data,
    BATCH_SIZE, 
    shuffle=True, 
    drop_last=True, 
    pin_memory=True, # not sure what this does...copy to device pin-memory before returning?
)

valid_dataloader = DataLoader(
    val_data,
    BATCH_SIZE, 
    shuffle=False, 
    drop_last=False, 
    pin_memory=True,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualize dataset
# mask is a 3D tensor w/ semantic labels along 'channnel' dimension
# img, mask = train_data[12] # 10
img, mask = next(iter(train_dataloader)) # get a batch
img, mask = img[0], mask[0]
plt.figure()
plt.imshow(TF.to_pil_image(img))
plt.show()
for t in range(mask.shape[0]):
    # if np.max(mask[t,:,:].numpy()) > 0:
        plt.imshow(mask[t,:,:])
        if t==0:
            plt.title('background')
        else:
            plt.title(list(train_data.map_supercat_ids.keys())[list(train_data.map_supercat_ids.values()).index(t)])
        plt.colorbar()
        plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Instantiate model

model = customUnetModel.ClassConditionedUnet().to(device)


# %%%%%%%%%%%%%%%%%%%%%%%
# Prepare for training

# scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')

# loss fn
loss_fn = nn.MSELoss()

# optimizer
opt = torch.optim.Adam(model.parameters(), lr=1e-3) 

# %%%%%%%%%%%%%%%%%%%%%%
# Training loop
# ideally: plot as we go? plot on test set too?

# Keeping a record of the losses for later viewing
losses = []

# The training loop
for epoch in range(N_EPOCHS):
    for x, y_masks in tqdm(train_dataloader):
        
        # Get some data and prepare the corrupted version
        x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
        y_masks = y_masks.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction for the noise
        pred = model(noisy_x, timesteps, y_masks) # Note that we pass in the labels y

        # Calculate the loss comparing to the actual noise
        loss = loss_fn(pred, noise) # How close is the output to the noise

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

    # Print our the average of the last 100 loss values to get an idea of progress:
    avg_loss = sum(losses[-100:])/100
    print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')

# %%
# View the loss curve
plt.plot(losses)
plt.show()

# %%%%%%%%%%%%%%%%%%
# Sampling / inference