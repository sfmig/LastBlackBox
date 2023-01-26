
'''

from https://colab.research.google.com/github/sachinruk/blog/blob/master/_notebooks/2021-08-21-coco-semantic-segmentation-data.ipynb#scrollTo=Xg2fGyUfz5Fq
https://sachinruk.github.io/blog/pytorch/data/2021/08/21/coco-semantic-segmentation-data.html 
'''

from typing import Any, Callable, Dict, List, Optional, Tuple
from pycocotools.coco import COCO
from pathlib import Path


import torch
from torchvision import io, transforms
import torchvision.transforms.functional as TF

import numpy as np
import skimage.io as io

import random


class customCOCO2017(torch.utils.data.Dataset): 
    def __init__(
        self, 
        train: bool,
        annotations_per_split: Dict, #COCO, 
        img_ids_per_split: Dict, #Dict[List[int]], #List[int], 
        cat_ids: List[int], 
        img_path_per_split: Dict, 
        transform: Optional[Callable]=None
    ) -> None:
        super().__init__()

        if train:
            self.annotations = annotations_per_split['train'] # annotations
            self.img_data = self.annotations.loadImgs(img_ids_per_split['train']) # image data
            self.files = [str(img_path_per_split['train'] / img["file_name"]) for img in self.img_data] # file per image
        else:
            self.annotations = annotations_per_split['val'] # annotations
            self.img_data = self.annotations.loadImgs(annotations_per_split['val']) # image data
            self.files = [str(img_path_per_split['val'] / img["file_name"]) for img in self.img_data] # file per image
            
        self.cat_ids = cat_ids # categories per image
        self.transform = transform # transform to apply
        
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        ## Build semantic mask for sampled image
        ann_ids = self.annotations.getAnnIds( # get annotations IDs for the given filter conds
            imgIds=self.img_data[i]['id'], # for this image id
            catIds=self.cat_ids,  # for these categories
            iscrowd=None # filter out those with iscrowd=True?
        )
        anns = self.annotations.loadAnns(ann_ids) # load filtered annotations
        # annToMask: converts polygons to binary mask--> multiply by categoryID
        # ---> stack binary masks along 'batch' dimension? ---> take max along max dimension (to combine all masks and remove 0s)
        # ---> add extra dimension with unsqueeze
        mask = torch.LongTensor(np.max(np.stack([self.annotations.annToMask(ann) * ann["category_id"] 
                                                 for ann in anns]), axis=0)).unsqueeze(0)
        
        
        ## If image is B/W?
        img = io.read_image(self.files[i])
        if img.shape[0] == 1: # if b/w, concatenate 3 copies along channels dim?
            img = torch.cat([img]*3)
        
        ## Apply transform to image and mask
        if self.transform is not None:
            return self.transform(img, mask)
        
        return img, mask


def train_transform(
    img1: torch.LongTensor, 
    img2: torch.LongTensor,
    IMAGE_SIZE=(64,64)
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    '''
    Note from author:
    When using augmentations we need to be careful to apply the same transformation to image and the mask. So for example when doing a random crop as below, we need to make it somewhat deterministic. The way to do that in torch is by getting the transformation parameters 
    and then using torchvision.transforms.functional which are deterministic transformations.
    '''
    params = transforms.RandomResizedCrop.get_params(img1, scale=(0.5, 1.0), ratio=(0.75, 1.33))
    
    img1 = TF.resized_crop(img1, *params, size=IMAGE_SIZE)
    img2 = TF.resized_crop(img2, *params, size=IMAGE_SIZE)
    
    # Random horizontal flipping
    if random.random() > 0.5:
        img1 = TF.hflip(img1)
        img2 = TF.hflip(img2)
        
    return img1, img2