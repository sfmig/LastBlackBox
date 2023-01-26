
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
from torch.nn import functional as F

import numpy as np
# import skimage.io as io

import random



class customCOCO2017(torch.utils.data.Dataset): 
    def __init__(
        self, 
        train: bool,
        annotations_per_split: Dict, #COCO, 
        img_ids_per_split: Dict, #Dict[List[int]], #List[int], 
        cat_ids: List[int], 
        img_path_per_split: Dict, 
        transform: Optional[Callable],
        map_supercat_ids
    ) -> None:
        super().__init__()

        key = 'train' if train else 'val'
        self.annotations = annotations_per_split[key] # annotations
        self.img_data = self.annotations.loadImgs(img_ids_per_split[key]) # image data
        self.files = [str(img_path_per_split[key] / img["file_name"]) for img in self.img_data] # file per image

        self.cat_ids = cat_ids # categories per image
        self.transform = transform # transform to apply
        self.map_supercat_ids = map_supercat_ids # map supercategories to integers
        
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
        # compute corresponding supercategory for each annotation (id) in this image
        supercats_per_cat_in_anns = {}
        for annot_cat_id in list(set([ann['category_id'] for ann in anns])): #[ann['category_id'] for ann in anns]:
            supercats_per_cat_in_anns[annot_cat_id] = self.annotations.loadCats(annot_cat_id)[0]['supercategory']

        #----
        # cat_details_per_annotation = self.annotations.loadCats([ann['category_id'] for ann in anns])
        # supercats_per_ann = [
        #     cat['supercategory'] for cat in self.annotations.loadCats([ann['category_id'] for ann in anns])
        # ] # TODO: probs a better way to do this....

        #---------------------

        # annToMask: converts polygons to binary mask--> multiply by categoryID
        # ---> stack binary masks along 'batch' dimension? ---> take max along max dimension (to combine all masks and remove 0s)
        # ---> add extra dimension with unsqueeze
        ### for semantic mask based on supercategories
        mask = torch.LongTensor(np.max(np.stack([self.annotations.annToMask(ann) * self.map_supercat_ids[supercats_per_cat_in_anns[ann["category_id"]]] 
                                                 for ann in anns]), axis=0)).unsqueeze(0) # shape: BS, 500, 381
        mask = torch.moveaxis(
            F.one_hot(
                mask.squeeze(),
                len(set([cat['supercategory'] for cat in self.annotations.loadCats(self.cat_ids)])) + 1 #one extra category for background?
            ) ,
            -1,
            0
        )
        
        # discard background?

        ### for semantic mask based on categories:
        # mask = torch.LongTensor(np.max(np.stack([self.annotations.annToMask(ann) * ann["category_id"] 
        #                                          for ann in anns]), axis=0)).unsqueeze(0) # shape: BS, 500, 381
        # mask = torch.moveaxis(
        #     F.one_hot(mask.squeeze(),len(self.cat_ids)) ,
        #     -1,
        #     0
        # )
        
        
        ## If image is B/W?
        img = io.read_image(self.files[i]) # shape: torch.Size([3, 500, 381])
        if img.shape[0] == 1: # if b/w, concatenate 3 copies along channels dim?
            img = torch.cat([img]*3)
        
        ## Apply transform to image and mask
        if self.transform is not None:
            return self.transform(img, mask)
        
        return img, mask


def train_transform(
    img1: torch.LongTensor, # img
    img2: torch.LongTensor, # mask
    IMAGE_SIZE=(64,64)
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    '''
    Apply identical transform (random crop and horiz flip) to two images

    Note from author:
    When using augmentations we need to be careful to apply the same transformation to image and the mask. So for example when doing a random crop as below, we need to make it somewhat deterministic. The way to do that in torch is by getting the transformation parameters 
    and then using torchvision.transforms.functional which are deterministic transformations.
    '''
    # Random crop
    params = transforms.RandomResizedCrop.get_params(img1, scale=(0.5, 1.0), ratio=(0.75, 1.33))
    img1 = TF.resized_crop(img1, *params, size=IMAGE_SIZE)
    img2 = TF.resized_crop(img2, *params, size=IMAGE_SIZE)
    
    # Random horizontal flipping
    if random.random() > 0.5:
        img1 = TF.hflip(img1)
        img2 = TF.hflip(img2)
        
    return img1, img2