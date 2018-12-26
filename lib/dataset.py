import torch.utils.data as Data

from skimage import io
from tqdm import tqdm
from glob import glob
import numpy as np
import imageio
import random
import cv2
import os

"""
    This script defines the implementation of the data loader
    Notice: you should following the format below:
        root --+------ IMAGE_1_FOLDER --+-----under_exposure_image1
               |                        |
               |                        +-----under_exposure_image2
               |                        |
               |                        +-----over_exposure_image1
               |                        |
               |                        +-----over_exposure_image2
               |
               +------ IMAGE_2_FOLDER --+-----under_exposure_image1
               |                        |
               |                        +-----under_exposure_image2
               |                        |
               |                        +-----over_exposure_image1
               |                        |
               |                        +-----over_exposure_image2
               ...                      
    In the root folder, each image use a sub-folder to represent
    In each sub-folder, there are several under exposure images and over exposure images
    The program will random select one under and over image to crop and return

    Author: SunnerLi
"""

class BracketedDataset(Data.Dataset):
    def __init__(self, root, crop_size = 64, transform = None):
        self.files = glob(os.path.join(root, '*/'))
        self.crop_size = crop_size
        self.transform = transform
        self.under_exposure_imgs = []
        self.over_exposure_imgs  = []
        self.statistic()

    def statistic(self):
        bar = tqdm(self.files)
        for folder_name in bar:
            bar.set_description(" Statistic the over-exposure and under-exposure image list...")
            # Get the mean
            mean_list = []
            imgs_list = glob(os.path.join(folder_name, '*'))
            for img_name in imgs_list:
                img = cv2.imread(img_name)
                mean = np.mean(img)
                mean_list.append(mean)
            mean = np.mean(mean_list)

            # Split the image name
            under_list = []
            over_list  = []
            for i, m in enumerate(mean_list):
                img = cv2.imread(imgs_list[i])
                img = cv2.resize(img, (1200, 800))
                if m > mean:
                    over_list.append(img)
                else:
                    under_list.append(img)
            assert len(under_list) > 0 and len(over_list) > 0

            # Store the result
            self.under_exposure_imgs.append(under_list)
            self.over_exposure_imgs.append(over_list)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Random select
        under_img = self.under_exposure_imgs[index][random.randint(0, len(self.under_exposure_imgs[index]) - 1)]
        over_img  = self.over_exposure_imgs[index][random.randint(0, len(self.over_exposure_imgs[index]) - 1)]
        under_img = cv2.cvtColor(under_img, cv2.COLOR_BGR2YCrCb)
        over_img = cv2.cvtColor(over_img, cv2.COLOR_BGR2YCrCb)

        # Transform
        if self.transform:
            under_img = self.transform(under_img)
            over_img  = self.transform(over_img)

        # Crop the patch
        _, h, w = under_img.shape
        y = random.randint(0, h - self.crop_size)
        x = random.randint(0, w - self.crop_size)
        under_patch = under_img[:, y:y + self.crop_size, x:x + self.crop_size]
        over_patch  = over_img [:, y:y + self.crop_size, x:x + self.crop_size]
        return under_patch, over_patch