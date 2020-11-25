import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random

# --------------------------------------------------------------
# --------------------------------------------------------------
# TRANSFORMS
# --------------------------------------------------------------
# --------------------------------------------------------------
class RandomResizedCrop(object):
    def __init__(self, size=[256,256], scale=(0.5, 1.5), ratio=(0.75, 1.3333333333333333)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
        image = transforms.functional.resized_crop(image, i, j, h, w, self.size)
        mask = transforms.functional.resized_crop(mask, i, j, h, w, self.size, interpolation=Image.NEAREST)

        return {'image': image, 'mask': mask}

class RandomRotation(object):
    def __init__(self, degrees=[0,360]):
        self.degrees = degrees

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        degree = transforms.RandomRotation.get_params(degrees=self.degrees)
        image = Image.fromarray(transform.rotate(np.asarray(image).copy(), degree, order=1, mode='constant'))
        mask = Image.fromarray(transform.rotate(np.asarray(mask).copy(), degree, order=0, mode='constant'))

        return {'image': image, 'mask': mask}

class RandomFlip(object):

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        
        if random.random() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)

        if random.random() > 0.5:
            image = transforms.functional.vflip(image)
            mask = transforms.functional.vflip(mask)

        return {'image': image, 'mask': mask}
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self):
        self.apply = transforms.ToTensor()

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        return {'image': self.apply(image),
                'mask': self.apply(mask)}
                   
class ExtendImageChannel(object):
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        image = image.repeat([3,1,1])
        return {'image': image,
                'mask': mask}

# --------------------------------------------------------------
# --------------------------------------------------------------
# MAIN DATASET
# --------------------------------------------------------------
# --------------------------------------------------------------
class MicroscopyDataset(Dataset):
    
    def __init__(self, DATASET_DIR, DATASET_LIST, transform=None, with_mask=True):
    
        self.image_dir = os.path.join(DATASET_DIR, 'rawimages')
        self.mask_dir = os.path.join(DATASET_DIR, 'groundtruth')

        data_lines = open(os.path.join(DATASET_DIR, DATASET_LIST), 'r').readlines()
        self.imagelist = list(map(lambda x: os.path.join(self.image_dir, x.strip()), data_lines))
        self.with_mask = with_mask
        if self.with_mask:
            self.masklist = list(map(lambda x: os.path.join(self.mask_dir, x.strip()), data_lines))
            
        self.transform = transform

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.imagelist[idx]
        image = np.asarray(ImageOps.grayscale(Image.open(img_path))).astype(np.float) / 255
        image = Image.fromarray(image)
        if self.with_mask:
            mask_path = self.masklist[idx]
            mask = (np.asarray(ImageOps.grayscale(Image.open(mask_path))) != 0).astype(np.float)
            mask = Image.fromarray(mask)
            sample = {'image': image, 'mask': mask}
            if self.transform:
                sample = self.transform(sample)
        else:
            sample = {'image': image}

        return sample['image'], sample['mask'] if self.with_mask else None


# --------------------------------------------------------------
# --------------------------------------------------------------
# UTILITY
# --------------------------------------------------------------
# --------------------------------------------------------------
def show_sample(sample):
    image, mask = np.squeeze(sample['image'].numpy()), np.squeeze(sample['mask'].numpy())
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.show()

def vis_res(img_tensor, gt_tensor, prob_tensor, pred_tensor, save_path, title=None):
    img = img_tensor.cpu().data.numpy()
    gt = gt_tensor.cpu().data.numpy()
    prob = prob_tensor.cpu().data.numpy()
    pred = pred_tensor.cpu().data.numpy()

    plt.subplot(1,4,1)
    plt.imshow(img)
    plt.subplot(1,4,2)
    plt.imshow(gt)
    plt.subplot(1,4,3)
    plt.imshow(prob)
    plt.subplot(1,4,4)
    plt.imshow(pred)
    plt.adjust_subplot(left=0.1, right=0.9, top=0.9, bottom=0.1)
    if title:
        plt.subtitle(title)
    plt.savefig(save_path)
    plt.close()
