import argparse
import csv
import itertools
import os
import copy
from typing import Dict, List

import numpy as np
import cv2 as cv
import torch

from cv2 import imread, COLOR_BGR2RGB, cvtColor, IMREAD_COLOR, COLOR_BGR2GRAY
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from PIL import Image
from sklearn.model_selection import train_test_split

from tqdm import tqdm



def convert_tensor_to_visual_np_img(img:torch.Tensor):
    """
    :param img: C * H * W, 一般为 3 * 224 * 224
    :return: img_np，经过还原的图像，该图像格式为 224 * 224 * 3, uint8
    """
    assert len(img.shape) == 3, f"Invalid img with shape:{img.shape}"

    img_np = img.to("cpu").permute(1, 2, 0).numpy()
    img_np = img_np * 127.5 + 127.5
    img_np = (img_np / (img_np.max() + 1e-6)) * 255
    img_np = img_np.astype("uint8")
    img_np = cv.cvtColor(img_np, cv.COLOR_BGR2RGB)

    return img_np



def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",type=str, help="Which server running the code.")
    parser.add_argument("--model_idx", type=str, help="Timm model index")
    parser.add_argument("--key_word",type=str, help="Key word for saving model")
    parser.add_argument("--network",type=str)
    parser.add_argument("--aug_complexity",type=int)

    parser.add_argument('--num_classes', type=int, help='Background, IRF, SRF')

    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, help="Batch size of each epoch")
    parser.add_argument("--secondary_batch_size", type=int, help="unlabeled batch size")
    parser.add_argument("--num_workers", type=int, help="CPU workers for dataloader")
    parser.add_argument("--dataloader_prefetch_factor", type=int, help="Prefetch factor")
    parser.add_argument("--lr", type=float, help="Learning rate for optimization")
    parser.add_argument("--aug_prob", type=float, help="Probability of data augmentation")
    parser.add_argument("--seg_num", type=int, help="Background, IRF, SRF, and PED")
    parser.add_argument("--CUDA_VISIBLE_DEVICES", type=str, help="Set the GPU id for training")
    parser.add_argument("--width", type=int, default=224, help="width")
    parser.add_argument("--height", type=int, default=224, help="height")

    parser.add_argument("--S4RF_Fluid_dataset_path", type=str)
    parser.add_argument("--S4RF_Fluid_train_lab_ratio", type=float)
    parser.add_argument("--S4RF_Fluid_valid_ratio", type=float)
    parser.add_argument("--S4RF_Fluid_test_ratio", type=float)

    config = parser.parse_args()
    return config



def iterate_once(iterable):
    return np.random.permutation(iterable)



def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())



def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)




class S4RF_dataset_for_4S_DS(Dataset):
    def __init__(self, opt, mode:str="Train"):
        assert mode in ["Train", "Valid", "Test"], f"Unknown mode:{mode}"
        self.opt = opt
        self.mode = mode
        self.num_classes = opt.num_classes
        self.TVT_splits = None
        self.img_mask_path_pair = []
        self.train_lab_lens = 0
        self.transforms = self._get_transforms()[mode]

        if opt.isolated_patient:
            dataset_split = self._isolated_split_train_val_test(opt=opt)
        else:
            dataset_split = self._split_train_val_test(opt=opt)

        self.img_mask_path_pair.extend(dataset_split[mode])

        self.length = len(self.img_mask_path_pair)
        self.train_lab_lens = int(len(dataset_split["Train"]) * self.opt.S4RF_Fluid_train_lab_ratio)


  
    def convert_mask(self, mask):
        if self.num_classes == 3:
            new_mask = ((mask==63) * 1 + (mask==127) * 2 + (mask==255) * 3).astype(np.int32)
        elif self.num_classes == 5:
            new_mask = ((mask==63) * 1 + (mask==127) * 2 + (mask==191) * 3 + (mask==255) * 4).astype(np.int32)
        else:
            raise NotImplementedError(f"Not implement {self.num_classes}")
        return new_mask

  

    def __len__(self):
        return self.length


  
    def __getitem__(self, item):
        cv.ocl.setUseOpenCL(False)
        cv.setNumThreads(0)

        image = cvtColor(imread(self.img_mask_path_pair[item][0], IMREAD_COLOR), COLOR_BGR2RGB)
        if item >= self.train_lab_lens and self.mode == "Train": # Unlab Train
            mode_processed = self.transforms(image=image)
            img_p = mode_processed["image"]
            msk_p = torch.full(size=(self.opt.height,self.opt.width), fill_value=-1)

        else: # Lab Train, Valid, Test
            mask = self.convert_mask(mask=cvtColor(imread(self.img_mask_path_pair[item][1], IMREAD_COLOR), COLOR_BGR2GRAY))
            mode_processed = self.transforms(image=image, mask=mask)
            img_p = mode_processed["image"]
            msk_p = mode_processed["mask"]

        return img_p, msk_p




if __name__ == "__main__":
    opt = get_parser()

    dset = S4RF_dataset_for_4S_DS(opt=opt, mode="Train")

    batch_sampler = dset.get_TwoStreamBatchSampler()
  
    import random
    def worker_init_fn(worker_id):
        random.seed(opt.torch_seed + worker_id)
    trainloader = DataLoader(dset, batch_sampler=batch_sampler,
                             num_workers=opt.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

  
    for sample in trainloader:
        img, msk = sample
        print(img.shape, msk.shape)

  
    for i in tqdm(range(100)):
        img_p, msk_p = dset[i]
        img_p = convert_tensor_to_visual_np_img(img_p)
    
        msk  = torch.zeros((3,224,224))
        msk[0] = (msk_p == 1) * 255
        msk[1] = (msk_p == 2) * 255
        msk[2] = (msk_p == 3) * 255
    
        msk = msk.permute(1,2,0).numpy().astype("uint8")
    
        plt.figure(figsize=(4, 2))
        plt.subplot(1, 2, 1), plt.axis('off'), plt.title("Image")
        plt.imshow(img_p)
        plt.subplot(1, 2, 2), plt.axis('off'), plt.title("Mask")
        plt.imshow(msk)
    
        plt.tight_layout()
        plt.show()
        plt.close()
