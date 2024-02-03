from operator import indexOf
import os
import cv2
import time
import json
import torch
import random
import requests
import numpy as np
import pandas as pd
from copy import deepcopy
from PIL import Image, ImageFile
from torch.utils.data.dataset import T
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2

def collate_fn(sample_list):
    obj_list, label_list =  [], []
    for sample in sample_list:
        obj_1, obj_2, label_1, label_2 = sample
        obj_list.extend([obj_1, obj_2])
        label_list.extend([label_1, label_2])
    label_list = torch.tensor(label_list, dtype=torch.int64)

    obj_list = torch.stack(obj_list, dim=0)
    return obj_list, label_list

class SFDataset(Dataset):
    def __init__(self, cfg):

        self.train_dir1 = os.path.join(cfg.train_dir, 'img4')
        self.train_dir2 = os.path.join(cfg.train_dir, 'img5')

        with open(cfg.train_file, 'r') as file:
            jpg_files = file.read().splitlines()

        self.img_list = jpg_files
        self.size = cfg.img_size
        self.random_choice = cfg.random
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),    # 水平翻转
            A.RandomResizedCrop(height=self.size, width=self.size,
            scale=(0.08, 1.0), ratio=(1. / 2., 2. / 1.), p=0.9),
            A.Resize(self.size, self.size), # 调整大小
            A.GaussNoise(p=0.5),    # 高斯噪声
            A.GaussianBlur(p=0.5),  # 高斯模糊
            A.ColorJitter(p=0.8),   # 颜色抖动
            A.ImageCompression(quality_lower=20, p=0.5),    # 图像压缩
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
            ToTensorV2(),  # 张量化
        ])
        
        self.source_transform = A.Compose([
            A.Resize(self.size,self.size),  # 调整大小
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
            ToTensorV2(),  # 张量化
        ])

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):

        img_name = self.img_list[idx]

        img_path1 = os.path.join(self.train_dir1, img_name)
        img_path2 = os.path.join(self.train_dir2, img_name)

        img1 = Image.open(img_path1)
        img2 = Image.open(img_path2)

        label1 = idx
        label2 = idx

        img1 = self.transform(image=np.asarray(img1))['image']
        img2 = self.transform(image=np.asarray(img2))['image']

        return img1, img2, label1, label2
        

if __name__ == '__main__':
    #train
    from afcl_config import cfg
    from pytorch_metric_learning import losses, distances, reducers, miners
    from model import ViT_B_16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    dataset = SFDataset(cfg.data)
   
    num_workers = 2

    model = ViT_B_16(embed_size=128)
    model = model.to(device)

    mining_func_img = miners.MultiSimilarityMiner(epsilon=0.1)
    loss_fn_img = losses.MultiSimilarityLoss(alpha=2, beta=50, base=0.5)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    for idx, batch in enumerate(dataloader):
        batch = tuple(t.to(device) for t in batch)
        input_img_batch, label_batch = batch[0], batch[1]

        print('img_batch:  ', input_img_batch.shape)
        print('label_batch:', label_batch.shape)

        output_embeddings = model(input_img_batch)
        print('output_embeddings: ', output_embeddings.shape)
    

    
