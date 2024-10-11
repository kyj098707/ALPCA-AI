from _model import EfficientV2

import pandas as pd
import numpy as np
import os
import argparse
from copy import deepcopy

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from tqdm.auto import tqdm
from experiment import experiment
import timm

import cv2

import random
import wandb
import warnings
warnings.filterwarnings(action='ignore') 


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = torch.FloatTensor(self.label_list[index])
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def get_classes():
    return pd.read_csv("./csv/webtoon_info.csv")["id"].unique()

def train(args):

    device = torch.device('cpu')
    early_stop = 0
    df = pd.read_csv('./comic.csv')
    train_transform = A.Compose([
                            A.Resize(args.img_size,args.img_size),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            ToTensorV2()
                        ])
    train_labels = df.iloc[:,2:].values
    train_dataset = CustomDataset(df['img_path'].values, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = EfficientV2()
    model.to(device)
    
    optimizer = optim.Adam(params = model.parameters(), lr = args.lr)
    best_similarity = 0
    best_model = None
    
    criterion = nn.BCELoss().to(device)

    for epoch in range(1, args.epochs+1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(imgs)
            loss = criterion(output, labels)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        similarity_list = experiment(model)
        _untrained_artworks_similarity = sum(similarity_list)/len(similarity_list)
        _train_loss = np.mean(train_loss)
        
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] untrained artwors : [{_untrained_artworks_similarity:.5f}]')
        
            
        if best_similarity < _untrained_artworks_similarity:
            best_similarity = _untrained_artworks_similarity
            best_model = deepcopy(model)
            early_stop = 0
            torch.save(best_model, f'./simx`_{epoch}.pth')
        else:
            early_stop += 1
        log_dic = {}
            
        log_dic["untrained similarity"] = _untrained_artworks_similarity
        
        if early_stop > 2:
            break

    torch.save(best_model, f'./simx`_{epoch}.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=777)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--img_size', type=int, default=384)
    parser.add_argument('--num_workers', type=int, default=1) 
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--model_name', default="ConvNext")
    parser.add_argument('--detail', default="xlarge_384")
    parser.add_argument('--makecsvfile', type=bool ,default=False)
    parser.add_argument('--ckpt', default=None)
    parser.add_argument('--clip', default=1)
    # parser.add_argument('--checkpoints', default="microsoft/beit-base-patch16-224-pt22k-ft22k")
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    train(args)