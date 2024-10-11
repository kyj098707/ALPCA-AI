import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
from glob import glob
from tqdm.auto import tqdm

from sentence_transformers import util
import timm

from _util import *
from _model import * 


def experiment(backbone):
    device = torch.device('cpu')
    folder_list = list(map(lambda x: x.split('\\')[-1], glob("./image/*")))
    test_transform = A.Compose([A.Resize(480, 480),
                                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                                    ToTensorV2()])

    similarity_list = []
    for folder in tqdm(folder_list[:20]):
        img = sorted(glob(f"./image/{folder}/*.jpg"))
        if len(img) < 10:
            continue
        test_dataset = CustomDataset(img, None, test_transform)
        test_loader = DataLoader(test_dataset, batch_size = 20, shuffle=False)
        preds = infer(backbone, test_loader, device)
        similarity = util.cos_sim(preds,preds)
        mean_similarity = similarity.numpy().mean()
        similarity_list.append(mean_similarity)

    return similarity_list
if __name__ == "__main__":
    similarity_list = experiment(BaseModel())
    
    print(similarity_list)
    print(sum(similarity_list)/len(similarity_list))


    # model = torch.load("./ckpt/comic_165label.pth")
    # trained_sims_list,untrained_sims_list = valid(model)
    # print(f"학습된 데이터에 대한 유사도 평균 : {sum(trained_sims_list)/len(trained_sims_list)}")
    # print(f"학습되지 않은 데이터에 대한 유사도 평균 : {sum(untrained_sims_list)/len(untrained_sims_list)}")