import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import time
import numpy as np
import matplotlib.pyplot as plt
import models
import config
import os
from PIL import Image

valid_directory = config.VALID_DATASET_DIR

batch_size = 1
num_classes = config.NUM_CLASSES
lr=config.LR
resume=config.PRETRAINED_MODEL

test_valid_transforms = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406],
         [0.229, 0.224, 0.225])])


test_datasets = datasets.ImageFolder(valid_directory,transform=test_valid_transforms)
test_data_size = len(test_datasets)
test_data = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

if __name__=='__main__':
    model=torch.load(r'C:\LongFor\work2020\classified_pytorch\e1\MachineLearning\lab1\trained_models\vehicle-12_record.pth')
    print(model)
    image_path=r'C:\LongFor\work2020\classified_pytorch\e1\MachineLearning\lab1\vehicle-10\train\heavy truck\2c6aab06eb4266f2cf20ce145bfc8612.jpg'
    test_image=Image.open(image_path)
    test_image_tensor=test_valid_transforms(test_image).unsqueeze(0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():

        model=model.to(device)

        model.eval()
        out=model(test_image_tensor)
        ps=torch.exp(out)
        topk,topclass=ps.topk(1,dim=1)
        print(topk,topclass)