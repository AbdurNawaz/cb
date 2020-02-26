import numpy as np 
import pandas as pd

import os
print(os.listdir("./RiceDiseaseDataset"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler

data_dir = './RiceDiseaseDataset'
data_dir1 = './rice_blast'


from glob import glob
images = glob(os.path.join(data_dir, '*/*.jpg'))
tot_images = len(images)
print('Total images:', tot_images)


im_cnt = []
class_names = []
print('{:18s}'.format('Class'), end='')
print('Count')
print('-' * 24)
for folder in os.listdir(os.path.join(data_dir)):
    folder_num = len(os.listdir(os.path.join(data_dir, folder)))
    im_cnt.append(folder_num)
    class_names.append(folder)
    print('{:20s}'.format(folder), end=' ')
    print(folder_num)
    if (folder_num < tot_images):
        tot_images = folder_num
        folder_num = folder
        
num_classes = len(class_names)
print('Total number of classes: {}'.format(num_classes))

data_transforms ={
    "train_transforms": transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224), 
                                           transforms.RandomHorizontalFlip(), 
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),
   "valid_transforms": transforms.Compose([transforms.Resize(225),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]), 
    "test_transforms": transforms.Compose([transforms.Resize(225),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
}

classes=['leafblast', 'healthy']

model_transfer = models.googlenet(pretrained=True)

use_cuda = torch.cuda.is_available()
if use_cuda:
    model_transfer = model_transfer.cuda()


for param in model_transfer.parameters():
    param.requires_grad=True

n_inputs = model_transfer.fc.in_features

last_layer = nn.Linear(n_inputs, len(classes))

model_transfer.fc = last_layer

if use_cuda:
    model_transfer = model_transfer.cuda()

model_transfer.load_state_dict(torch.load('model_transfer.pt'))

model_transfer.eval()

import cv2
from PIL import Image

for i in range(100):
    img = cv2.imread('./RiceDiseaseDataset/healthy/'+str(i)+'.jpg')
    img = Image.fromarray(img)
    # img = torch.from_numpy(img).long()
    img_tensor = data_transforms["test_transforms"](img)
    img_tensor = img_tensor.cuda()
    img_tensor.unsqueeze_(0)
    output = model_transfer(img_tensor)
    pred = output.data.max(1, keepdim=True)[1]
    print(pred, i)
