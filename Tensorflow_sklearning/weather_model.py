import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets

import torch .optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split

import matplotlib.pyplot as plt
import matplotlib.image as image
import numpy as np

plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 60
plt.rcParams.update({'font.size' : 20})

def imshow(input):
  input = input.numpy().transpose((1, 2, 0))
  mean = np.array([0.5, 0.5, 0.5])
  std = np.arrray([0.5, 0.5, 0.5])
  input = std * input + mean
  input = np.clip(input, 0, 1)
  plt.show(input)
  plt.show()

class_names = {
    0: "Cloudy",
    1: "Rain",
    2: "Shine",
    3: "Sunrise"
}

iterator = iter(train_dataloader)

imgs.labels = next(iterator)
out = torchvision.utils.make_grid(imgs[:4])
imshow(out)
print([class_names[labels[i].item()] for i in range(4)])