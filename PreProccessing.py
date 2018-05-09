from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import scipy.misc as misc
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os



BlueMean=103.939
GreenMean=116.779
RedMean=123.68
######################################################################################################################################
def LabelConvert(label,NumClasses):
    label = torch.from_numpy(label).cuda()
    # create one-hot encoding
    batchsize, h, w= label.size()
    target = torch.zeros(batchsize,NumClasses, h, w).cuda()
    for c in range(NumClasses):
       for b in range(batchsize):
         target[b][c][label[b] == c] = 1
    return torch.autograd.Variable(target,requires_grad=False)
#####################################################################################################################################3
def ImageConvert(rgb):
    rgb = torch.from_numpy(rgb).cuda()
    rgb =rgb.transpose(1,3).transpose(2,3)
    batchsize, d,  h, w = rgb.size()
    [r,g,b]=torch.split(rgb, 1, dim=1)
    bgr=torch.cat((b-BlueMean, g-GreenMean,r-RedMean),dim=1)
    return torch.autograd.Variable(bgr,requires_grad=False)