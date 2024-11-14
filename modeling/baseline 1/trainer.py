import os
import time
import pickle
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as T
from torch.utils.data import random_split, DataLoader
from model import Group_Activity_Classifer
# from torch.utils.tensorboard import SummaryWriter

import sys
import os
root = os.path.abspath("/teamspace/studios/this_studio/Group-Activity-Recognition") 
sys.path.append(root)

from data_utils import Group_Activity_DataSet
from utils import load_config



def train_one_epoch():
    ...


def validate_model():
    # model.eval()
    ...


def train_model():
    ...

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    print(Group_Activity_Classifer)
    print(Group_Activity_DataSet)
   
   