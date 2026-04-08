import json
import torch
import random
import os
import numpy as np
from scipy.io import loadmat

from .pyExt import Dict2Obj

def getDatasetInfo(dataset):
    with open("datasets/dataset_config.json", "r") as f:
        info = json.load(f)[dataset]

    return Dict2Obj(info)

def getDataByInfo(info):
    dataset_path = os.path.join('./datasets', info.path)

    if info.type is None:
        data = loadmat(os.path.join(dataset_path, info.file_name))[info.mat_name].astype(np.float32)
    elif info.type == 'npy':
        data = np.load(os.path.join(dataset_path, info.file_name)).astype(np.float32)

    return data

def getGTByInfo(info):
    dataset_path = os.path.join('./datasets', info.path)

    if info.type is None:
        gt = loadmat(os.path.join(dataset_path, info.gt_file_name))[info.gt_mat_name].astype(np.int64)
    elif info.type == 'npy':
        gt = np.load(os.path.join(dataset_path, info.gt_file_name)).astype(np.int64)
    
    return gt

def seed_torch(seed):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def getDevice(device=None):
    if device is None:
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    elif device == -1:
        return torch.device('cpu')
    else:
        return torch.device(f'cuda:{device}')
