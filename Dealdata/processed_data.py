import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
import random 
import shutil
# from .builder import DATASETS
# from .defaults import DefaultDataset
# from .transform import Compose, TRANSFORMS

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc,m,centroid



# 预测的结果是归一化的，需要转换回去
data_root= './temp_data/points'
pred_root = './Pointcept/exp/teeth_land/semseg-pt-v3m1-0-base/result'
save_root = './Pointcept/exp/teeth_land/semseg-pt-v3m1-0-base/result_test_norm_new'

num = 0
os.makedirs(save_root, exist_ok=True)
for file_name in os.listdir(pred_root): 
    data_path = os.path.join(data_root,file_name)
    if not os.path.exists(data_path):
        continue
    num += 1
    pc = np.loadtxt(data_path)
    pred = np.loadtxt(os.path.join(pred_root, file_name))
    _, m, c = pc_normalize(pc)
    pred[:, :3] = pred[:, :3] * m + c
    np.savetxt(os.path.join(save_root, file_name), pred)

