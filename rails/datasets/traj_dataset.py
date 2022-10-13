import ray
import glob
import yaml
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from common.augmenter import augment


class MainDataset(Dataset):
    def __init__(self, data_dir, config_path):
        super().__init__()

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        