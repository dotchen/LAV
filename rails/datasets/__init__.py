import tqdm
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, WeightedRandomSampler
from .ego_dataset import EgoDataset
from .main_dataset import LabeledMainDataset

def data_loader(data_type, config):

    if data_type == 'ego':
        dataset = EgoDataset(config.data_dir, T=config.ego_traj_len)
    elif data_type == 'main':
        dataset = LabeledMainDataset(config.data_dir, config.config_path)
    else:
        raise NotImplementedError(f'Unknown data type {data_type}')

    
    return DataLoader(dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True, drop_last=True)

__all__ = ['data_loader']