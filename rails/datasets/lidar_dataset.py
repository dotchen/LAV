import ray
import glob
import yaml
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

from .main_dataset import MainDataset

class LiDARMainDataset(MainDataset):
    def __getitem__(self, idx):
        
        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]

        locs = self.__class__.access('loc', lmdb_txn, index, self.T+1, dtype=np.float32)
        rots = self.__class__.access('rot', lmdb_txn, index, self.T, dtype=np.float32)
        spds = self.__class__.access('spd', lmdb_txn, index, self.T, dtype=np.float32).flatten()
        lbls = self.__class__.access('lbl', lmdb_txn, index+1, self.T, dtype=np.uint8).reshape(-1,96,96,12)

        lids = self.__class__.get_lidar(lmdb_txn, index)

        cmd = self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.float32).flatten()

        rgbs, sems = [], []
        for i, _ in enumerate(self.camera_yaws):
            rgbs.append(self.__class__.access('rgb_{}'.format(i), lmdb_txn, index, 1, dtype=np.uint8).reshape(320,320,3))
            sems.append(self.__class__.access('sem_{}'.format(i), lmdb_txn, index, 1, dtype=np.uint8).reshape(320,320))

        rgbs = np.stack(rgbs)[...,::-1]
        sems = np.stack(sems)

        return rgbs, sems, lids, lbls, locs, rots, spds, int(cmd)
    
    def __len__(self):
        return self.num_frames

    @staticmethod
    def get_lidar(lmdb_txn, index):

        curr_lids = LiDARMainDataset.access('lidar', lmdb_txn, index, 1, dtype=np.float32).reshape(-1,4)

        if index > 0:
            prev_lids = LiDARMainDataset.access('lidar', lmdb_txn, index-1, 1, dtype=np.float32).reshape(-1,4)
            lids = np.concatenate([prev_lids, curr_lids])
        else:
            lids = curr_lids

        return lids
