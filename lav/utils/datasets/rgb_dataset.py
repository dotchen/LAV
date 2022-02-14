import numpy as np
import cv2
import torch

from lav.utils.augmenter import augment
from lav.utils import filter_sem
from .basic_dataset import BasicDataset
from .lidar_dataset import transform_ego

class RGBDataset(BasicDataset):
    def __init__(self, config_path):
        super().__init__(config_path)

        self.augmenter = augment(0.5)

    def __getitem__(self, idx):

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]

        rgb1 = self.__class__.load_img(lmdb_txn, 'rgb_2', index)
        rgb2 = self.__class__.load_img(lmdb_txn, 'rgb_3', index)
        sem1 = self.__class__.load_img(lmdb_txn, 'sem_2', index)
        sem2 = self.__class__.load_img(lmdb_txn, 'sem_3', index)
        
        # BEV images
        bev = self.__class__.load_bev(lmdb_txn, index, channels=[0,1,2,6])
        bev = (bev>0).astype(np.uint8).transpose(2,0,1)

        rgb = np.concatenate([rgb1, rgb2], axis=1)
        sem = np.concatenate([sem1, sem2], axis=1)

        rgb = self.augmenter(images=rgb[...,::-1][None])[0]
        sem = filter_sem(sem, self.seg_channels)

        # Vehicle locations/orientations
        ego_id, ego_locs, ego_oris, ego_bbox, msks, locs, oris, bbox, typs = self.__class__.filter(
            lmdb_txn, index,
            max_pedestrian_radius=self.max_pedestrian_radius,
            max_vehicle_radius=self.max_vehicle_radius,
            T=self.num_plan)

        # Normalize coordinates to ego frame
        ego_locs, locs, oris, bbox, typs = transform_ego(ego_locs, locs, oris, bbox, typs, ego_oris[0], self.num_plan+1)

        cmd = int(self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.uint8))
        nxp = self.__class__.access('nxp', lmdb_txn, index, 1).reshape(2)

        return rgb, sem, bev, -(ego_locs[1:]-ego_locs[:1]), cmd, nxp
