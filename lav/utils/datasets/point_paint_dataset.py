import lmdb
import numpy as np
import cv2
import torch
from .basic_dataset import BasicDataset


class PointPaintDataset(BasicDataset):
    def __init__(self, config_path):
        super().__init__(config_path, close_txn=True)


    def __getitem__(self, idx):

        full_path = self.nam_map[idx]
        lmdb_env = lmdb.open(
                full_path,
                readonly=True,
                lock=False, readahead=False, meminit=False)

        index = self.idx_map[idx]
        
        with lmdb_env.begin(write=False) as lmdb_txn:
            lidar = self.__class__.access('lidar', lmdb_txn, index, 1).reshape(-1,4)
  
            rgbs = np.stack([
                self.__class__.load_img(lmdb_txn, 'rgb_{}'.format(camera_index), index)
            for camera_index in range(len(self.camera_yaws))])

        lmdb_env.close()

        return lidar, rgbs[...,::-1].transpose((0,3,1,2))

    def commit(self, idx, lidar_painted):
        
        file_name = self.nam_map[idx]
        file_idx  = self.idx_map[idx]

        lmdb_env = lmdb.open(file_name, map_size=int(1e10))
        with lmdb_env.begin(write=True) as txn:
            txn.put(
                f'lidar_sem_{file_idx:05d}'.encode(),
                np.ascontiguousarray(lidar_painted).astype(np.float32),
            )

        lmdb_env.close()

if __name__ == '__main__':
    
    dataset = PointPaintDataset('config.yaml')
    import tqdm
    for i in tqdm.tqdm(range(20)):
        lidar, sems = dataset[i]
        print (lidar.shape, sems.shape)
