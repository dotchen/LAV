import glob
import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset

class EgoDataset(Dataset):
    def __init__(self, data_dir, T=10):
        super().__init__()
        
        self.T = T
        
        self.num_frames = 0
        self.txn_map = dict()
        self.idx_map = dict()
        
        # Load dataset
        for full_path in glob.glob(f'{data_dir}/**'):
            txn = lmdb.open(
                full_path,
                max_readers=1, readonly=True,
                lock=False, readahead=False, meminit=False).begin(write=False)
            
            n = int(txn.get('len'.encode()))
            if n < T:
                txn.__exit__()
            else:
                offset = self.num_frames
                for i in range(n-T+1):
                    self.num_frames += 1
                    self.txn_map[offset+i] = txn
                    self.idx_map[offset+i] = i
            
        print(f'{data_dir}: {self.num_frames} frames')

    def __len__(self):
        return self.num_frames
    
    def __getitem__(self, idx):
        
        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]
        
        locs = self.__class__.access('loc', lmdb_txn, index, self.T)
        rots = self.__class__.access('rot', lmdb_txn, index, self.T)
        spds = self.__class__.access('spd', lmdb_txn, index, self.T)
        acts = self.__class__.access('act', lmdb_txn, index, self.T)
        
        return locs, rots, spds, acts
    
    @staticmethod
    def access(tag, lmdb_txn, index, T):
        return np.stack([np.frombuffer(lmdb_txn.get((f'{tag}_{t:05d}').encode()), np.float32) for t in range(index,index+T)])

if __name__ == '__main__':
    
    dataset = EgoDataset('/ssd2/dian/challenge_data/ego_trajs')
    
    print (dataset[0])
