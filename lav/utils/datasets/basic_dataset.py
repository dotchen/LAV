import glob
import lmdb
import yaml
import cv2
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict

TRAIN_TOWNS = ['Town01', 'Town03', 'Town04', 'Town06']

class BasicDataset(Dataset):
    def __init__(self, config_path, close_txn=False, seed=2021):
        super().__init__()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        self.num_frames = 0
        self.nam_map = dict()
        self.txn_map = dict()
        self.idx_map = dict()
        self.dir_map = dict()

        num_frames = defaultdict(int)

        np.random.seed(seed)
        print ("Using np random seed:", seed)
        print ("Using all towns:", self.all_towns)

        # Load dataset
        for full_path in glob.glob('{}/**'.format(self.data_dir)):
            
            # Toss a coin
            if np.random.random() > self.percentage_data:
                continue
            exist_file = False
            for lmb_file in glob.glob('{}/**'.format(full_path)):
                # print(lmb_file.split('/')[-1])
                if lmb_file.split('/')[-1] == 'data.mdb':
                    exist_file = True
            if not exist_file:
                print('data.mdb is not exist in folder', full_path)
                continue
            txn = lmdb.open(
                full_path,
                max_readers=1, readonly=True,
                lock=False, readahead=False, meminit=False).begin(write=False)

            n = int(txn.get('len'.encode()))
            town = str(txn.get('town'.encode()))[2:-1]


            if not self.all_towns and town not in TRAIN_TOWNS:
                continue

            offset = self.num_frames
            for i in range(n-self.num_plan):
                self.num_frames += 1
                self.nam_map[offset+i] = full_path
                self.txn_map[offset+i] = txn
                self.idx_map[offset+i] = i # ????? why +1
                self.dir_map[offset+i] = full_path

                num_frames[town] += 1


            # Note: skip the first frame
            if close_txn or n < self.num_plan + 1:
                txn.__exit__()

        print('{}: {} frames'.format(self.data_dir, self.num_frames))
        # print (num_frames, sum(num_frames.values()))
    
    def __len__(self):
        return self.num_frames


    @staticmethod
    def access(tag, lmdb_txn, index, T, suffix='', dtype=np.float32):
        return np.stack([np.frombuffer(lmdb_txn.get((f'{tag}_{t:05d}{suffix}').encode()), dtype) for t in range(index,index+T)])

    @staticmethod
    def load_img(lmdb_txn, tag, idx):
        if 'rgb' in tag:
            mode = cv2.IMREAD_COLOR
        elif 'sem' in tag:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            raise NotImplemented

        return cv2.imdecode(np.frombuffer(lmdb_txn.get('{}_{:05d}'.format(tag, idx).encode()), dtype=np.uint8), mode)

    @staticmethod
    def load_bev(lmdb_txn, idx, channels=range(12)):

        bevs = [cv2.imdecode(np.frombuffer(lmdb_txn.get(f'map_{i}_{idx:05d}'.encode()), dtype=np.uint8), cv2.IMREAD_GRAYSCALE) for i in channels]

        return np.stack(bevs, axis=-1)
    
    @staticmethod
    def filter(lmdb_txn, index, max_pedestrian_radius=10, max_vehicle_radius=20, T=10):

        ids_0 = BasicDataset.access('id',  lmdb_txn, index, 1, dtype=np.int32).flatten()
        ego_id = ids_0[0]

        msks = {actor_id: np.zeros(T+1)     for actor_id in ids_0}
        locs = {actor_id: np.zeros((T+1,2)) for actor_id in ids_0}
        oris = {actor_id: np.zeros(T+1)     for actor_id in ids_0}
        bbox = {actor_id: np.zeros((T+1,2)) for actor_id in ids_0}
        typs = {actor_id: np.zeros(T+1)     for actor_id in ids_0}

        for t in range(index,index+T+1):

            ids_t   = BasicDataset.access('id',   lmdb_txn, t, 1, dtype=np.int32).flatten()
            locs_t  = BasicDataset.access('loc',  lmdb_txn, t, 1).reshape(-1,2)
            oris_t  = BasicDataset.access('ori',  lmdb_txn, t, 1).flatten()
            bboxs_t = BasicDataset.access('bbox', lmdb_txn, t, 1).reshape(-1,2)
            types_t = BasicDataset.access('type', lmdb_txn, t, 1, dtype=np.uint8).flatten()

            for id_t, loc_t, ori_t, bbox_t, type_t in zip(ids_t, locs_t, oris_t, bboxs_t, types_t):

                if id_t not in ids_0:
                    continue

                msks[id_t][t-index] = 1
                locs[id_t][t-index] = loc_t
                oris[id_t][t-index] = np.deg2rad(ori_t)
                bbox[id_t][t-index] = bbox_t
                typs[id_t][t-index] = type_t

        ego_locs = locs[ego_id]
        ego_oris = oris[ego_id]
        ego_bbox = bbox[ego_id]

        to_pop = set([])
        for actor_id, msk in msks.items():
            if not np.all(msk):
                to_pop.add(actor_id)
        
        for actor_id in msks.keys():
            loc = locs[actor_id][0]
            if typs[actor_id][0] == 0 and np.linalg.norm(loc-ego_locs[0]) > max_pedestrian_radius:
                to_pop.add(actor_id)
            elif typs[actor_id][0] == 1 and np.linalg.norm(loc-ego_locs[0]) > max_vehicle_radius:
                to_pop.add(actor_id)

        for actor_id in to_pop:
            msks.pop(actor_id)
            typs.pop(actor_id)
            locs.pop(actor_id)
            oris.pop(actor_id)
            bbox.pop(actor_id)

        return ego_id, ego_locs, ego_oris, ego_bbox, msks, locs, oris, bbox, typs
