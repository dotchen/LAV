import numpy as np
import cv2
import torch
from .basic_dataset import BasicDataset
from PIL import Image

TOWNS = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town06', 'Town07', 'Town10HD']

class BEVDataset(BasicDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        w = h = self.crop_size
        world_w = world_h = self.crop_size//self.pixels_per_meter
        self.margin = 32

    def __getitem__(self, idx):

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]

        # Vehicle locations/orientations
        ego_id, ego_locs, ego_oris, ego_bbox, msks, locs, oris, bbox, typs = self.__class__.filter(
            lmdb_txn, index,
            max_pedestrian_radius=self.max_pedestrian_radius,
            max_vehicle_radius=self.max_vehicle_radius,
            T=self.num_plan)

        # Normalize coordinates to ego frame
        ego_locs, locs, oris, bbox, typs = transform_ego(ego_locs, locs, oris, bbox, typs, ego_oris[0], self.num_plan+1)

        # Random jitter
        offset = int((torch.rand(1)*2-1)*self.x_jitter)
        offset = np.clip(offset, -self.margin, self.margin)
        angle  = float(torch.rand(1)*2-1)*self.angle_jitter

        # BEV images
        bev = self.__class__.load_bev(lmdb_txn, index, channels=[0,1,2,9,10])
        bev = rotate_image(bev, angle)
        bev = (bev>0).astype(np.uint8).transpose(2,0,1)
        bev = np.pad(bev, [[0,0],[self.margin,self.margin],[self.margin,self.margin]])
        bev = bev[:,self.margin:self.margin+320,self.margin+offset:self.margin+offset+320]

        nxp = self.__class__.access('nxp', lmdb_txn, index, 1).reshape(2)

        ego_locs = rotate_points(ego_locs, -angle, ego_locs[0]) + [offset/self.pixels_per_meter, 0]
        nxp      = rotate_points(nxp, -angle, ego_locs[0]) + [offset/self.pixels_per_meter, 0]

        cmd = int(self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.uint8))
        bra = int(self.__class__.access('bra', lmdb_txn, index, 1, dtype=np.uint8))

        # Overwrite cmd with the additional STOP command.
        spd = np.mean(np.linalg.norm(ego_locs[1:]-ego_locs[:-1],axis=-1))

        locs = rotate_points(locs, -angle, ego_locs[0]) + [offset/self.pixels_per_meter, 0]
        oris[1:] = oris[1:] - np.deg2rad(angle) # Ego vehicle not affected

        # Pad tensors
        num_objs    = min(len(locs), self.max_objs)
        padded_locs = np.zeros((self.max_objs,self.num_plan+1,2), dtype=np.float32)
        padded_oris = np.zeros((self.max_objs,), dtype=np.float32)
        padded_typs = np.zeros((self.max_objs,), dtype=np.int32)

        padded_locs[:num_objs] = locs[:num_objs]
        padded_oris[:num_objs] = oris[:num_objs,0]
        padded_typs[:num_objs] = typs[:num_objs,0]

        return (
            bev,                                             # Segmentation targets
            -ego_locs, cmd, -nxp, bra,                       # Planning targets
            -padded_locs, padded_oris, padded_typs, num_objs # Motion forecast targets
        )



def rotate_image(image, angle, image_center=(160,280)):
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def rotate_points(points, angle, ego_loc):
    radian = np.deg2rad(angle)
    return (points-ego_loc) @ [
        [ np.cos(radian), np.sin(radian)],
        [-np.sin(radian), np.cos(radian)]
    ] + ego_loc


def transform_ego(ego_locs, locs, oris, bbox, typs, ego_ori, T=11):
    
    ego_loc = ego_locs[0]
    
    keys = sorted(list(locs.keys()))
    locs = np.array([locs[k] for k in keys]).reshape(-1,T,2)
    oris = np.array([oris[k] for k in keys]).reshape(-1,T)
    bbox = np.array([bbox[k] for k in keys]).reshape(-1,T,2)
    typs = np.array([typs[k] for k in keys]).reshape(-1,T)

    R = [[np.sin(ego_ori),np.cos(ego_ori)],[-np.cos(ego_ori),np.sin(ego_ori)]] 
    
    locs = (locs-ego_loc) @ R
    ego_locs = (ego_locs-ego_loc) @ R
    oris = oris - ego_ori

    return ego_locs, locs, oris, bbox, typs


if __name__ == '__main__':

    dataset = BEVDataset('config.yaml')

    import tqdm
    for t in tqdm.tqdm(range(0,20)):
        dataset[t]
