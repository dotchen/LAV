import math
import numpy as np
import cv2
import torch
from PIL import Image
from .bev_dataset import BEVDataset, rotate_image, rotate_points, transform_ego

class TemporalBEVDataset(BEVDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        bev = np.zeros((3+2*(self.num_frame_stack+1),320,320), dtype=np.uint8)
        bev[:3] = self.load_bev_channels(lmdb_txn, index, angle_offset=angle, channels=[0,9,10], y_offset=offset)

        for t, i in enumerate(reversed(range(index-self.num_frame_stack,index+1))):
            if i < 0:
                continue

            _, _ego_locs, _ego_oris, _, _, _, _, _, _ = self.__class__.filter(
                lmdb_txn, i,
                max_pedestrian_radius=self.max_pedestrian_radius,
                max_vehicle_radius=self.max_vehicle_radius,
                T=self.num_plan)

            ego_loc = _ego_locs[0]
            ego_ori = _ego_oris[0]

            if i == index:
                ego_loc0, ego_ori0 = ego_loc, ego_ori

            dloc = (ego_loc - ego_loc0) @ [
                [ np.cos(ego_ori0), -np.sin(ego_ori0)],
                [ np.sin(ego_ori0),  np.cos(ego_ori0)]
            ] * self.pixels_per_meter

            bev[3+t*2:3+(t+1)*2] = self.load_bev_channels(
                lmdb_txn, i, 
                angle_offset=angle,
                angle=ego_ori-ego_ori0,
                y_offset=offset,
                channels=[1,2],
                loc=dloc
            )

        # bev_road = self.__class__.load_bev(lmdb_txn, index, channels=[0,1,2,9,10])
        # bev_road = rotate_image(bev_road, angle)
        # bev_road = (bev_road>0).astype(np.uint8).transpose(2,0,1)
        # bev_road = np.pad(bev_road, [[0,0],[16,16],[16,16]])
        # bev_road = bev_road[:,self.margin:self.margin+320,self.margin+offset:self.margin+offset+320]

        locs = rotate_points(locs, -angle, ego_locs[0]) + [offset/self.pixels_per_meter, 0]
        oris[1:] = oris[1:] - np.deg2rad(angle) # Ego vehicle not affected

        nxp = self.__class__.access('nxp', lmdb_txn, index, 1).reshape(2)

        ego_locs = rotate_points(ego_locs, -angle, ego_locs[0]) + [offset/self.pixels_per_meter, 0]
        nxp      = rotate_points(nxp, -angle, ego_locs[0]) + [offset/self.pixels_per_meter, 0]

        cmd = int(self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.uint8))
        bra = int(self.__class__.access('bra', lmdb_txn, index, 1, dtype=np.uint8))

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

    def load_bev_channels(self, lmdb_txn, index, channels=[0,1,2,9,10], angle=0, angle_offset=0, y_offset=0, loc=np.array([0,0])):

        # dx, dy = loc @ [
        #     [ np.cos(angle_offset),  -np.sin(angle_offset)],
        #     [ np.sin(angle_offset),  np.cos(angle_offset)],
        # ]
        # dx, dy = map(int, [dx, dy])

        dx, dy = map(int, loc)

        bev = self.__class__.load_bev(lmdb_txn, index, channels=channels)
        bev = rotate_image(bev, -angle*180/math.pi)
        bev = np.pad(bev, [[self.margin,self.margin],[self.margin,self.margin],[0,0]])
        bev = bev[dx+self.margin:dx+self.margin+320,dy+self.margin+y_offset:dy+self.margin+y_offset+320,:]
        bev = rotate_image(bev, angle_offset)
        bev = (bev>0).astype(np.uint8).transpose(2,0,1)
        return bev


if __name__ == '__main__':

    dataset = TemporalBEVDataset('config.yaml')
    import tqdm
    for i in tqdm.tqdm(range(100)):
        dataset[i]
