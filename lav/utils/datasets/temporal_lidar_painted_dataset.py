import math
import numpy as np
import numba
import cv2
import torch
from lav.utils.point_painting import CoordConverter, point_painting
from .lidar_dataset import LiDARDataset, transform_ego, rotate_image, rotate_lidar, rotate_points

from .lidar_painted_dataset import LiDARPaintedDataset

class TemporalLiDARPaintedDataset(LiDARPaintedDataset):

    def __getitem__(self, idx):

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]

        lidars_xyzr = []
        lidars_painted = []

        # Random rotation jitter
        angle = float(torch.rand(1)*2-1)*self.angle_jitter

        # Extract LiDAR and propriceptive sensors
        for i in reversed(range(index-self.num_frame_stack,index+1)):

            if i < 0:
                continue

            # print (i, index)

            lidar_xyzr = self.__class__.access('lidar', lmdb_txn, i, 1).reshape(-1,4)
            lidar_painted = self.__class__.access('lidar_sem', lmdb_txn, i, 1).reshape(-1,len(self.seg_channels))

            lidar_xyzr, lidar_painted = self.preprocess(lidar_xyzr, lidar_painted)

            # Vehicle locations/orientations
            ego_id, ego_locs, ego_oris, ego_bbox, msks, locs, oris, bbox, typs = self.__class__.filter(
                lmdb_txn, i,
                max_pedestrian_radius=self.max_pedestrian_radius,
                max_vehicle_radius=self.max_vehicle_radius,
                T=self.num_plan)

            ego_loc = ego_locs[0]
            ego_ori = ego_oris[0]

            if i == index:
                ego_loc0, ego_ori0 = ego_loc, ego_ori
                loc_jitter, ori_jitter = 0, 0
            else:
                loc_jitter = np.random.uniform(low=-self.stack_loc_jitter, high=self.stack_loc_jitter, size=2)
                ori_jitter = np.random.uniform(low=-self.stack_ori_jitter, high=self.stack_ori_jitter)

            # Rotation augment
            lidar_xyzr = rotate_lidar(lidar_xyzr, -angle)
            lidar_painted_mask = point_painting(lidar_xyzr, self.dummy, self.coord_converters)

            lidar_painted *= lidar_painted_mask

            lidar_xyzr = move_lidar_points(
                lidar_xyzr, 
                ego_loc-ego_loc0+loc_jitter,
                ego_ori0, ego_ori+ori_jitter
            )

            lidars_xyzr.append(lidar_xyzr)
            lidars_painted.append(lidar_painted)


        lidar = np.zeros((sum(map(len, lidars_xyzr)), 4+len(self.seg_channels)+self.num_frame_stack+1), dtype=np.float32)
        total_num_points = 0

        # Stack-up
        for t, (lidar_xyzr, lidar_painted) in enumerate(zip(lidars_xyzr, lidars_painted)):
            num_points = len(lidar_xyzr)
            lidar[total_num_points:total_num_points+num_points,:4] = lidar_xyzr
            lidar[total_num_points:total_num_points+num_points,4:4+len(self.seg_channels)] = lidar_painted
            lidar[total_num_points:total_num_points+num_points,4+len(self.seg_channels)+t] = 1.

            total_num_points += num_points

        idxs = np.arange(len(lidar))
        np.random.shuffle(idxs)
        lidar = lidar[idxs[:self.max_lidar_points]]

        cmd = int(self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.uint8))
        bra = int(self.__class__.access('bra', lmdb_txn, index, 1, dtype=np.uint8))
        nxp = self.__class__.access('nxp', lmdb_txn, index, 1).reshape(2)

        # BEV images
        # bev = self.__class__.load_bev(lmdb_txn, index, channels=[0,1,2,9,10])
        # bev = rotate_image(bev, angle)
        # bev = (bev>0).astype(np.uint8).transpose(2,0,1)

        # Detection: Vehicle locations/orientations
        ego_id, ego_locs, ego_oris, ego_bbox, msks, locs, oris, bbox, typs = self.__class__.filter(
            lmdb_txn, index,
            max_pedestrian_radius=self.max_pedestrian_radius,
            max_vehicle_radius=self.max_vehicle_radius,
            T=self.num_plan)

        # Normalize coordinates to ego frame
        ego_locs, locs, oris, bbox, typs = transform_ego(ego_locs, locs, oris, bbox, typs, ego_oris[0], self.num_plan+1)

        # Temporal stacked BEV images
        bev = np.zeros((3+2*(self.num_frame_stack+1),320,320), dtype=np.uint8)
        bev[:3] = self.load_bev_channels(lmdb_txn, index, angle_offset=angle, channels=[0,9,10])

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
                channels=[1,2],
                loc=dloc
            )


        locs = rotate_points(locs, -angle, ego_locs[0])
        oris[1:] = oris[1:] - np.deg2rad(angle) # Ego vehicle not affected

        # Detection labels
        heatmaps, sizemaps, orimaps = self.detections_to_heatmap(locs[:,0], oris[:,0], bbox[:,0], typs[:,0])

        # Pad tensors
        num_objs    = min(len(locs), self.max_objs)
        padded_locs = np.zeros((self.max_objs,self.num_plan+1,2), dtype=np.float32)
        padded_oris = np.zeros((self.max_objs,), dtype=np.float32)
        padded_typs = np.zeros((self.max_objs,), dtype=np.int32)

        padded_locs[:num_objs] = locs[:num_objs]
        padded_oris[:num_objs] = oris[:num_objs,0]
        padded_typs[:num_objs] = typs[:num_objs,0]

        padded_lidar = np.zeros((self.max_lidar_points, lidar.shape[1]), dtype=np.float32)
        num_points = min(self.max_lidar_points, total_num_points)
        padded_lidar[:num_points] = lidar[:num_points]

        # Motion: Vehicle locations/orientations
        # Note that the range is more strict (smaller)
        ego_id, ego_locs, ego_oris, ego_bbox, msks, locs, oris, bbox, typs = self.__class__.filter(
            lmdb_txn, index,
            max_pedestrian_radius=self.max_pedestrian_radius,
            max_vehicle_radius=self.max_mot_vehicle_radius,
            T=self.num_plan)

        ego_locs, locs, oris, bbox, typs = transform_ego(ego_locs, locs, oris, bbox, typs, ego_oris[0], self.num_plan+1)

        ego_locs   = rotate_points(ego_locs, -angle, ego_locs[0])
        nxp        = rotate_points(nxp, -angle, ego_locs[0])

        return (
            padded_lidar, num_points,
            # pillars, num_points, coords,                   # LiDAR inputs 
            heatmaps, sizemaps, orimaps,                     # Detection targets 
            bev,                                             # Segmentation targets
            -ego_locs, cmd, -nxp, bra,                       # Planning targets
            -padded_locs, padded_oris, padded_typs, num_objs # Motion forecast targets
        )


    def load_bev_channels(self, lmdb_txn, index, channels=[0,1,2,9,10], angle=0, angle_offset=0, loc=np.array([0,0])):

        # dx, dy = loc @ [
        #     [ np.cos(angle_offset),  -np.sin(angle_offset)],
        #     [ np.sin(angle_offset),  np.cos(angle_offset)],
        # ]
        # dx, dy = map(int, [dx, dy])

        dx, dy = map(int, loc)

        bev = self.__class__.load_bev(lmdb_txn, index, channels=channels)
        bev = rotate_image(bev, -angle*180/math.pi)
        bev = np.pad(bev, [[self.margin,self.margin],[self.margin,self.margin],[0,0]])
        bev = bev[dx+self.margin:dx+self.margin+320,dy+self.margin:dy+self.margin+320,:]
        bev = rotate_image(bev, angle_offset)
        bev = (bev>0).astype(np.uint8).transpose(2,0,1)
        return bev


def move_lidar_points(lidar, dloc, ori0, ori1):

    dloc = dloc @ [
        [ np.cos(ori0), -np.sin(ori0)],
        [ np.sin(ori0), np.cos(ori0)]
    ]

    ori = ori1 - ori0
    lidar = lidar @ [
        [np.cos(ori), np.sin(ori),0,0],
        [-np.sin(ori), np.cos(ori),0,0],
        [0,0,1,0],
        [0,0,0,1]
    ]

    lidar[:,:2] += dloc

    return lidar


if __name__ == '__main__':

    dataset = TemporalLiDARPaintedDataset('config.yaml')
    import tqdm
    for i in tqdm.tqdm(range(100)):
        dataset[i]
