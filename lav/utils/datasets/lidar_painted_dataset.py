import numpy as np
import cv2
import torch
from lav.utils.point_painting import CoordConverter, point_painting
from .lidar_dataset import LiDARDataset, transform_ego, rotate_image, rotate_lidar, rotate_points

class LiDARPaintedDataset(LiDARDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.coord_converters = [CoordConverter(
            cam_yaw, lidar_xyz=[0,0,self.camera_z], cam_xyz=[self.camera_x,0,self.camera_z],
            rgb_h=288, rgb_w=256, fov=64
        ) for cam_yaw in self.camera_yaws[1:-1]]

        self.dummy = np.ones((len(self.camera_yaws[1:-1]),1,288,256))

        self.margin = 32

    def __getitem__(self, idx):

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]

        lidar_xyzr = self.__class__.access('lidar', lmdb_txn, index, 1).reshape(-1,4)
        lidar_painted = self.__class__.access('lidar_sem', lmdb_txn, index, 1).reshape(-1,len(self.seg_channels))

        lidar_xyzr, lidar_painted = self.preprocess(lidar_xyzr, lidar_painted)

        # Vehicle locations/orientations
        ego_id, ego_locs, ego_oris, ego_bbox, msks, locs, oris, bbox, typs = self.__class__.filter(
            lmdb_txn, index,
            max_pedestrian_radius=self.max_pedestrian_radius,
            max_vehicle_radius=self.max_vehicle_radius,
            T=self.num_plan)

        # Normalize coordinates to ego frame
        ego_locs, locs, oris, bbox, typs = transform_ego(ego_locs, locs, oris, bbox, typs, ego_oris[0], self.num_plan+1)

        # Random rotation jitter
        angle = float(torch.rand(1)*2-1)*self.angle_jitter

        cmd = int(self.__class__.access('cmd', lmdb_txn, index, 1, dtype=np.uint8))
        bra = int(self.__class__.access('bra', lmdb_txn, index, 1, dtype=np.uint8))
        nxp = self.__class__.access('nxp', lmdb_txn, index, 1).reshape(2)

        # BEV images
        bev = self.__class__.load_bev(lmdb_txn, index, channels=[0,1,2,9,10])
        bev = rotate_image(bev, angle)
        bev = (bev>0).astype(np.uint8).transpose(2,0,1)

        lidar_xyzr = rotate_lidar(lidar_xyzr, -angle)
        ego_locs   = rotate_points(ego_locs, -angle, ego_locs[0])
        nxp        = rotate_points(nxp, -angle, ego_locs[0])

        lidar_painted_mask = point_painting(lidar_xyzr, self.dummy, self.coord_converters)
        lidar_painted *= lidar_painted_mask

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

        idxs = np.arange(len(lidar_xyzr))
        np.random.shuffle(idxs)
        lidar_xyzr = lidar_xyzr[idxs]
        lidar_painted = lidar_painted[idxs]

        lidar = np.empty((self.max_lidar_points, 4+len(self.seg_channels)), dtype=np.float32)
        num_points = min(self.max_lidar_points, len(lidar_xyzr))
        lidar[:num_points,:4] = lidar_xyzr
        lidar[:num_points,4:] = lidar_painted

        return (
            lidar, num_points,
            heatmaps, sizemaps, orimaps,                     # Detection targets 
            bev,                                             # Segmentation targets
            -ego_locs, cmd, -nxp, bra,                       # Planning targets
            -padded_locs, padded_oris, padded_typs, num_objs # Motion forecast targets
        )
