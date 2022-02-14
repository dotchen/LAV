import numpy as np
import cv2
import torch
from .basic_dataset import BasicDataset


class LiDARDataset(BasicDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.x_edges = np.linspace(self.min_x, self.max_x, (self.max_x-self.min_x)*self.pixels_per_meter)
        self.y_edges = np.linspace(self.min_y, self.max_y, (self.max_y-self.min_y)*self.pixels_per_meter)

    def preprocess(self, lidar_xyzr, lidar_painted=None):

        idx = (lidar_xyzr[:,0] > -2.4)&(lidar_xyzr[:,0] < 0)&(lidar_xyzr[:,1]>-0.8)&(lidar_xyzr[:,1]<0.8)&(lidar_xyzr[:,2]>-1.5)&(lidar_xyzr[:,2]<-1)

        idx = np.argwhere(idx)

        if lidar_painted is None:
            return np.delete(lidar_xyzr, idx, axis=0)
        else:
            return np.delete(lidar_xyzr, idx, axis=0), np.delete(lidar_painted, idx, axis=0)

    def __getitem__(self, idx):

        lmdb_txn = self.txn_map[idx]
        index = self.idx_map[idx]

        lidar = self.__class__.access('lidar', lmdb_txn, index, 1).reshape(-1,4)

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

        lidar = self.preprocess(lidar)

        lidar_xyzr = rotate_lidar(lidar[:,:4], -angle)
        ego_locs = rotate_points(ego_locs, -angle, ego_locs[0])
        nxp      = rotate_points(nxp, -angle, ego_locs[0])

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

        lidar = np.empty((self.max_lidar_points, 4), dtype=np.float32)
        num_points = min(self.max_lidar_points, len(lidar_xyzr))
        lidar[:num_points,:4] = lidar_xyzr

        return (
            lidar, num_points,
            heatmaps, sizemaps, orimaps,                     # Detection targets 
            bev,                                             # Segmentation targets
            -ego_locs, cmd, -nxp, bra,                       # Planning targets
            -padded_locs, padded_oris, padded_typs, num_objs # Motion forecast targets
        )

    def detections_to_heatmap(self, locs, oris, bbox, typs, radius=1):
        h, w = len(self.y_edges), len(self.x_edges)
        heatmap = torch.zeros((2, h, w)) # 2 is for class: pedestrians/vehicles
        sizemap = torch.zeros((2, h, w)) # 2 is for bbox w/h
        orimap  = torch.zeros((2, h, w)) # for sin and cos

        for i in [0,1]:

            idx = typs==i
            if sum(idx)==0:
                continue

            loc = torch.tensor(locs[idx], dtype=torch.float32)
            ori = torch.tensor(oris[idx], dtype=torch.float32)
            box = torch.tensor(bbox[idx], dtype=torch.float32)

            # To BEV coordinates
            x = torch.arange(w)
            y = torch.arange(h)
            cx, cy = loc[:,0]*self.pixels_per_meter, loc[:,1]*self.pixels_per_meter

            cx = -cx + (self.max_y-self.min_y)*self.pixels_per_meter/2
            cy = -cy + h+self.min_x*self.pixels_per_meter

            gx = (-((x[:, None] - cx[None, :]) / radius)**2).exp()
            gy = (-((y[:, None] - cy[None, :]) / radius)**2).exp()

            gaussian, id = (gx[None] * gy[:, None]).max(dim=-1)
            mask = gaussian > heatmap.max(dim=0)[0]

            sizemap[:, mask] = box.T[:,id[mask]]*self.pixels_per_meter
            orimap[0, mask] = np.cos(ori[id[mask]])
            orimap[1, mask] = np.sin(ori[id[mask]])
            heatmap[i] = gaussian

        return heatmap, sizemap, orimap



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



def rotate_image(image, angle, image_center=(160,280)):
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    return result


def rotate_lidar(lidar, angle):
    radian = np.deg2rad(angle)
    return lidar @ [
        [ np.cos(radian), np.sin(radian), 0, 0],
        [-np.sin(radian), np.cos(radian), 0, 0],
        [0,0,1,0],
        [0,0,0,1]
    ]


def rotate_points(points, angle, ego_loc):
    radian = np.deg2rad(angle)
    return (points-ego_loc) @ [
        [ np.cos(radian), np.sin(radian)],
        [-np.sin(radian), np.cos(radian)]
    ] + ego_loc


if __name__ == '__main__':

    dataset = LiDARDataset('config.yaml')
    import tqdm
    for i in tqdm.tqdm(range(100)):
        dataset[i]
