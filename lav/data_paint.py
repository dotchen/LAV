import ray
import tqdm
import yaml
import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import torch.nn.functional as F
from lav.models.rgb import RGBSegmentationModel
from lav.utils.logger import Logger
from lav.utils.datasets.point_paint_dataset import PointPaintDataset
from lav.utils import _numpy
from lav.utils.point_painting import CoordConverter, point_painting
from lav.utils.visualization import lidar_to_bev, visualize_semantic_processed

class Coordinator:
    def __init__(self, args):
        wandb.init(project='lav_point_painting')

        self.dataset = PointPaintDataset(args.config_path)

    @property
    def num_frames(self):
        return self.dataset.num_frames

    def log(self, sems, lidar, lidar_painted):

        num_channels = len(self.dataset.seg_channels)

        f, axes = plt.subplots(1,num_channels,figsize=(4*num_channels,4))
        for i in range(num_channels):
            lidar_viz = lidar_to_bev(lidar[lidar_painted[:,i]>0.5])
            axes[i].imshow(lidar_viz.astype(np.uint8))

        sem = visualize_semantic_processed(np.concatenate(sems, axis=1), labels=self.dataset.seg_channels)

        wandb.log({'viz': wandb.Image(plt), 'sem': wandb.Image(sem)})
        plt.close('all')

    def commit(self, idx, lidar_painted):
        self.dataset.commit(idx, lidar_painted)

@ray.remote(num_gpus=1./4)
class PointPainter:
    def __init__(self, args):
        with open(args.config_path, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        # Save configs
        self.device = torch.device(args.device)

        self.seg_model = RGBSegmentationModel(self.seg_channels).to(self.device)
        self.seg_model.load_state_dict(torch.load(self.seg_model_dir, map_location=self.device))
        self.seg_model.eval()

        self.coord_converters = [CoordConverter(
            cam_yaw, lidar_xyz=[0,0,self.camera_z], cam_xyz=[self.camera_x,0,self.camera_z],
            rgb_h=288, rgb_w=256, fov=64
        ) for cam_yaw in self.camera_yaws]

        self.dataset = PointPaintDataset(args.config_path)


    def step(self, idx):

        lidar, rgbs = self.dataset[idx]

        rgbs = torch.tensor(rgbs.copy()).float().to(self.device)
        sems = _numpy(torch.softmax(self.seg_model(rgbs), dim=1))

        # Normalize with the first channel (background)
        norm_sems = sems[:,1:] * (1-sems[:,:1])

        lidar_painted = point_painting(lidar, norm_sems, self.coord_converters)

        # write to lmdb
        return idx, sems.argmax(axis=1), lidar, lidar_painted


def main(args):

    ray.init(logging_level=30, local_mode=False, log_to_driver=False)    

    workers = [PointPainter.remote(args) for _ in range(args.num_workers)]
    coordinator = Coordinator(args)
    num_frames = coordinator.num_frames

    num_completed = 0
    pbar = tqdm.tqdm(total=num_frames)

    for idx in range(0, num_frames, args.num_workers):
        idxes = range(idx,min(idx+args.num_workers, num_frames))
        
        jobs = [worker.step.remote(idx) for idx, worker in zip(idxes, workers)]
        
        # for idx, sems, lidar, lidar_painted in workers.map(lambda a, i: a.step.remote(i), idxes):
        for idx, sems, lidar, lidar_painted in ray.get(jobs):
            num_completed += 1
            
            coordinator.commit(idx, lidar_painted)
            
            if num_completed % args.num_per_log == 0:
                coordinator.log(sems, lidar, lidar_painted)
                pbar.update(args.num_per_log)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config-path', default='config.yaml')

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])

    # Training misc
    parser.add_argument('--num-per-log', type=int, default=100, help='log per iter')

    parser.add_argument('--num-workers', type=int, default=8)

    args = parser.parse_args()

    main(args)
