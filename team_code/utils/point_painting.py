import carla
import numpy as np


class CoordConverter:
    def __init__(self, cam_yaw, lidar_xyz=[0,0,2.5], cam_xyz=[1.4,0,2.5], rgb_h=320, rgb_w=320, fov=60):
        focal = rgb_w / (2.0 * np.tan(fov * np.pi / 360.0))
        
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = rgb_w / 2.0
        K[1, 2] = rgb_h / 2.0

        lidar_to_world = np.array(carla.Transform(
            carla.Location(*lidar_xyz),
        ).get_matrix())

        world_to_cam = np.array(carla.Transform(
            carla.Location(*cam_xyz),
            carla.Rotation(yaw=cam_yaw),
        ).get_inverse_matrix())

        self.K = K
        self.lidar_to_world = lidar_to_world
        self.world_to_cam   = world_to_cam

    def lidar_to_cam(self, lidar):

        lidar_xyz = lidar[:,:3].T
        lidar_xyz1 = np.r_[lidar_xyz, [np.ones(lidar_xyz.shape[1])]]

        world = self.lidar_to_world @ lidar_xyz1
        cam   = self.world_to_cam @ world

        cam   = np.array([cam[1], -cam[2], cam[0]])
        cam_2d = self.K @ cam

        cam_2d = np.array([
            cam_2d[0, :] / (1e-5+cam_2d[2, :]),
            cam_2d[1, :] / (1e-5+cam_2d[2, :]),
            cam_2d[2, :]]).T

        return cam_2d.astype(int)


def point_painting(lidar, sems, coord_converters):

    assert len(sems) == len(coord_converters)

    _, lidar_d = lidar.shape
    sem_c, sem_h, sem_w = sems[0].shape

    lidar_painted = np.zeros((len(lidar), sem_c))

    for sem, coord_converter in zip(sems, coord_converters):

        lidar_cam = coord_converter.lidar_to_cam(lidar)
        lidar_cam_u, lidar_cam_v, lidar_cam_z = map(lambda x: x[...,0], np.split(lidar_cam, 3, axis=-1))
        valid_idx = (lidar_cam_z>=0)&(lidar_cam_u>=0)&(lidar_cam_u<sem_w)&(lidar_cam_v>=0)&(lidar_cam_v<sem_h)
        lidar_cam = lidar_cam[valid_idx]

        lidar_sem = sem[:,lidar_cam[...,1],lidar_cam[...,0]].T

        lidar_painted[valid_idx] = lidar_sem
    
    return lidar_painted
