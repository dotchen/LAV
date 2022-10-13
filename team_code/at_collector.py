import os
import math
import yaml
import lmdb
import numpy as np
import cv2
import torch
import wandb
import carla
import random
import string
from copy import deepcopy

from torch.distributions.categorical import Categorical

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
from leaderboard.envs.pretty_map_utils import Wrapper as map_utils
from utils import visualize_obs, _numpy, PALETTE

from rails.bellman import BellmanUpdater
from rails.models import EgoModel
from waypointer import Waypointer
from planner import RoutePlanner
from autopilot import Autopilot


FPS = 20
STOP_THRESH = 0.1
MAX_STOP = 500
MAX_STUCK = 1000

def get_entry_point():
    return 'AutopilotCollector'


class AutopilotCollector(AutonomousAgent):
    
    def setup(self, path_to_conf_file):

        self.track = Track.MAP
        self.num_frames = 0

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        device = torch.device('cuda')
        ego_model = EgoModel(1./FPS*(self.num_repeat+1)).to(device)
        ego_model.load_state_dict(torch.load(self.ego_model_dir))
        ego_model.eval()
        BellmanUpdater.setup(config, ego_model, device=device)

        # Visualization stuff
        self.vizs = []

        # RGB camera
        self.rgbs = []
        self.tel_rgbs = []
        # semantic segmentation labels
        self.sems = []
        self.tel_sems = []
        
        # LiDAR points
        self.lids = []
        # BEV GT Maps
        self.maps = []

        # GNSS readings
        self.gnsses = []
        self.gnss = []

        # IMU readings
        self.lin_accs = []
        self.ang_vels = []
        self.lin_acc = []
        self.ang_vel = []
        
        # Actor ids 
        self.ids  = []
        # Actor locations (ego's always first and same below)
        self.locs = []
        # Actor orientations
        self.oris = []
        # Actor bounding boxes
        self.bbox = []
        # Actor types (pedestrian, walker)
        self.type = []
        # Actor speeds
        self.spds = []

        # Ego cmds
        self.cmds = []
        # Ego target waypoints
        self.nxps = []
        # Ego brakes
        self.bras = []

        if self.log_wandb:
            wandb.init(project='lav_data')

        self.autopilot  = None
        self.waypointer = None
        self.planner    = None
        self.town_name  = None
        self.prev_lidar = None
        
        self.stop_count = 0
        
    def destroy(self):
        if len(self.rgbs) == 0:
            return

        self.flush_data()

        self.autopilot  = None
        self.waypointer = None
        self.planner    = None
        self.town_name  = None
        self.prev_lidar = None
        
        self.stop_count = 0
        self.num_frames = 0

    def flush_data(self):

        if self.log_wandb:
            wandb.log({
                'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
            })

        # Save data
        data_path = os.path.join(self.data_dir, _random_string())

        print ('Saving to {}'.format(data_path))

        lmdb_env = lmdb.open(data_path, map_size=int(1e10))

        length = len(self.rgbs)
        with lmdb_env.begin(write=True) as txn:

            # Trajectory length
            txn.put('len'.encode(), str(length).encode())
            txn.put('town'.encode(), self.town_name.encode())

            for t in range(length):
                
                _, tel_rgb_bytes = cv2.imencode('.jpg', self.tel_rgbs[t])
                _, tel_sem_bytes = cv2.imencode('.png', self.tel_sems[t])

                txn.put(
                    f'tel_rgb_{t:05d}'.encode(),
                    np.ascontiguousarray(tel_rgb_bytes).astype(np.uint8)
                )

                txn.put(
                    f'tel_sem_{t:05d}'.encode(),
                    np.ascontiguousarray(tel_sem_bytes).astype(np.uint8)
                )

                # Camera RGB and semantic segmentation
                for i in range(len(self.camera_yaws)):
                    _, rgb_bytes = cv2.imencode('.jpg', self.rgbs[t][i])
                    _, sem_bytes = cv2.imencode('.png', self.sems[t][i])

                    txn.put(
                        f'rgb_{i}_{t:05d}'.encode(),
                        np.ascontiguousarray(rgb_bytes).astype(np.uint8)
                    )

                    txn.put(
                        f'sem_{i}_{t:05d}'.encode(),
                        np.ascontiguousarray(sem_bytes).astype(np.uint8)
                    )

                for channel in range(self.maps[t].shape[-1]):
                    _, map_bytes = cv2.imencode('.png', self.maps[t][...,channel])
                    txn.put(
                        f'map_{channel}_{t:05d}'.encode(),
                        np.ascontiguousarray(map_bytes).astype(np.uint8)
                    )

                # LiDAR points
                txn.put(
                    f'lidar_{t:05d}'.encode(),
                    np.ascontiguousarray(self.lids[t]).astype(np.float32)
                )

                # GNSS readings
                txn.put(
                    f'gnss_{t:05d}'.encode(),
                    np.ascontiguousarray(self.gnsses[t]).astype(np.float32)
                )

                # IMU readings
                txn.put(
                    f'lin_acc_{t:05d}'.encode(),
                    np.ascontiguousarray(self.lin_accs[t]).astype(np.float32)
                )

                txn.put(
                    f'ang_vel_{t:05d}'.encode(),
                    np.ascontiguousarray(self.ang_vels[t]).astype(np.float32)
                )

                # Proprioceptive readings
                txn.put(
                    f'id_{t:05d}'.encode(),
                    np.ascontiguousarray(self.ids[t]).astype(np.int32)
                )
                
                txn.put(
                    f'loc_{t:05d}'.encode(),
                    np.ascontiguousarray(self.locs[t]).astype(np.float32)
                )

                txn.put(
                    f'ori_{t:05d}'.encode(),
                    np.ascontiguousarray(self.oris[t]).astype(np.float32)
                )

                txn.put(
                    f'bbox_{t:05d}'.encode(),
                    np.ascontiguousarray(self.bbox[t]).astype(np.float32)
                )

                txn.put(
                    f'type_{t:05d}'.encode(),
                    np.ascontiguousarray(self.type[t]).astype(np.uint8)
                )

                txn.put(
                    f'spd_{t:05d}'.encode(),
                    np.ascontiguousarray(self.spds[t]).astype(np.float32)
                )

                # High-level command
                txn.put(
                    f'cmd_{t:05d}'.encode(),
                    np.ascontiguousarray(self.cmds[t]).astype(np.uint8)
                )

                # Target waypoint
                txn.put(
                    f'nxp_{t:05d}'.encode(),
                    np.ascontiguousarray(self.nxps[t]).astype(np.float32)
                )

                # Red/vehicle/walkers hazard
                txn.put(
                    f'bra_{t:05d}'.encode(),
                    np.ascontiguousarray(self.bras[t]).astype(np.uint8)
                )

        self.vizs.clear()
        self.rgbs.clear()
        self.tel_rgbs.clear()
        self.sems.clear()
        self.tel_sems.clear()
        self.lids.clear()
        self.maps.clear()

        self.gnsses.clear()
        self.lin_accs.clear()
        self.ang_vels.clear()

        self.ids.clear()
        self.locs.clear()
        self.oris.clear()
        self.bbox.clear()
        self.type.clear()
        self.spds.clear()
        self.cmds.clear()
        self.nxps.clear()
        self.bras.clear()

        lmdb_env.close()

    def sensors(self):
        sensors = [
            {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.map', 'id': 'LBL'},
            {'type': 'sensor.pretty_map', 'id': 'BEV'},
            {'type': 'sensor.objects', 'id': 'OBJ'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0., 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.other.imu',  'x': 0., 'y': 0., 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,'sensor_tick': 0.05, 'id': 'IMU'},
        ]

        # Add LiDAR
        sensors.append({
            'type': 'sensor.lidar.ray_cast', 'x': 0.0, 'y': 0.0, 'z': self.camera_z, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 
            'id': 'LIDAR'
        })

        # Add cameras
        for i, yaw in enumerate(self.camera_yaws):
            sensors.append({'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0.0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 256, 'height': 288, 'fov': 64, 'id': f'RGB_{i}'})
            sensors.append({'type': 'sensor.camera.semantic_segmentation', 'x': self.camera_x, 'y': 0.0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 256, 'height': 288, 'fov': 64, 'id': f'SEG_{i}'})

        sensors.append({'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0.0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 480, 'height': 288, 'fov': 40, 'id': 'TEL_RGB'})
        sensors.append({'type': 'sensor.camera.semantic_segmentation', 'x': self.camera_x, 'y': 0.0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 480, 'height': 288, 'fov': 40, 'id': 'TEL_SEG'})

        return sensors


    def run_step(self, input_data, timestamp):

        rgbs = []
        sems = []

        _, lids = input_data.get('LIDAR')

        for i in range(len(self.camera_yaws)):

            _, rgb = input_data.get(f'RGB_{i}')
            _, sem = input_data.get(f'SEG_{i}')

            rgbs.append(rgb[...,:3])
            sems.append(sem[...,2])

        _, lbl = input_data.get('LBL')
        _, bev = input_data.get('BEV')
        _, col = input_data.get('COLLISION')
        _, obj = input_data.get('OBJ')
        _, gps = input_data.get('GPS')
        _, imu = input_data.get('IMU')
        
        _, tel_rgb = input_data.get('TEL_RGB')
        _, tel_sem = input_data.get('TEL_SEG')

        tel_rgb = tel_rgb[...,:3]
        tel_sem = tel_sem[...,2]

        if self.waypointer is None:

            self.waypointer = Waypointer(
                self._global_plan, gps, pop_lane_change=False
            )

            self.planner = RoutePlanner(self._global_plan)

            # privileged agent runner
            self._autopilot = Autopilot(CarlaDataProvider.get_ego_vehicle(), map_utils)
            coarse_route = CarlaDataProvider.get_ego_vehicle_coarse_route()
            if coarse_route is None:
                self._autopilot.set_destination_waypoints(CarlaDataProvider.get_ego_vehicle_route_waypoint())
            else:
                self._autopilot.set_destination_coarse(coarse_route)

            self.town_name = CarlaDataProvider.get_world().get_map().name

        _, _, cmd = self.waypointer.tick(gps)
        wx, wy = self.planner.run_step(gps)

        # Transform to ego-coordinates
        wx, wy = _rotate(wx, wy, -imu[-1]+np.pi/2)

        yaw = obj.get('ori')[0]
        spd = obj.get('spd')[0]
        loc = obj.get('loc')[0]

        

        delta_locs, delta_yaws, next_spds = BellmanUpdater.compute_table(yaw/180*math.pi)

        # Convert lbl to rew maps
        lbl_copy = lbl.copy()
        waypoint_rews, stop_rews, brak_rews, free = BellmanUpdater.get_reward(lbl_copy, [0,0], ref_yaw=yaw/180*math.pi)

        # If it is idle, make it LANE_FOLLOW
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value

        if len(self.vizs) > self.num_per_flush:
            self.flush_data()

        rgb = np.concatenate(rgbs[1:-1]+[tel_rgb], axis=1)
        lidar = lids if self.prev_lidar is None else np.concatenate([lids, self.prev_lidar])

        if col:
            self.flush_data()
            raise Exception('Collector has collided!! Heading out :P')

        if spd < STOP_THRESH:
            self.stop_count += 1
        else:
            self.stop_count = 0

        self.prev_lidar = lids

        self.num_frames += 1

        self._autopilot.update_information(CarlaDataProvider.get_world())
        control = self._autopilot.run_step()

        h, w = brak_rews.shape

        brak = round(float(brak_rews[h//2,w//2]/(brak_rews.max()+1e-7)))

        control.brake = max(control.brake, brak)
        if control.brake > 0:
            control.throttle = 0

        if self.stop_count > MAX_STUCK:
            self.flush_data()
            raise Exception('Collector is stuck.. Heading out :D')

        self.gnss.append([gps[0], gps[1], imu[-1], spd, control.steer])
        self.lin_acc.append(imu[0:3])
        self.ang_vel.append(imu[3:6])

        # Save data
        if self.num_frames % (self.num_repeat+1) == 0 and self.stop_count < MAX_STOP and not np.isnan([wx,wy]).any():
            self.rgbs.append(rgbs)
            self.tel_rgbs.append(tel_rgb)
            self.sems.append(sems)
            self.tel_sems.append(tel_sem)
            self.lids.append(lidar)
            self.maps.append(bev)

            self.gnsses.append(deepcopy(self.gnss))
            self.lin_accs.append(deepcopy(self.lin_acc))
            self.ang_vels.append(deepcopy(self.ang_vel))

            self.gnss.clear()
            self.lin_acc.clear()
            self.ang_vel.clear()

            self.ids.append(np.array(obj.get('id')))
            self.locs.append(np.array(obj.get('loc')))
            self.oris.append(np.array(obj.get('ori')))
            self.bbox.append(np.array(obj.get('bbox')))
            self.type.append(np.array(obj.get('type')))
            self.spds.append(np.array(obj.get('spd')))

            self.cmds.append(cmd_value)
            self.nxps.append(np.array([wx, wy]))
            self.bras.append(np.array([control.brake]))

        self.vizs.append(visualize_obs(rgb, 0, (control.steer, control.throttle, control.brake), spd, map=bev, tgt=(wx, wy), cmd=cmd.value, lidar=lidar))

        return control

def _random_string(length=10):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))


def _rotate(x, y, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    return R @ [x, y]

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
