import os
import math
import yaml
import numpy as np
import cv2
import torch
import carla
import random
import string

from torch.nn import functional as F

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from models.lidar import LiDARModel
from models.bev_planner import BEVPlanner
from models.uniplanner import UniPlanner
from models.rgb import RGBSegmentationModel, RGBBrakePredictionModel
from pid import PIDController
from point_painting import CoordConverter, point_painting
from planner import RoutePlanner
from waypointer import Waypointer

def get_entry_point():
    return 'LAVAgent'

FPS = 20.
PIXELS_PER_METER = 4

CAMERA_YAWS = [-60,0,60]

class LAVAgent(AutonomousAgent):

    def sensors(self):
        sensors = [
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0., 'z': self.camera_z, 'id': 'GPS'},
            {'type': 'sensor.other.imu',  'x': 0., 'y': 0., 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,'sensor_tick': 0.05, 'id': 'IMU'},
            
        ]

        # Add LiDAR
        sensors.append({
            'type': 'sensor.lidar.ray_cast', 'x': 0.0, 'y': 0.0, 'z': self.camera_z, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0, 
            'id': 'LIDAR'
        })

        # Add cameras
        for i, yaw in enumerate(CAMERA_YAWS):
            sensors.append({'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0.0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': yaw,
            'width': 256, 'height': 288, 'fov': 64, 'id': f'RGB_{i}'})

        sensors.append({'type': 'sensor.camera.rgb', 'x': self.camera_x, 'y': 0.0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            'width': 480, 'height': 288, 'fov': 40, 'id': 'TEL_RGB'})

        return sensors

    def setup(self, path_to_conf_file):

        self.track = Track.SENSORS

        with open(path_to_conf_file, 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            setattr(self, key, value)

        self.device = torch.device('cuda')

        self.lidar_model = LiDARModel(
            num_input=len(self.seg_channels)+9 if self.point_painting else 9,
            backbone=self.backbone,
            num_features=self.num_features,
            min_x=self.min_x, max_x=self.max_x,
            min_y=self.min_y, max_y=self.max_y,
            pixels_per_meter=self.pixels_per_meter,
        ).to(self.device)

        bev_planner = BEVPlanner(
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size,
            feature_x_jitter=self.feature_x_jitter,
            feature_angle_jitter=self.feature_angle_jitter,
            x_offset=0, y_offset=1+self.min_x/((self.max_x-self.min_x)/2),
            num_cmds=self.num_cmds,
            num_plan=self.num_plan,
            num_plan_iter=self.num_plan_iter,
        ).to(self.device)
        bev_planner.load_state_dict(torch.load(self.bev_model_dir))
        bev_planner.eval()

        self.uniplanner = UniPlanner(
            bev_planner,
            pixels_per_meter=self.pixels_per_meter,
            crop_size=self.crop_size,
            feature_x_jitter=self.feature_x_jitter,
            feature_angle_jitter=self.feature_angle_jitter,
            x_offset=0, y_offset=1+self.min_x/((self.max_x-self.min_x)/2),
            num_cmds=self.num_cmds,
            num_plan=self.num_plan,
            num_input_feature=self.num_features[-1]*6,
            num_plan_iter=self.num_plan_iter,
        ).to(self.device)

        self.bra_model = RGBBrakePredictionModel([4,10,18]).to(self.device)
        self.seg_model = RGBSegmentationModel(self.seg_channels).to(self.device)

        self.lidar_model.eval()
        self.uniplanner.eval()
        self.bra_model.eval()
        self.seg_model.eval()

        self.coord_converters = [CoordConverter(
            cam_yaw, lidar_xyz=[0,0,self.camera_z], cam_xyz=[self.camera_x,0,self.camera_z],
            rgb_h=288, rgb_w=256, fov=64
        ) for cam_yaw in CAMERA_YAWS]

        # Load the models
        self.lidar_model.load_state_dict(torch.load(self.lidar_model_dir))
        self.uniplanner.load_state_dict(torch.load(self.uniplanner_dir))
        self.bra_model.load_state_dict(torch.load(self.bra_model_dir))
        self.seg_model.load_state_dict(torch.load(self.seg_model_dir))

        self.waypointer = None
        self.planner    = None
        self.prev_lidar = None

        self.turn_controller = PIDController(K_P=self.turn_KP, K_I=self.turn_KI, K_D=self.turn_KD, n=self.turn_n)
        self.speed_controller = PIDController(K_P=self.speed_KP, K_I=self.speed_KI, K_D=self.speed_KD, n=self.speed_n)
        
        self.num_frames = 0


    def destroy(self):

        self.waypointer = None
        self.planner    = None
        self.prev_lidar = None
        self.coord_converters = None
        self.turn_controller = None
        self.speed_controller = None

        self.num_frames = 0


        del self.lidar_model
        del self.uniplanner
        del self.bra_model
        del self.seg_model

        torch.cuda.empty_cache()

    @torch.no_grad()
    def run_step(self, input_data, timestamp):

        self.num_frames += 1

        _, lidar = input_data.get('LIDAR')
        _, gps   = input_data.get('GPS')
        _, imu   = input_data.get('IMU')
        _, ego   = input_data.get('EGO')
        spd      = ego.get('speed')

        if self.num_frames <= 1:
            self.prev_lidar = lidar
            return carla.VehicleControl()

        rgbs = []

        for i in range(len(CAMERA_YAWS)):
            _, rgb = input_data.get(f'RGB_{i}')
            rgbs.append(rgb[...,:3][...,::-1])

        _, tel_rgb = input_data.get('TEL_RGB')
        tel_rgb = tel_rgb[...,:3][...,::-1].copy()
        tel_rgb = tel_rgb[:-self.crop_tel_bottom]

        rgb = np.concatenate(rgbs, axis=1)
        all_rgb = np.stack(rgbs, axis=0)

        if self.waypointer is None:

            self.waypointer = Waypointer(
                self._global_plan, gps
            )

            self.planner = RoutePlanner(self._global_plan)

        _, _, cmd = self.waypointer.tick(gps)
        wx, wy = self.planner.run_step(gps)

        cmd_value = cmd.value - 1
        cmd_value = 3 if cmd_value < 0 else cmd_value

        # Transform to ego-coordinates
        wx, wy = _rotate(wx, wy, -imu[-1]+np.pi/2)

        # Proccess LiDAR
        fused_lidar = np.concatenate([lidar, self.prev_lidar])
        self.prev_lidar = lidar

        fused_lidar = self.preprocess(fused_lidar)

        # Paint lidar
        rgbs       = torch.tensor(rgb[None]).permute(0,3,1,2).float().to(self.device)
        all_rgbs   = torch.tensor(all_rgb).permute(0,3,1,2).float().to(self.device)
        tel_rgbs= torch.tensor(tel_rgb[None]).permute(0,3,1,2).float().to(self.device)

        pred_bra = self.bra_model(rgbs, tel_rgbs)
        pred_sem = to_numpy(torch.softmax(self.seg_model(all_rgbs), dim=1))

        pred_sem = pred_sem[:,1:] * (1-pred_sem[:,:1])
        painted_lidar = point_painting(fused_lidar, pred_sem, self.coord_converters)


        lidar_points = torch.tensor(np.concatenate([fused_lidar, painted_lidar], axis=-1),dtype=torch.float32).to(self.device)

        nxps       = torch.tensor([-wx,-wy]).float().to(self.device)

        features,      \
        pred_heatmaps, \
        pred_sizemaps, \
        pred_orimaps,  \
        pred_bev = self.lidar_model([lidar_points], [len(fused_lidar)])

        # Object detection
        det = self.det_inference(torch.sigmoid(pred_heatmaps[0]), pred_sizemaps[0], pred_orimaps[0])

        # Motion forecast & planning
        ego_plan_locs, ego_cast_locs, other_cast_locs, other_cast_cmds = self.uniplanner.infer(features[0], det[1], cmd_value, nxps)
        ego_plan_locs = to_numpy(ego_plan_locs)
        ego_cast_locs = to_numpy(ego_cast_locs)
        other_cast_locs = to_numpy(other_cast_locs)
        other_cast_cmds = to_numpy(other_cast_cmds)
        
        if self.no_refine:
            ego_plan_locs = ego_cast_locs

        if not np.isnan(ego_plan_locs).any():
            steer, throt, brake = self.pid_control(ego_plan_locs, spd, cmd_value)
        else:
            steer, throt, brake = 0, 0, 0

        if float(pred_bra) > 0.3:
            throt, brake = 0, 1
        elif self.plan_collide(ego_plan_locs, other_cast_locs, other_cast_cmds):
            throt, brake = 0, 1
        if spd * 3.6 > self.max_speed:
            throt = 0

        del rgbs
        del all_rgbs
        del tel_rgbs
        del lidar_points
        del nxps
        del features
        del pred_heatmaps
        del pred_sizemaps
        del pred_orimaps
        del pred_bev

        return carla.VehicleControl(steer=steer, throttle=throt, brake=brake)

    def plan_collide(self, ego_plan_locs, other_cast_locs, other_cast_cmds, dist_threshold=2.0):
        
        if self.no_forecast:
            return False
        
        # TODO: Do a proper occupancy map?
        for other_trajs, other_cmds in zip(other_cast_locs, other_cast_cmds):
            init_x, init_y = other_trajs[0,0]
            if init_y > 0.5*self.pixels_per_meter:
            # if init_y > 0:
                continue
            for other_traj, other_cmd in zip(other_trajs, other_cmds):
                if other_cmd < self.cmd_thresh:
                    continue
                dist = np.linalg.norm(other_traj-ego_plan_locs, axis=-1).min()
                if dist < dist_threshold:
                    return True

        return False


    def pid_control(self, waypoints, speed, cmd):

        waypoints = np.copy(waypoints) * self.pixels_per_meter
        waypoints[:,1] *= -1

        desired_speed = np.linalg.norm(waypoints[1:]-waypoints[:-1], axis=1).mean()

        aim = waypoints[self.aim_point]
        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        steer = self.turn_controller.step(angle)
        steer = np.clip(steer, -1.0, 1.0)

        brake = desired_speed < self.brake_speed * self.pixels_per_meter
        delta = np.clip(desired_speed - speed, 0.0, self.clip_delta)
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, self.max_throttle)
        throttle = throttle if not brake else 0.0

        return float(steer), float(throttle), float(brake)


    @torch.no_grad()
    def det_inference(self, heatmaps, sizemaps, orimaps, **kwargs):

        dets = []
        for i, c in enumerate(heatmaps):
            det = []
            for s, x, y in extract_peak(c, **kwargs):
                w, h = float(sizemaps[0,y,x]),float(sizemaps[1,y,x])
                cos, sin = float(orimaps[0,y,x]), float(orimaps[1,y,x])
                
                if i==1 and w < 0.1*self.pixels_per_meter or h < 0.2*self.pixels_per_meter:
                    continue
                
                # TODO: remove hardcode
                if np.linalg.norm([x-160,y-280]) <= 2:
                    continue

                det.append((x,y,w,h,cos,sin))
            dets.append(det)
        
        return dets

    def preprocess(self, lidar_xyzr, lidar_painted=None):

        idx = (lidar_xyzr[:,0] > -2.4)&(lidar_xyzr[:,0] < 0)&(lidar_xyzr[:,1]>-0.8)&(lidar_xyzr[:,1]<0.8)&(lidar_xyzr[:,2]>-1.5)&(lidar_xyzr[:,2]<-1)

        idx = np.argwhere(idx)

        if lidar_painted is None:
            return np.delete(lidar_xyzr, idx, axis=0)
        else:
            return np.delete(lidar_xyzr, idx, axis=0), np.delete(lidar_painted, idx, axis=0)


def _rotate(x, y, theta):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    return R @ [x, y]

def to_numpy(x):
    return x.detach().cpu().numpy()


def extract_peak(heatmap, max_pool_ks=7, min_score=0.2, max_det=20, break_tie=False):
    
    if break_tie:
        heatmap = heatmap + 1e-7*torch.randn(*heatmap.size(), device=heatmap.device)
    
    max_cls = F.max_pool2d(heatmap[None, None], kernel_size=max_pool_ks, padding=max_pool_ks//2, stride=1)[0, 0]
    possible_det = heatmap - (max_cls > heatmap).float() * 1e5
    if max_det > possible_det.numel():
        max_det = possible_det.numel()
    score, loc = torch.topk(possible_det.view(-1), max_det)

    return [(float(s), int(l) % heatmap.size(1), int(l) // heatmap.size(1))
            for s, l in zip(score.cpu(), loc.cpu()) if s > min_score]
