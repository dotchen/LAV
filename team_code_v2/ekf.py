import math
import numpy as np

class EKF:
    
    EARTH_RADIUS = 6371e3
    
    def __init__(self, cos0, lf, lr, gnss_noise=0.000005, compass_noise=1e-7, max_steer_angle=70, freq=20):
        
        xy_noise = EKF.EARTH_RADIUS * gnss_noise * math.pi/180. # In meters, approx.
        compass_noise = compass_noise * math.pi/180. # In rads

        self.Q = np.eye(3) * 1e-7
        self.R = np.array([
            [xy_noise**2,0,0],
            [0,xy_noise**2,0],
            [0,0,compass_noise**2]
        ])
        
        self.x = np.zeros((3,))
        self.P = np.zeros((3,3))
        self.F = np.eye(3)
        self.H = np.eye(3)
        
        self.max_steer_angle = max_steer_angle * math.pi/180.

        self.cos0 = cos0
        self.lr = lr
        self.L = lf + lr

        self.dt = 1./freq

    def init(self, lat, lon, compass):

        # Convert lat/lon to xy
        x_gps, y_gps = self.latlon_to_xy(lat, lon)


        self.x[0] = x_gps
        self.x[1] = y_gps
        self.x[2] = compass
        
        self.P = np.zeros((3,3))
        
    def step(self, spd, steer, lat, lon, compass):
        """
        Warning: ori = compass-math.pi/2
        """

        # Convert lat/lon to xy
        x_gps, y_gps = self.latlon_to_xy(lat, lon)
        
        # Predict state
        self.x = self.kbm_step(spd, steer)

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        # Innovation state
        y_kp = [x_gps, y_gps, compass] - self.x

        # Innovation covariance
        S_kp = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K_kp = self.P @ self.H.T @ np.linalg.inv(S_kp)

        # Update state estimate
        self.x = self.x + K_kp @ y_kp
                
        # Update covariance
        self.P = (np.eye(3) - K_kp@self.H) @ self.P
    
    def kbm_step(self, spd, steer):
        """
        kbm stands for kinematic bicycle model
        """
        
        # Pullout current state
        x_k, y_k, theta_k = self.x

        # Convert steer into rad
        wheel_steer = steer * self.max_steer_angle

        beta = np.arctan(self.lr * np.tan(wheel_steer)/self.L)

        x_kp = x_k + spd * math.cos(theta_k + beta) * self.dt
        y_kp = y_k + spd * math.sin(theta_k + beta) * self.dt
        theta_kp = theta_k + spd * np.tan(theta_k) * np.cos(beta) / self.L * self.dt
        
        return np.array([x_kp, y_kp, theta_kp])

        
    def latlon_to_xy(self, lat, lon):

        x = EKF.EARTH_RADIUS * lat * (math.pi / 180)
        y = EKF.EARTH_RADIUS * lon * (math.pi / 180) * math.cos(self.cos0)

        return x, y


def move_lidar_points(lidar, dloc, ori0, ori1):

    dloc = dloc @ [
        [ np.cos(ori0), -np.sin(ori0)],
        [ np.sin(ori0), np.cos(ori0)]
    ]

    ori = ori1 - ori0
    lidar = lidar @ [
        [np.cos(ori), np.sin(ori),0],
        [-np.sin(ori), np.cos(ori),0],
        [0,0,1],
    ]

    lidar[:,:2] += dloc
    
    return lidar
