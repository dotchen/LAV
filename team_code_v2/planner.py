import os
from collections import deque

import numpy as np
import math

class RoutePlanner(object):
    
    EARTH_RADIUS = 6371e3 # 6371km
    
    def __init__(self, global_plan, curr_threshold=20, next_threshold=75, debug=False):
        self.route = deque()
        self.curr_threshold = curr_threshold
        self.next_threshold = next_threshold

        # Convert lat,lon to x,y
        cos_0 = 0.
        for gnss, _ in global_plan:
            cos_0 += gnss['lat'] * (math.pi / 180)
        cos_0 = cos_0 / (len(global_plan))
        self.cos_0 = cos_0
        
        for node in global_plan:
            gnss, cmd = node

            x, y = self.latlon_to_xy(gnss['lat'], gnss['lon'])
            self.route.append((x, y))

        self.debug = debug

        self.current_idx = 0
        self.checkpoint = self.route[0]

    def run_step(self, gnss):

        x, y = self.latlon_to_xy(gnss[0], gnss[1])
        
        wx, wy = np.array(self.checkpoint)
        curr_distance = np.linalg.norm([wx-x, wy-y])

        for i, (wx, wy) in enumerate(self.route):
            
            distance = np.linalg.norm([wx-x, wy-y])
            
            if distance < self.next_threshold and i - self.current_idx==1 and curr_distance < self.curr_threshold:
                self.checkpoint = [wx, wy]
                self.current_idx += 1
                break
        
        return np.array(self.checkpoint) - [x,y]


    def latlon_to_xy(self, lat, lon):

        x = self.EARTH_RADIUS * lat * (math.pi / 180)
        y = self.EARTH_RADIUS * lon * (math.pi / 180) * math.cos(self.cos_0)

        return x, y
