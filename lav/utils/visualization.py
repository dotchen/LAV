import enum
import cv2
import numpy as np


PIXELS_PER_METER = 4
PIXELS_AHEAD_VEHICLE = 120

BACKGROUND = [238, 238, 236]

PALETTE = [
    0, 0, 0,        # Unlabeled
    70, 70, 70,     # Building
    100, 40, 40,    # Fence
    55, 90, 80,     # Other
    220, 20, 60,    # Pedestrian
    153, 153, 153,  # Pole
    157, 234, 50,   # Roadline
    128, 64, 128,   # Road
    244, 35, 232,   # Side walk
    107, 142, 35,   # Vegetation
    0, 0, 142,      # Vehicles
    102, 102, 156,  # Wall
    220, 220, 0,    # TrafficSign
    70, 130, 180,   # Sky
    81, 0, 81,      # Ground
    150, 100, 100,  # Bridge
    230, 150, 140,  # Rail Track
    180, 165, 180,  # Guard Rails
    250, 170, 30,   # Traffic lights
    110, 190, 160,  # Static
    170, 120, 50,   # Dynamic
    45, 60, 150,    # Water
    145, 170, 100,  # Terrain
]
PALETTE += [0]*(256-len(PALETTE))

COLORS = [
        (102, 102, 102),
        (253, 253, 17),
        (255, 64, 64),
        (255, 165, 5),
        (255, 255, 255),
        (220, 20, 60),
        (255,0,0),
        (0,255,0),
        (0,0,0),
        (255,255,255),
        (0,255,128),
        (0,128,255),
        ]

SEM_COLORS = {
    4 : (220, 20, 60),
    5 : (153, 153, 153),
    6 : (157, 234, 50),
    7 : (128, 64, 128),
    8 : (244, 35, 232),
    10: (0, 0, 142),
    18: (220, 220, 0),
}

def visualize_big(rgb, yaw, control, speed, cmd=None, lbl=None, sem=None, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)):
    """
    0 road
    1 stop signs
    2 pedestrian
    3-5 lane
    6-8 red light
    9-11 vehicle
    """
    canvas = np.array(rgb[...,::-1])
    if lbl is not None:
        ori_x, ori_y = np.cos(yaw), np.sin(yaw)
        H = canvas.shape[0]
        lbl = visualize_birdview_big(lbl)
        h, w = lbl.shape[:2]
        # cv2.arrowedLine(lbl, (w//2,h//2), (w//2+int(ori_x*50),h//2+int(ori_y*50)), (255,128,0), 15)
        canvas = np.concatenate([canvas, cv2.resize(lbl, (H,H))], axis=1)

    if sem is not None:
        sem_viz = visualize_semantic(sem)
        canvas = np.concatenate([sem_viz, canvas], axis=1)

    return canvas

def visualize_obs(rgb, yaw, control, speed, cmd=None, red=None, lbl=None, tgt=None, map=None, sem=None, lidar=None, text_args=(cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)):
    """
    0 road
    1 lane
    2 stop signs
    3 red light
    4 vehicle
    5 pedestrian
    6-11 waypoints
    """
    canvas = np.array(rgb[...,::-1])
    if lbl is not None:
        ori_x, ori_y = np.cos(yaw), np.sin(yaw)
        H = canvas.shape[0]
        lbl = visualize_birdview(lbl, num_channels=12)
        h, w = lbl.shape[:2]
        cv2.arrowedLine(lbl, (w//2,h//2), (w//2+int(ori_x*10),h//2+int(ori_y*10)), (255,128,0), 3)
        canvas = np.concatenate([canvas, cv2.resize(lbl, (H,H))], axis=1)
    
    if map is not None:
        map = visualize_birdview_big(map)
        H, W = canvas.shape[:2]
        if tgt is not None:
            wx, wy = tgt
            h, w = map.shape[:2]
            # px, py = int(w/2 + wx * PIXELS_PER_METER), int(h + wy * PIXELS_PER_METER - PIXELS_AHEAD_VEHICLE)
            px, py = int(w/2 - wx * PIXELS_PER_METER), int(h/2 - wy * PIXELS_PER_METER + PIXELS_AHEAD_VEHICLE)
            px, py = np.clip([px, py], [0,0], [w-1,h-1])
            cv2.circle(map, (px, py), 2, (0,0,0), -1)
            # cv2.putText(canvas, '{:.2f},{:.2f}'.format(wx,wy),(4,50), *text_args)
        canvas = np.concatenate([canvas, cv2.resize(map, (H,H))], axis=1)


    if sem is not None:
        sem_viz = visualize_semantic(sem)
        canvas = np.concatenate([sem_viz, canvas], axis=1)
    
    if lidar is not None:
        lidar_viz = lidar_to_bev(lidar).astype(np.uint8)
        lidar_viz = cv2.cvtColor(lidar_viz,cv2.COLOR_GRAY2RGB)
        canvas = np.concatenate([canvas, cv2.resize(lidar_viz.astype(np.uint8), (canvas.shape[0], canvas.shape[0]))], axis=1)

    cv2.putText(canvas, f'speed: {speed:.3f}m/s', (4, 10), *text_args)
    cv2.putText(
        canvas, 
        f'steer: {control[0]:.3f} throttle: {control[1]:.3f} brake: {control[2]:.3f}',
        (4, 20), *text_args
    )
    if cmd is not None:
        cv2.putText(canvas, 'cmd: {}'.format({1:'left',2:'right',3:'straight',4:'follow',5:'change left',6:'change right'}.get(cmd)), (4, 30), *text_args)

    if red is not None:
        cv2.putText(canvas, 'red: {}'.format(bool(red)), (4,40), *text_args)

    return canvas


def visualize_birdview(birdview, no_show=[9], num_channels=12):
    
    h, w = birdview.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[...] = BACKGROUND

    for i in range(num_channels):
        if i in no_show:
            continue
        canvas[birdview[:,:,i] > 0] = COLORS[i]
    
    return canvas

def visualize_semantic(sem, labels=[4,6,7,10,18]):
    canvas = np.zeros(sem.shape+(3,), dtype=np.uint8)
    for label in labels:
        canvas[sem==label] = SEM_COLORS[label]

    return canvas

def visualize_semantic_processed(sem, labels=[4,6,7,10,18]):
    canvas = np.zeros(sem.shape+(3,), dtype=np.uint8)
    for i,label in enumerate(labels):
        canvas[sem==i+1] = SEM_COLORS[label]

    return canvas

def lidar_to_bev(lidar, min_x=-10,max_x=70,min_y=-40,max_y=40, pixels_per_meter=4, hist_max_per_pixel=10):
    xbins = np.linspace(
        min_x, max_x+1,
        (max_x - min_x) * pixels_per_meter + 1,
    )
    ybins = np.linspace(
        min_y, max_y+1,
        (max_y - min_y) * pixels_per_meter + 1,
    )
    # Compute histogram of x and y coordinates of points.
    hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
    # Clip histogram
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    # Normalize histogram by the maximum number of points in a bin we care about.
    overhead_splat = hist / hist_max_per_pixel * 255.
    # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground.
    return overhead_splat[::-1,:]

def filter_sem(sem, labels=[4,6,7,8,10]):
    resem = np.zeros_like(sem)
    for i, label in enumerate(labels):
        resem[sem==label] = i+1
    
    return resem