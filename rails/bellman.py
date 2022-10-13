import math
import numpy as np
import torch
import torch.nn.functional as F
import itertools

from utils import _numpy

PIXELS_PER_METER = 3
FPS = 20

class BellmanUpdater(object):

    _ego_model = None
    _ego_table = None

    _speeds = None
    _orient = None
    
    @staticmethod
    def setup(config, ego_model, device=torch.device('cuda')):
        
        # Discretization ranges
        BellmanUpdater._max_steers = config['max_steers']
        BellmanUpdater._max_throts = config['max_throts']
        BellmanUpdater._min_speeds = config['min_speeds'] # m/s
        BellmanUpdater._max_speeds = config['max_speeds'] # m/s
        BellmanUpdater._min_orient = math.pi/180*config['min_orient'] # degree to radian
        BellmanUpdater._max_orient = math.pi/180*config['max_orient'] # radian
        
        # Discretization precision
        BellmanUpdater._num_steers = config['num_steers']
        BellmanUpdater._num_throts = config['num_throts']
        BellmanUpdater._num_speeds = config['num_speeds']
        BellmanUpdater._num_orient = config['num_orient']
        
        # Target speeds
        BellmanUpdater._tgt_speeds = torch.tensor([
            config['turn_tgt_speeds'],
            config['turn_tgt_speeds'],
            config['lane_tgt_speeds'],
            config['lane_tgt_speeds'],
            config['lane_tgt_speeds'],
            config['lane_tgt_speeds']
        ]).float().to(device)
        
        BellmanUpdater._turn_tgt_speeds = config['turn_tgt_speeds']
        BellmanUpdater._lane_tgt_speeds = config['lane_tgt_speeds']

        BellmanUpdater._ego_model = ego_model
        BellmanUpdater._device = device

        # State bins
        _speeds = np.linspace(BellmanUpdater._min_speeds, BellmanUpdater._max_speeds, BellmanUpdater._num_speeds+1)
        _orient = np.linspace(BellmanUpdater._min_orient, BellmanUpdater._max_orient, BellmanUpdater._num_orient+1)

        BellmanUpdater._speeds = torch.tensor([(a+b)/2 for a, b in zip(_speeds[:-1], _speeds[1:])]).float().to(BellmanUpdater._device)
        BellmanUpdater._orient = torch.tensor([(a+b)/2 for a, b in zip(_orient[:-1], _orient[1:])]).float().to(BellmanUpdater._device)

        # Action bins
        _steers = np.linspace(-BellmanUpdater._max_steers,BellmanUpdater._max_steers,BellmanUpdater._num_steers)
        _throts = np.linspace(0,BellmanUpdater._max_throts,BellmanUpdater._num_throts)

        _steers = np.append(np.tile(_steers, BellmanUpdater._num_throts), 0)
        _throts = np.append(np.repeat(_throts, BellmanUpdater._num_steers), 0)
        _brakes = np.append(np.zeros(len(_steers)-1), 1)
    
        BellmanUpdater._actions = torch.tensor(np.stack([_steers,_throts,_brakes], axis=-1)).float().to(BellmanUpdater._device)
    
        # Brak reward: used at peds/vehicles/red lights
        BellmanUpdater._brak_rews = torch.zeros(BellmanUpdater._num_steers*BellmanUpdater._num_throts+1).float().to(BellmanUpdater._device)
        BellmanUpdater._brak_rews[-1] = 1
        
        # Thro reward: used at zero-speed at stop signs
        BellmanUpdater._thro_rews = torch.zeros(BellmanUpdater._num_steers*BellmanUpdater._num_throts+1).float().to(BellmanUpdater._device)
        BellmanUpdater._thro_rews[-BellmanUpdater._num_steers//2-1] = 1
        
        BellmanUpdater._brak_rew = config['brak_rew']
        BellmanUpdater._thro_rew = config['thro_rew']
        BellmanUpdater._stop_rew = config['stop_rew']
        
        # Shift grids
        grid_x = torch.linspace(-1,1,96).unsqueeze(0).repeat(96,1).unsqueeze(-1)
        grid_y = torch.linspace(-1,1,96).unsqueeze(0).repeat(96,1).unsqueeze(-1)
        BellmanUpdater._grids = torch.cat([grid_x,grid_y.transpose(1,0)],dim=2).to(device)

        # dt
        BellmanUpdater._dt = 1./FPS*(1+config['num_repeat'])

    @staticmethod
    def compute_table(ref_yaw, device=torch.device('cuda')):

        ref_yaw = torch.tensor(ref_yaw).float().to(device)

        next_locs = []
        next_yaws = []
        next_spds = []

        locs = torch.zeros((ref_yaw.shape)+(2,)).float().to(device)

        action = BellmanUpdater._actions.expand(len(BellmanUpdater._speeds),len(BellmanUpdater._orient),-1,-1).permute(2,0,1,3)
        speed  = BellmanUpdater._speeds.expand(len(BellmanUpdater._actions),len(BellmanUpdater._orient),-1).permute(0,2,1)[...,None]
        orient = BellmanUpdater._orient.expand(len(BellmanUpdater._actions),len(BellmanUpdater._speeds),-1)[...,None]

        delta_locs, next_yaws, next_spds = BellmanUpdater._ego_model(locs, orient+ref_yaw, speed, action)

        # Normalize to grid's units
        # Note: speed is orientation-agnostic
        delta_locs = delta_locs*PIXELS_PER_METER
        delta_yaws = torch.atan2(torch.sin(next_yaws-ref_yaw-BellmanUpdater._orient[0]), torch.cos(next_yaws-ref_yaw-BellmanUpdater._orient[0]))
        delta_yaws = delta_yaws[...,0,0]/(BellmanUpdater._max_orient-BellmanUpdater._min_orient)*BellmanUpdater._num_orient

        next_spds = (next_spds[...,0,0]-BellmanUpdater._min_speeds)/(BellmanUpdater._max_speeds-BellmanUpdater._min_speeds)*BellmanUpdater._num_speeds

        return delta_locs, delta_yaws, next_spds


    @staticmethod
    def get_reward(lbl, loc_offsets, ref_yaw=0):

        road, lane, stop, red, vehicle, pedestrian = map(lambda x: x[...,0], np.split(lbl[...,:6],6,axis=-1))
        waypoints = lbl[...,6:]

        free = torch.tensor((road>0)&(vehicle==0)&(pedestrian==0),dtype=torch.float32).expand(BellmanUpdater._num_speeds,BellmanUpdater._num_orient,-1,-1).to(BellmanUpdater._device)

        wpt_tgt_orient = BellmanUpdater._to_dense_orient(waypoints, ref_yaw)
        red_tgt_orient = BellmanUpdater._to_dense_orient(red, ref_yaw)
        stp_tgt_orient = BellmanUpdater._to_dense_orient(stop, ref_yaw)

        # Preprocess lane change reward maps
        # LEFT,LEFTCHANGE,RIGHTCHANGE
        offset_coeffs = {0:False,1:False,4:True,5:True}
        for cmd in [4,5]:
            waypoints[...,cmd] = BellmanUpdater._lane_change_filter(waypoints[...,cmd]>0, wpt_tgt_orient[...,cmd], ref_yaw, loc_offsets, offset_coeffs.get(cmd, False))

        batch_shape = lbl.shape[:-3]

        # Waypoint reward
        brak_rews = torch.tensor((vehicle>0)|(pedestrian>0)|(red>0), dtype=torch.float32).to(BellmanUpdater._device)
        # brak_rews = torch.tensor(red>0, dtype=torch.float32).to(BellmanUpdater._device)
        brak_rews *= BellmanUpdater._brak_rew

        # x2 hack to still reward zero-speed
        stop_rews = BellmanUpdater.to_dense(torch.tensor((vehicle>0)|(pedestrian>0),dtype=torch.float32).to(BellmanUpdater._device), batch_shape, target_speed=0, target_orient='all')
        stop_rews += BellmanUpdater.to_dense(torch.tensor(red>0,dtype=torch.float32).to(BellmanUpdater._device), batch_shape, target_speed=0, target_orient=red_tgt_orient)
        stop_rews *= BellmanUpdater._stop_rew

        # DEBUG
        # stop_rews = torch.zeros_like(stop_rews)
        # END DEBUG

        waypoint_rews = BellmanUpdater.to_dense(torch.tensor(waypoints>0,dtype=torch.float32).to(BellmanUpdater._device), batch_shape, target_speed=BellmanUpdater._tgt_speeds, target_orient=wpt_tgt_orient)
        waypoint_rews -= BellmanUpdater._stop_rew * BellmanUpdater.to_dense(torch.tensor((vehicle>0)|(pedestrian>0),dtype=torch.float32).to(BellmanUpdater._device), batch_shape, target_speed='all', target_orient='all')[...,None]
        # waypoint_rews -= BellmanUpdater.to_dense(torch.tensor(road==0,dtype=torch.float32).to(BellmanUpdater._device), batch_shape, target_speed='all', target_orient='all')[...,None]

        # Lane change hack
        waypoint_rews[...,4] += waypoint_rews[...,0]*4e-2
        waypoint_rews[...,5] += waypoint_rews[...,1]*4e-2

        return waypoint_rews, stop_rews, brak_rews, free


    @staticmethod
    def to_dense(x, batch_shape, target_speed='all', target_orient='all'):
        """
        (...) -> (Ns, No, ...)
        """
        orient = torch.zeros(BellmanUpdater._num_orient).float().to(BellmanUpdater._orient.device).expand(*batch_shape, -1)
        if target_orient == 'all':
            orient = torch.ones_like(orient)[...,None,None]
        else:
            orient = BellmanUpdater._lerp_grids(
                BellmanUpdater._min_orient,
                BellmanUpdater._max_orient,
                BellmanUpdater._num_orient,
                target_orient*torch.ones_like(orient[...,0])
            )

        speeds = torch.zeros(BellmanUpdater._num_speeds).float().to(BellmanUpdater._speeds.device).expand(*batch_shape, -1)

        if target_speed == 'all':
            speeds += torch.ones_like(speeds)
            speeds = speeds[(...,)+(None,)*len(orient.shape[len(batch_shape):])]
        elif isinstance(target_speed, (float, int)):
            speeds += BellmanUpdater._lerp_grids(
                BellmanUpdater._min_speeds,
                BellmanUpdater._max_speeds,
                BellmanUpdater._num_speeds,
                target_speed*torch.ones_like(speeds[...,0])
            )
            speeds = speeds[(...,)+(None,)*len(orient.shape[len(batch_shape):])]
        else:
            speeds = speeds[:,None] + BellmanUpdater._lerp_grids(
                BellmanUpdater._min_speeds,
                BellmanUpdater._max_speeds,
                BellmanUpdater._num_speeds,
                target_speed
            )
            speeds = speeds[...,None,None,None,:]

        return x*orient*speeds

    @staticmethod
    def _lane_change_filter(mask, orient, ref_yaw, offset, lane_change):
        
        x_offset, y_offset = offset
        
        peek_x, peek_y = 48 + x_offset*PIXELS_PER_METER, 48 + y_offset*PIXELS_PER_METER
        
        if lane_change:
            peek_x += np.cos(ref_yaw)*PIXELS_PER_METER*BellmanUpdater._lane_tgt_speeds*BellmanUpdater._dt
            peek_y += np.sin(ref_yaw)*PIXELS_PER_METER*BellmanUpdater._lane_tgt_speeds*BellmanUpdater._dt
        else:
            peek_x -= np.cos(ref_yaw)*PIXELS_PER_METER*BellmanUpdater._turn_tgt_speeds*BellmanUpdater._dt
            peek_y -= np.sin(ref_yaw)*PIXELS_PER_METER*BellmanUpdater._turn_tgt_speeds*BellmanUpdater._dt

        new_mask = dfs_search(mask, orient, ref_yaw, peek_x, peek_y)

        return new_mask

    @staticmethod
    def _to_dense_orient(orient, ref_yaw):
        orient = decode_orient(orient) - ref_yaw
        orient = np.arctan2(np.sin(orient), np.cos(orient))
        orient = torch.tensor(orient).float().to(BellmanUpdater._device)

        return orient

    @staticmethod
    def _lerp_grids(min_val, max_val, num_val, x):
        grids = torch.zeros_like(x).expand(num_val, *x.shape)
        v = (x - min_val)/(max_val-min_val)*(num_val-1)

        v0, v1 = torch.clamp(v.to(int), min=0, max=num_val-1), torch.clamp(v.to(int)+1, min=1, max=num_val)
        w = v1-v

        # out-of-bounds values are 0
        w0 = torch.where((x>=min_val)&(x<=max_val), w, torch.zeros_like(w))
        w1 = torch.where((x>=min_val)&(x<=max_val), 1-w, torch.zeros_like(w))

        v1 = torch.clamp(v1, max=num_val-1)

        grids = grids.scatter(0,v0[None],w0[None])
        grids = grids.scatter(0,v1[None],w1[None])

        return grids

    @staticmethod
    def _lerp(v, x):
        """
        x guaranteed non-negative
        """
        x0, x1 = x.floor().long(), x.ceil().long()
        w = (x1-x)[(..., ) + (None,)*(len(v.shape)-1)]
        w0 = torch.where((w>=0)&(w<v.shape[0]), w, torch.zeros_like(w))
        w1 = torch.where((w>=0)&(w<v.shape[0]), 1-w, torch.zeros_like(w))
        
        x0 = torch.clamp(x0,min=0,max=v.shape[0]-1)
        x1 = torch.clamp(x1,min=0,max=v.shape[0]-1)

        return v[x0] * w0 + v[x1] * w1

    @staticmethod
    def _batch_lerp(v, x, min_val=0, max_val=1):
        D = v.shape[-1]
        x = (x - min_val)/(max_val-min_val)*(D-1)
        x = x[(...,)+(None,)*len(v.shape[1:])]
        
        x0, x1 = torch.clamp(x.to(int), min=0), torch.clamp(x.to(int)+1, max=D-1)
        w = x - x0
        
        output = (1-w)*v.gather(-1, torch.ones_like(v[...,-1:]).long()*x0) + \
                  w *v.gather(-1, torch.ones_like(v[...,-1:]).long()*x1)
        
        return output.squeeze(-1)

    @staticmethod
    def _batch_shift_lerp(v, dx, dim=0):
        # return v
        dx0, dx1 = dx.floor().int(), dx.ceil().int()
        w = dx - dx0
        D = v.shape[dim]

        # dx0, dx1, w, D = int(dx), int(dx)+1, dx-int(dx), v.shape[dim]
        vs = v.view((np.prod([1]+list(v.shape[:dim])), D, -1))
        ws = w.flatten()[:,None,None]
        # print (D)
        Ds = torch.arange(D,device=v.device)[None,:,None]
        
        r = torch.zeros_like(vs)
        dx0 = dx0.view(-1,1).expand(vs.shape[-1],-1,1).permute(1,2,0)
        dx1 = dx1.view(-1,1).expand(vs.shape[-1],-1,1).permute(1,2,0)
        
        r = torch.where(
            (dx0+Ds>=0)&(dx1+Ds<D), 
            (1-ws)*vs.gather(1,torch.clamp(dx0+Ds,min=0,max=D-1))+\
            ws    *vs.gather(1,torch.clamp(dx1+Ds,min=0,max=D-1)),
            torch.zeros_like(vs))

        return r.view(v.shape)

    @staticmethod
    def get_action_all(delta_locs, delta_yaws, next_spds, waypoint_rews_all, brak_rews, stop_rews, frees, locs):
        actions = []
        for cmd in range(6):
            action = BellmanUpdater.get_action(delta_locs, delta_yaws, next_spds, waypoint_rews_all[...,cmd], brak_rews, stop_rews, frees, locs)
            actions.append(action)

        return torch.stack(actions, dim=-1)

    @staticmethod
    @torch.no_grad()
    def get_action(delta_locs, delta_yaws, next_spds, waypoint_rews, brak_rews, stop_rews, frees, locs, extract, visualize=None, dense_action_values=False):
        V = torch.zeros_like(stop_rews[0])
        cur_loc = locs[-1]

        # DEBUG
        # delta_locs = torch.zeros_like(delta_locs)
        # END DEBUG

        for t in range(len(waypoint_rews)-1,-1,-1):

            waypoint_rew = waypoint_rews[t]
            brak_rew = brak_rews[t]
            stop_rew = stop_rews[t]
            free = frees[t]
            
            Q = BellmanUpdater.compute_Q(delta_locs, delta_yaws, next_spds, V, waypoint_rew, stop_rew, free)
            V, _ = Q.max(dim=0)

            if t > 0:
                V = BellmanUpdater.shift(V, locs[t]-locs[t-1])
            else:
                # Make sure locs are always ref off t=0!
                # import pdb; pdb.set_trace()
                Q = BellmanUpdater.shift(Q, locs[t])
                Q = Q + brak_rew[None] * BellmanUpdater._brak_rews[:,None,None,None,None]
                # Q = Q + thro_rew[None] * BellmanUpdater._thro_rews[:,None,None,None,None]
                
        if visualize is not None:
            
            V_axes, Q_axis1, Q_axis2, rotate, buff = visualize
            
            # One last max
            # V, _ = Q.max(dim=0)
            
            import matplotlib.pyplot as plt
            import itertools 
            # f, axes = plt.subplots(BellmanUpdater._num_throts,BellmanUpdater._num_steers,figsize=(BellmanUpdater._num_steers*3,BellmanUpdater._num_throts*3))
            # for i, j in itertools.product(range(BellmanUpdater._num_throts), range(BellmanUpdater._num_steers)):
            #     axes[i,j].imshow(Q[i*BellmanUpdater._num_steers+j,0,BellmanUpdater._num_orient//2].detach().cpu().numpy())
            
            # f, axes = plt.subplots(BellmanUpdater._num_speeds,BellmanUpdater._num_orient,figsize=(BellmanUpdater._num_orient*2,BellmanUpdater._num_speeds*2))
            # for i, j in itertools.product(range(BellmanUpdater._num_speeds), range(BellmanUpdater._num_orient)):
            #     # axes[i,j].imshow(V[i,j].detach().cpu().numpy())
            #     axes[i,j].imshow(waypoint_rews[0,i,j].detach().cpu().numpy(), cmap=plt.get_cmap('gray'), vmin=0,vmax=2)
            
            # f, axes = plt.subplots(BellmanUpdater._num_speeds,BellmanUpdater._num_orient,figsize=(BellmanUpdater._num_orient*2,BellmanUpdater._num_speeds*2))
            # vmin, vmax = V.min(), V.max()+0.5
            vmin, vmax = 0, 2
            for i, j in itertools.product(range(BellmanUpdater._num_speeds), range(BellmanUpdater._num_orient)):
                V_axes[i][j].imshow(rotate(V[i,j].detach().cpu().numpy()), cmap=plt.get_cmap('jet'), vmin=vmin,vmax=vmax)
                V_axes[i][j].set_xticks([])
                V_axes[i][j].set_yticks([])
                V_axes[i][j].spines['top'].set_visible(False)
                V_axes[i][j].spines['right'].set_visible(False)
                V_axes[i][j].spines['bottom'].set_visible(False)
                V_axes[i][j].spines['left'].set_visible(False)
                # axes[i,j].imshow(waypoint_rews[0,i,j].detach().cpu().numpy())
            
            # import pdb; pdb.set_trace()
        
        
        # Extract action
        delta_locs, delta_yaws, spds = map(lambda x: x.to(Q.device), extract)
            
        Na, Ns, No, Nh, Nw = Q.size()
        
        qs, all_qs = [], []
        for delta_loc, delta_yaw, spd in zip(delta_locs, delta_yaws, spds):
            all_q = F.grid_sample(Q.view(1,-1,Nh,Nw), delta_loc[None,None,None]*PIXELS_PER_METER).view(-1,Na,Ns,No)
            all_q = BellmanUpdater._batch_lerp(all_q, delta_yaw[None], min_val=BellmanUpdater._min_orient, max_val=BellmanUpdater._max_orient)
            q = BellmanUpdater._batch_lerp(all_q, spd[None], min_val=BellmanUpdater._min_speeds, max_val=BellmanUpdater._max_speeds)
        
            qs.append(q)
            if dense_action_values:
                all_qs.append(Q)
            else:
                all_qs.append(all_q)
        
        if visualize is not None:
            qs[0][0,-1] -= 0.5
            max_q = float(max(qs[0][0]))
            Q_axis1.imshow(_numpy(qs[0][0,:27].reshape(3,9)), cmap=plt.get_cmap('jet'), vmin=vmin,vmax=vmax)
            # Do bar
            bar = Q_axis2.imshow(np.ones((1,12))*float(qs[0][0,-1]), cmap=plt.get_cmap('jet'), vmin=vmin,vmax=vmax)
            buff.append(bar)
            from matplotlib.patches import Rectangle
            if max_q == qs[0][0,-1]:
                rect = Rectangle((0-0.52, 0-0.49), 12, 1, linewidth=2, edgecolor='red', facecolor='none')
                # Add the patch to the Axes
                Q_axis2.add_patch(rect)
            else:
                
                y, x = torch.nonzero(qs[0][0,:27].reshape(3,9)==max_q)[0]
                x, y = float(x), float(y)
                rect = Rectangle((x-0.52, y-0.49), 1, 1, linewidth=2, edgecolor='red', facecolor='none')
                # Add the patch to the Axes
                Q_axis1.add_patch(rect)
        #     plt.show()

        return torch.cat(qs), torch.cat(all_qs)

    @staticmethod
    def compute_Q(delta_locs, delta_yaws, next_spds, V0, waypoint_rew, stop_rew, free):
        
        Q = BellmanUpdater._lerp(V0 + waypoint_rew + stop_rew, next_spds)
        Q = BellmanUpdater._batch_shift_lerp(Q, delta_yaws,        2)
        Q = BellmanUpdater._batch_shift_lerp(Q, delta_locs[...,1], 3)
        Q = BellmanUpdater._batch_shift_lerp(Q, delta_locs[...,0].expand(V0.shape[2],-1,-1,-1).permute(1,2,3,0), 4)
        Q = Q * free

        return Q

    @staticmethod
    def shift(V, offset):
        pixel_offset = offset * PIXELS_PER_METER/48
        Vs = V.view(1,-1,96,96)
        Vs = F.grid_sample(Vs, -pixel_offset+BellmanUpdater._grids[None])
        return Vs.view(*V.shape)

def decode_orient(lbl):
    """
    decode angle (1-255) to (0,2pi)
    """
    yaw = (lbl.astype(float)-1)/254*math.pi*2
    return yaw

def dfs_search(msks, yaws, ref_yaw, init_x, init_y, a_limit=math.radians(60)):

    if msks.sum()==0:
        return msks

    visited = np.zeros_like(msks)
    H, W = msks.shape
    hs, ws = np.nonzero(msks)

    # Compute init node
    min_idx = np.argmin(np.linalg.norm([hs-init_y,ws-init_x], axis=0))
    y, x = hs[min_idx], ws[min_idx]
    yaw = yaws[y, x]

    queue = [(x,y,yaw)]
    while len(queue) > 0:

        x, y, yaw = queue.pop()
        visited[y, x] = True

        for dx, dy in itertools.product([-1,0,1], [-1,0,1]):

            nx, ny = x + dx, y + dy
            if nx >= 0 and nx < W and ny >= 0 and ny < H and \
            msks[ny,nx] and \
            abs(angle_difference(yaws[ny,nx], yaw)) < a_limit and \
            not visited[ny,nx]:
                queue.append((nx, ny, yaws[ny,nx]))

    return visited


def angle_difference(a1, a2):
    return (a1 - a2 + math.pi)%(math.pi*2) - math.pi
