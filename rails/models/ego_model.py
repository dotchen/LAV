import math
import torch
import torch.nn.functional as F
from torch import nn

class EgoModel(nn.Module):
    def __init__(self, dt=1./4):
        super().__init__()
        
        self.dt = dt

        # Kinematic bicycle model
        self.front_wb = nn.Parameter(torch.tensor(1.),requires_grad=True)
        self.rear_wb  = nn.Parameter(torch.tensor(1.),requires_grad=True)

        self.steer_gain  = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.brake_accel = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.throt_accel = nn.Sequential(
            nn.Linear(1,1,bias=False),
        )
        
    def forward(self, locs, yaws, spds, acts):
        
        '''
        only plannar
        '''
        
        steer = acts[...,0:1]
        throt = acts[...,1:2]
        brake = acts[...,2:3].byte()
        
        accel = torch.where(brake, self.brake_accel.expand(*brake.size()), self.throt_accel(throt))
        wheel = self.steer_gain * steer
        
        beta = torch.atan(self.rear_wb/(self.front_wb+self.rear_wb) * torch.tan(wheel))
        
        next_locs = locs + spds * torch.cat([torch.cos(yaws+beta), torch.sin(yaws+beta)],-1) * self.dt
        next_yaws = yaws + spds / self.rear_wb * torch.sin(beta) * self.dt
        next_spds = spds + accel * self.dt
        
        return next_locs, next_yaws, F.relu(next_spds)
