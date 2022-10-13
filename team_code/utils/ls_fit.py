import math
import numpy as np

def ls_polynomial(points, deg=2):
    '''
    points are {(x,y)} of shape (B,2)
    '''
    
    return np.polyfit(points[:,0], points[:,1], deg)
    
def ls_circle(points):
    '''
    Input: Nx2 points
    Output: cx, cy, r
    '''
    xs = points[:,0]
    ys = points[:,1]

    us = xs - np.mean(xs)
    vs = ys - np.mean(ys)

    Suu = np.sum(us**2)
    Suv = np.sum(us*vs)
    Svv = np.sum(vs**2)
    Suuu = np.sum(us**3)
    Suvv = np.sum(us*vs*vs)
    Svvv = np.sum(vs**3)
    Svuu = np.sum(vs*us*us)

    A = np.array([
        [Suu, Suv],
        [Suv, Svv]
    ])

    b = np.array([1/2.*Suuu+1/2.*Suvv, 1/2.*Svvv+1/2.*Svuu])

    cx, cy = np.linalg.solve(A, b)
    r = np.sqrt(cx*cx+cy*cy+(Suu+Svv)/len(xs))

    cx += np.mean(xs)
    cy += np.mean(ys)

    return np.array([cx, cy]), r

def project_point_to_circle(point, c, r):
    direction = point - c
    closest = c + (direction / np.linalg.norm(direction)) * r

    return closest

def signed_angle(u, v):
    theta = math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    if np.cross(u, v)[2] < 0:
        theta *= -1.0

    return theta
