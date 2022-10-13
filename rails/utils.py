import numpy as np

def to_numpy(x):
    return x.detach().cpu().numpy()