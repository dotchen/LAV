import numpy as np

def filter_sem(sem, labels=[4,6,7,8,10]):
    resem = np.zeros_like(sem)
    for i, label in enumerate(labels):
        resem[sem==label] = i+1
    
    return resem

def _numpy(x):
    return x.detach().cpu().numpy()
