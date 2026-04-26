import numpy as np
import torch

MAX_FRAMES = 50

def preprocess_input(kp):
    kp = kp / 1000.0
    kp = kp - kp[:, 0:1, :]
    kp = np.clip(kp, -1, 1)

    if kp.shape[0] < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - kp.shape[0], 17, 2))
        kp = np.vstack([kp, pad])
    else:
        kp = kp[:MAX_FRAMES]

    kp = kp.transpose(2, 0, 1)
    return torch.tensor(kp).unsqueeze(0).float()