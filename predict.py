import torch
import torch.nn as nn
import numpy as np
import pickle
import json
import os

device = torch.device("cpu")

# ===== PATH SETUP =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== LOAD LABEL MAP =====
label_path = os.path.join(BASE_DIR, "label_map.json")

if not os.path.exists(label_path):
    raise FileNotFoundError(f"label_map.json not found at {label_path}")

with open(label_path, "r") as f:
    label_map = json.load(f)

# ===== GRAPH =====
V = 17
EDGES = [
    (0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),
    (5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16)
]

def build_adj():
    A = np.zeros((3, V, V), dtype=np.float32)

    for i in range(V):
        A[0, i, i] = 1

    for a, b in EDGES:
        A[1, a, b] = A[1, b, a] = 1

    for k in range(3):
        s = A[k].sum(1, keepdims=True)
        s[s == 0] = 1
        A[k] /= s

    return torch.tensor(A)

A = build_adj()

# ===== MODEL =====
class GraphConv(nn.Module):
    def __init__(self, in_c, out_c, A):
        super().__init__()
        self.A = A
        self.K = A.shape[0]
        self.conv = nn.Conv2d(in_c, out_c * self.K, 1)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        B, C, T, V = x.shape
        x = self.conv(x).view(B, self.K, -1, T, V)
        out = torch.einsum('bkctv,kvw->bctw', x, self.A.to(x.device))
        return self.bn(out)

class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A):
        super().__init__()
        self.gcn = GraphConv(in_c, out_c, A)
        self.tcn = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, (9,1), padding=(4,0)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.4)
        )
        self.res = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        return torch.relu(self.tcn(self.gcn(x)) + self.res(x))

class STGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(6 * 17)

        self.layers = nn.ModuleList([
            STGCNBlock(6, 64, A),
            STGCNBlock(64, 128, A),
            STGCNBlock(128, 256, A),
        ])

        self.fc = nn.Linear(256, 60)

    def forward(self, x):
        B, C, T, V = x.shape

        x = x.permute(0,1,3,2).contiguous().view(B, C * V, T)
        x = self.data_bn(x)
        x = x.view(B, C, V, T).permute(0,1,3,2)

        for layer in self.layers:
            x = layer(x)

        x = x.mean(dim=[2,3])
        return self.fc(x)

# ===== LOAD MODEL =====
model = STGCN().to(device)

model_path = os.path.join(BASE_DIR, "best_model.pth")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ===== PREPROCESS =====
def preprocess(kp):
    kp = kp[:, :, :2].astype(np.float32)
    kp /= 1000.0

    hip = (kp[:,11:12] + kp[:,12:13]) / 2
    kp -= hip

    if kp.shape[0] < 50:
        pad = np.zeros((50 - kp.shape[0], 17, 2))
        kp = np.vstack([kp, pad])
    else:
        idx = np.linspace(0, kp.shape[0] - 1, 50).astype(int)
        kp = kp[idx]

    vel = np.zeros_like(kp)
    vel[1:] = kp[1:] - kp[:-1]

    acc = np.zeros_like(kp)
    acc[2:] = vel[2:] - vel[1:-1]

    x = np.concatenate([kp, vel, acc], axis=-1)
    x = x.transpose(2, 0, 1)

    return torch.tensor(x).unsqueeze(0).float()

# ===== PREDICT =====
def predict(file):
    data = pickle.load(file)

    kp = data['annotations'][0]['keypoint'][0]
    x = preprocess(kp).to(device)

    with torch.no_grad():
        out = model(x)
        prob = torch.softmax(out, dim=1)
        pred = torch.argmax(prob, dim=1).item()

    probs_dict = {
        label_map[str(i)]: prob[0][i].item()
        for i in range(len(prob[0]))
    }

    return label_map[str(pred)], prob[0][pred].item(), probs_dict

# ===== EXPLAIN =====
def explain_prediction(kp):
    kp = kp[:, :, :2]

    velocity = np.zeros_like(kp)
    velocity[1:] = kp[1:] - kp[:-1]

    joint_motion = np.mean(np.abs(velocity), axis=(0,2))

    top_joints = np.argsort(joint_motion)[-5:][::-1]

    joint_names = [
        "Nose","Neck","R-Shoulder","R-Elbow","R-Wrist",
        "L-Shoulder","L-Elbow","L-Wrist",
        "Mid-Hip","R-Hip","R-Knee","R-Ankle",
        "L-Hip","L-Knee","L-Ankle","R-Eye","L-Eye"
    ]

    return [joint_names[i] for i in top_joints]