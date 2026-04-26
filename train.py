import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle, random
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# ── CONFIG ─────────────────────────────────────────
NUM_CLASSES = 60
MAX_FRAMES  = 50
EPOCHS      = 80
BATCH_SIZE  = 32
LR          = 1e-3
PATIENCE    = 15
SEED        = 42
IN_CH = 6
V     = 17

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── GRAPH ─────────────────────────────────────────
EDGES = [(0,1),(0,2),(1,3),(2,4),(0,5),(0,6),(5,6),
         (5,7),(7,9),(6,8),(8,10),
         (5,11),(6,12),(11,12),
         (11,13),(13,15),(12,14),(14,16)]

def build_adj():
    A = np.zeros((3, V, V), dtype=np.float32)
    for i in range(V): A[0,i,i] = 1
    for a,b in EDGES:
        A[1,a,b] = A[1,b,a] = 1
    for k in range(3):
        s = A[k].sum(1, keepdims=True)
        s[s==0]=1
        A[k] /= s
    return torch.tensor(A)

A = build_adj()

# ── PREPROCESS ─────────────────────────────────────
def preprocess(kp):
    kp = kp[:, :, :2].astype(np.float32)
    kp /= 1000.0

    hip = (kp[:,11:12] + kp[:,12:13]) / 2
    kp -= hip

    if kp.shape[0] < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - kp.shape[0],17,2))
        kp = np.vstack([kp,pad])
    else:
        idx = np.linspace(0,kp.shape[0]-1,MAX_FRAMES).astype(int)
        kp = kp[idx]

    return kp

def augment(x):
    x += np.random.randn(*x.shape)*0.01
    if np.random.rand() < 0.5:
        x[:,:,0] *= -1
    return x

# ── DATASET ───────────────────────────────────────
class SkeletonDataset(Dataset):
    def __init__(self, samples, aug=None):
        self.samples = samples
        self.aug = aug

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        ann = self.samples[i]
        x = preprocess(ann['keypoint'][0])

        if self.aug: x = self.aug(x)

        vel = np.zeros_like(x)
        vel[1:] = x[1:] - x[:-1]

        acc = np.zeros_like(x)
        acc[2:] = vel[2:] - vel[1:-1]

        x = np.concatenate([x, vel, acc], axis=-1)
        x = x.transpose(2,0,1)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(ann['label'])

# ── MODEL ─────────────────────────────────────────
class GraphConv(nn.Module):
    def __init__(self, in_c, out_c, A):
        super().__init__()
        self.A = A
        self.K = A.shape[0]
        self.conv = nn.Conv2d(in_c, out_c*self.K, 1)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        B,C,T,V = x.shape
        x = self.conv(x).view(B,self.K,-1,T,V)
        out = torch.einsum('bkctv,kvw->bctw', x, self.A.to(x.device))
        return self.bn(out)

class STGCNBlock(nn.Module):
    def __init__(self, in_c, out_c, A):
        super().__init__()
        self.gcn = GraphConv(in_c,out_c,A)
        self.tcn = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_c,out_c,(9,1),padding=(4,0)),
            nn.BatchNorm2d(out_c),
            nn.Dropout(0.4)
        )
        self.res = nn.Identity() if in_c==out_c else nn.Conv2d(in_c,out_c,1)

    def forward(self,x):
        return F.relu(self.tcn(self.gcn(x)) + self.res(x))

class STGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(IN_CH*V)

        self.layers = nn.ModuleList([
            STGCNBlock(IN_CH,64,A),
            STGCNBlock(64,128,A),
            STGCNBlock(128,256,A),
        ])

        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self,x):
        B,C,T,V = x.shape
        x = x.permute(0,1,3,2).contiguous().view(B,C*V,T)
        x = self.data_bn(x)
        x = x.view(B,C,V,T).permute(0,1,3,2)

        for l in self.layers:
            x = l(x)

        x = x.mean(dim=[2,3])
        return self.fc(x)

# ── EVAL ─────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct,total = 0,0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X,y in loader:
            X,y = X.to(device), y.to(device)
            out = model(X)
            preds = out.argmax(1)

            correct += (preds==y).sum().item()
            total += y.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return correct/total, confusion_matrix(all_labels, all_preds)

# ── MAIN ─────────────────────────────────────────
def main():
    print("Device:", device)

    data = pickle.load(open("data/ntu60_hrnet.pkl","rb"))
    samples = data['annotations']
    random.shuffle(samples)

    n = len(samples)
    train = samples[:int(0.7*n)]
    val   = samples[int(0.7*n):int(0.85*n)]
    test  = samples[int(0.85*n):]

    # class weights
    labels = [s['label'] for s in train]
    class_counts = np.bincount(labels)
    weights = 1. / (class_counts + 1e-6)
    sample_weights = [weights[l] for l in labels]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(SkeletonDataset(train, augment),
                              batch_size=BATCH_SIZE, sampler=sampler)

    val_loader = DataLoader(SkeletonDataset(val),
                            batch_size=BATCH_SIZE)

    test_loader = DataLoader(SkeletonDataset(test),
                             batch_size=BATCH_SIZE)

    model = STGCN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    patience_counter = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for X,y in tqdm(train_loader):
            X,y = X.to(device), y.to(device)

            out = model(X)
            loss = criterion(out,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_acc,_ = evaluate(model, val_loader)
        scheduler.step()

        print(f"Epoch {epoch+1} | Loss {total_loss:.2f} | Val Acc {val_acc:.4f}")

        # early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),"best_model.pth")
            patience_counter = 0
            print("✅ Best model saved")
        else:
            patience_counter += 1
            if patience_counter > PATIENCE:
                print("⛔ Early stopping")
                break

    # final test
    model.load_state_dict(torch.load("best_model.pth"))
    test_acc, cm = evaluate(model, test_loader)

    print("\n🎯 Final Test Accuracy:", test_acc)
    print("\n📊 Confusion Matrix:\n", cm)

# ── RUN ─────────────────────────────────────────
if __name__ == "__main__":
    main()