import pickle
import random
import os

# ── CONFIG ─────────────────────────
SEED = 42
random.seed(SEED)

# ── CREATE FOLDER ──────────────────
os.makedirs("data/test_samples", exist_ok=True)

# ── LOAD DATA ──────────────────────
data = pickle.load(open("data/ntu60_hrnet.pkl", "rb"))
samples = data['annotations']

print("Total samples:", len(samples))

# ── SHUFFLE ────────────────────────
random.shuffle(samples)

# ── SPLIT (same as training) ───────
n = len(samples)

train = samples[:int(0.7*n)]
val   = samples[int(0.7*n):int(0.85*n)]
test  = samples[int(0.85*n):]

print("Test samples:", len(test))

# ── SELECT 59 RANDOM TEST SAMPLES ──
selected = random.sample(test, 59)

# ── SAVE INDIVIDUAL FILES ──────────
for i, sample in enumerate(selected):
    file_path = f"data/test_samples/sample_{i}.pkl"

    with open(file_path, "wb") as f:
        pickle.dump({"annotations": [sample]}, f)

print("✅ 59 individual test samples created in data/test_samples/")