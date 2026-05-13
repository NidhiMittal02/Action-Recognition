# 🧠 Human Action Recognition using ST-GCN

A deep learning-based Human Action Recognition system using **ST-GCN (Spatio-Temporal Graph Convolutional Network)** and human skeleton keypoints.

The project predicts human actions from skeleton movement sequences and provides an interactive Streamlit dashboard for visualization and analysis.

---

# 📁 Project Structure

```text
Human-Action-Recognition/
│
├── data/
│   ├── ntu60_hrnet.pkl              # Main dataset
│   └── test_samples/                # Test skeleton samples
│
├── app.py                           # Streamlit dashboard
├── train.py                         # Model training script
├── predict.py                       # Prediction/inference script
├── best_model.pth                   # Saved trained model
├── label_map.json                   # Label-to-action mapping
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

---

# 🚀 Features

- ST-GCN based skeleton action recognition
- Human skeleton visualization
- Motion trail visualization
- Real-time prediction dashboard
- Top-K action predictions
- Confusion matrix analysis
- Per-class accuracy analysis
- Explainable AI interface

---

# 🧠 Model Architecture

The project uses:

# 👉 ST-GCN (Spatio-Temporal Graph Convolutional Network)

### Model Pipeline

```text
Skeleton Keypoints
        ↓
Preprocessing
        ↓
Velocity + Acceleration Features
        ↓
Spatial Graph Convolution
        ↓
Temporal Convolution
        ↓
Feature Extraction
        ↓
Global Average Pooling
        ↓
Fully Connected Layer
        ↓
Action Prediction
```

---

# 📊 Dataset Information

- Total Classes: 60
- Joints: 17
- Features:
  - Position (x, y)
  - Velocity
  - Acceleration

---

# ⚙️ Installation

## 1️⃣ Clone Repository

```bash
git clone <repository-link>
cd Human-Action-Recognition
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Training

Run the training script:

```bash
python train.py
```

The best model will be saved as:

```text
best_model.pth
```

---

# ▶️ Run Streamlit Dashboard

```bash
streamlit run app.py
```

---

# 📈 Evaluation Metrics

- Accuracy
- Confusion Matrix
- Per-Class Accuracy
- Confidence Score

---

# 🧬 Skeleton Viewer

The dashboard includes:
- Joint visualization
- Bone connections
- Motion trail analysis
- Frame-by-frame inspection

---

# 🔥 Why ST-GCN?

ST-GCN is used because:

- Human skeleton naturally forms a graph
- Captures spatial joint relationships
- Learns temporal movement patterns
- Efficient for skeleton-based action recognition

---

# 📌 Future Improvements

- Real-time webcam prediction
- Transformer-based hybrid models
- Attention mechanisms
- 3D skeleton support
- Multi-person action recognition

---
