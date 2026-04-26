import streamlit as st
import os
import pickle
import json
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import confusion_matrix

from predict import predict

# ================= CONFIG =================
st.set_page_config(page_title="Action Recognition AI", layout="wide")

DATA_FOLDER = "data/test_samples"

# ================= LOAD =================
@st.cache_data
def load_label_map():
    with open("label_map.json") as f:
        return json.load(f)

@st.cache_data
def load_files():
    return os.listdir(DATA_FOLDER)

@st.cache_data
def load_sample(path):
    return pickle.load(open(path, "rb"))

label_map = load_label_map()
files = load_files()

# ================= PREMIUM UI =================
st.markdown("""
<style>

/* ===== BACKGROUND ===== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1f1c2c, #928dab);
}

/* ===== HEADER ===== */
.header {
    font-size: 36px;
    font-weight: 700;
    background: -webkit-linear-gradient(#00f5a0, #00d9f5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.glow {
    text-shadow: 0 0 10px #00f5a0, 0 0 20px #00d9f5;
}

/* ===== METRIC CARDS ===== */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 10px;
    box-shadow: 0 0 15px rgba(0,255,255,0.1);
}

/* ===== BUTTON ===== */
.stButton>button {
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    color: black;
    border-radius: 10px;
    border: none;
    font-weight: bold;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #00d9f5, #00f5a0);
}

/* ===== TEXT ===== */
h1, h2, h3, p {
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<h1 class='header' style='text-align:center;'>
🧠 Human Action Recognition AI
</h1>
<p style='text-align:center; color:#cfd8dc;'>
ST-GCN • Skeleton Intelligence • 60 Actions
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ================= SIDEBAR =================
page = st.sidebar.radio(
    "🧭 Navigation",
    ["Home","Predict","Analytics","Confusion Matrix","Per-Class Accuracy","Error Analysis","Skeleton Viewer"]
)
# ================= HOME =================
if page == "Home":

    # ================= HERO =================
    # st.markdown("""
    # <h1 style='text-align:center;'>🧠 Human Action Recognition AI</h1>
    # <p style='text-align:center; color:gray;'>
    # ST-GCN Based Skeleton Intelligence • 60 Action Classes • Real-Time Ready
    # </p>
    # """, unsafe_allow_html=True)

    # st.markdown("---")

    # ================= METRICS =================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📦 Samples", len(files))
    col2.metric("🎯 Classes", len(label_map))
    col3.metric("🧠 Model", "ST-GCN")
    col4.metric("⚡ Status", "READY")

    st.markdown("---")

    # ================= TABS =================
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "🔍 Predict", "📈 Insights"])

    # =========================================================
    # ================= TAB 1: OVERVIEW =======================
    # =========================================================
    with tab1:

        st.subheader("⚙️ Model Pipeline")

        st.markdown("""
        1. Input: Skeleton keypoints (17 joints)  
        2. Preprocessing: Normalization + Velocity + Acceleration  
        3. Model: ST-GCN (Spatial + Temporal learning)  
        4. Output: 60 human actions  
        """)

        st.markdown("---")

        st.subheader("📊 Dataset Distribution")

        labels = []

        for f in files:
            try:
                data = load_sample(os.path.join(DATA_FOLDER, f))
                lbl = data['annotations'][0]['label']
                labels.append(label_map[str(lbl)])
            except:
                pass

        if labels:
            df = pd.DataFrame(labels, columns=["Action"])

            st.plotly_chart(
                px.histogram(df, x="Action", title="Action Distribution"),
                use_container_width=True
            )

        st.markdown("---")

        st.subheader("🔎 Sample Predictions Preview")

        preview_data = []

        for f in files[:5]:
            try:
                with open(os.path.join(DATA_FOLDER, f), "rb") as file:
                    label, conf, _ = predict(file)

                preview_data.append([f, label, round(conf, 3)])
            except:
                pass

        if preview_data:
            df_preview = pd.DataFrame(
                preview_data,
                columns=["File", "Prediction", "Confidence"]
            )
            st.dataframe(df_preview, use_container_width=True)

    # =========================================================
    # ================= TAB 2: PREDICT ========================
    # =========================================================
    with tab2:

        st.subheader("🚀 AI Prediction Engine")

        selected = st.selectbox("Select Sample", files)
        top_k = st.slider("Top-K Predictions", 1, 10, 5)

        if st.button("⚡ Run Prediction"):

            with st.spinner("Analyzing motion..."):

                path = os.path.join(DATA_FOLDER, selected)

                data = load_sample(path)
                kp = data['annotations'][0]['keypoint'][0]

                with open(path, "rb") as f:
                    label, conf, probs = predict(f)

                # ===== Explainability =====
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

                important_joints = [joint_names[i] for i in top_joints]

            # ===== RESULT =====
            st.success(f"🎯 Prediction: {label}")
            st.progress(conf)
            st.info(f"Confidence: {conf:.4f}")

            # ===== EXPLANATION =====
            st.markdown("### 🧠 Why this prediction?")
            st.write(
                f"Model focused on movement in: **{', '.join(important_joints)}**"
            )

            # ===== TOP-K =====
            df = pd.DataFrame(
                sorted(probs.items(), key=lambda x: x[1], reverse=True)[:top_k],
                columns=["Action", "Probability"]
            )

            st.plotly_chart(
                px.bar(df, x="Probability", y="Action", orientation="h"),
                use_container_width=True
            )

            # ===== DOWNLOAD =====
            st.download_button(
                "⬇️ Download Results",
                df.to_csv(index=False),
                "prediction.csv"
            )

    # =========================================================
    # ================= TAB 3: INSIGHTS =======================
    # =========================================================
    with tab3:

        st.subheader("📊 Live Action Leaderboard")

        predictions = []

        for f in files[:50]:
            try:
                with open(os.path.join(DATA_FOLDER, f), "rb") as file:
                    label, conf, _ = predict(file)
                    predictions.append(label)
            except:
                pass

        if predictions:

            df = pd.DataFrame(predictions, columns=["Action"])

            leaderboard = df["Action"].value_counts().reset_index()
            leaderboard.columns = ["Action", "Count"]

            st.plotly_chart(
                px.bar(
                    leaderboard.head(10),
                    x="Count",
                    y="Action",
                    orientation="h",
                    title="Top Predicted Actions"
                ),
                use_container_width=True
            )

        st.markdown("---")

        st.subheader("📈 Confidence Trend")

        trend = []

        for f in files[:30]:
            try:
                with open(os.path.join(DATA_FOLDER, f), "rb") as file:
                    _, conf, _ = predict(file)
                trend.append(conf)
            except:
                pass

        if trend:
            df2 = pd.DataFrame({
                "Index": range(len(trend)),
                "Confidence": trend
            })

            st.plotly_chart(
                px.line(df2, x="Index", y="Confidence"),
                use_container_width=True
            )

            col1, col2, col3 = st.columns(3)
            col1.metric("Average", round(np.mean(trend), 3))
            col2.metric("Max", round(np.max(trend), 3))
            col3.metric("Min", round(np.min(trend), 3))

# ================= PREDICT PAGE =================
elif page == "Predict":

    st.title("🔍 Prediction")

    selected = st.selectbox("Select Sample", files)

    if st.button("Predict"):
        with st.spinner("Processing..."):
            path = os.path.join(DATA_FOLDER, selected)

            with open(path, "rb") as f:
                label, conf, probs = predict(f)

        st.success(label)
        st.info(f"Confidence: {conf:.4f}")

        df = pd.DataFrame(
            sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5],
            columns=["Action", "Probability"]
        )

        st.plotly_chart(px.bar(df, x="Probability", y="Action", orientation="h"))

# ================= ANALYTICS =================
elif page == "Analytics":

    st.title("📊 Analytics")

    labels = []

    for f in files:
        try:
            data = load_sample(os.path.join(DATA_FOLDER, f))
            lbl = data['annotations'][0]['label']
            labels.append(label_map[str(lbl)])
        except:
            pass

    df = pd.DataFrame(labels, columns=["Action"])
    st.plotly_chart(px.histogram(df, x="Action"))

# ================= CONFUSION MATRIX =================
elif page == "Confusion Matrix":

    st.title("📉 Confusion Matrix")

    y_true, y_pred = [], []

    for f in files:
        try:
            path = os.path.join(DATA_FOLDER, f)

            data = load_sample(path)
            true = data['annotations'][0]['label']

            with open(path, "rb") as file:
                pred, _, _ = predict(file)

            pred_idx = list(label_map.values()).index(pred)

            y_true.append(true)
            y_pred.append(pred_idx)

        except:
            pass

    if y_true:
        cm = confusion_matrix(y_true, y_pred)
        st.plotly_chart(px.imshow(cm))
    else:
        st.warning("No data available")

# ================= PER CLASS =================
elif page == "Per-Class Accuracy":

    st.title("🎯 Per-Class Accuracy")

    correct, total = {}, {}

    for f in files:
        try:
            path = os.path.join(DATA_FOLDER, f)

            data = load_sample(path)
            true = data['annotations'][0]['label']

            with open(path, "rb") as file:
                pred, _, _ = predict(file)

            pred_idx = list(label_map.values()).index(pred)

            total[true] = total.get(true, 0) + 1

            if true == pred_idx:
                correct[true] = correct.get(true, 0) + 1

        except:
            pass

    results = []

    for cls in total:
        acc = correct.get(cls, 0) / total[cls]
        results.append([label_map[str(cls)], acc])

    df = pd.DataFrame(results, columns=["Action", "Accuracy"])
    st.plotly_chart(px.bar(df, x="Action", y="Accuracy"))

# ================= ERROR ANALYSIS =================
elif page == "Error Analysis":

    st.title("❌ Error Analysis")

    errors = []

    for f in files:
        try:
            path = os.path.join(DATA_FOLDER, f)

            data = load_sample(path)
            true = data['annotations'][0]['label']

            with open(path, "rb") as file:
                pred, conf, _ = predict(file)

            pred_idx = list(label_map.values()).index(pred)

            if true != pred_idx:
                errors.append({
                    "File": f,
                    "True": label_map[str(true)],
                    "Predicted": pred,
                    "Confidence": round(conf, 3)
                })

        except:
            pass

    if errors:
        st.dataframe(pd.DataFrame(errors))
    else:
        st.success("No errors found 🎉")

# ================= SKELETON =================
elif page == "Skeleton Viewer":

    st.title("🧬 Explainable Skeleton Viewer")

    selected = st.selectbox("Select File", files)

    data = load_sample(os.path.join(DATA_FOLDER, selected))
    kp = data['annotations'][0]['keypoint'][0]

    # ================= SETTINGS =================
    frame = st.slider("Frame", 0, len(kp)-1, 0)

    show_trail = st.checkbox("Show Motion Trail", True)
    show_labels = st.checkbox("Show Joint Labels", False)

    # ================= JOINT NAMES =================
    joint_names = [
        "Nose","Neck","R-Shoulder","R-Elbow","R-Wrist",
        "L-Shoulder","L-Elbow","L-Wrist",
        "Mid-Hip","R-Hip","R-Knee","R-Ankle",
        "L-Hip","L-Knee","L-Ankle","R-Eye","L-Eye"
    ]

    # ================= SKELETON EDGES =================
    edges = [
        (0,1),(1,2),(2,3),(3,4),
        (1,5),(5,6),(6,7),
        (1,8),
        (8,9),(9,10),(10,11),
        (8,12),(12,13),(13,14)
    ]

    coords = kp[frame][:, :2]

    import plotly.graph_objects as go

    fig = go.Figure()

    # ================= DRAW BONES =================
    for e in edges:
        x = [coords[e[0]][0], coords[e[1]][0]]
        y = [coords[e[0]][1], coords[e[1]][1]]

        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(width=3),
            showlegend=False
        ))

    # ================= DRAW JOINTS =================
    fig.add_trace(go.Scatter(
        x=coords[:,0],
        y=coords[:,1],
        mode='markers+text' if show_labels else 'markers',
        text=joint_names if show_labels else None,
        textposition="top center",
        marker=dict(size=8),
        name="Joints"
    ))

    # ================= MOTION TRAIL =================
    if show_trail and frame > 0:
        trail_frames = range(max(0, frame-10), frame)

        for f in trail_frames:
            trail = kp[f][:, :2]

            fig.add_trace(go.Scatter(
                x=trail[:,0],
                y=trail[:,1],
                mode='markers',
                opacity=0.2,
                marker=dict(size=4),
                showlegend=False
            ))

    # ================= LAYOUT =================
    fig.update_layout(
        title=f"Frame {frame}",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, autorange="reversed"),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # ================= EXPLANATION PANEL =================
    st.markdown("---")
    st.subheader("🧠 Explanation")

    st.write(f"Total Frames: {len(kp)}")
    st.write(f"Current Frame: {frame}")

    if frame > 0:
        movement = np.mean(np.abs(kp[frame] - kp[frame-1]))
        st.write(f"Motion Intensity: {movement:.4f}")

        if movement < 0.001:
            st.info("🧍 Low movement → likely static action (standing)")
        elif movement < 0.01:
            st.info("🚶 Moderate movement")
        else:
            st.info("🏃 High movement → dynamic action")