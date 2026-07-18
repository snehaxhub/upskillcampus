"""
app.py
-------
Streamlit dashboard for the Gearbox Predictive Maintenance project.

Run with:
    streamlit run app.py

Pages (via sidebar):
  - Overview        : project summary + model performance
  - Predict          : upload a raw vsibration file -> get a healthy/broken prediction
  - Explore Dataset   : browse the training data, signals, and feature distributions
  - Prediction History: everything logged to the SQLite database so far
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from feature_utils import (
    load_raw_file, extract_windowed_features, extract_whole_file_features,
    get_feature_column_order
)
from db_utils import init_db, log_prediction, fetch_history, fetch_summary_counts, clear_history

MODEL_DIR = 'models'

# ----------------------------------------------------------------------------
# Page config + industrial dark theme styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Gearbox Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #0E1117;
    }
    .metric-card {
        background-color: #1B1F27;
        border: 1px solid #2A2F3A;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }
    .status-pill {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 18px;
        letter-spacing: 0.5px;
    }
    .status-healthy {
        background-color: #113B27;
        color: #4ADE80;
        border: 1px solid #1E5A3C;
    }
    .status-broken {
        background-color: #3B1414;
        color: #F87171;
        border: 1px solid #6B1F1F;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        letter-spacing: 0.3px;
    }
    section[data-testid="stSidebar"] {
        background-color: #14171F;
        border-right: 1px solid #2A2F3A;
    }
    .footer-note {
        color: #6B7280;
        font-size: 12px;
        margin-top: 30px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

init_db()

# ----------------------------------------------------------------------------
# Cached loaders
# ----------------------------------------------------------------------------
@st.cache_resource
def load_model_and_metadata():
    model = joblib.load(os.path.join(MODEL_DIR, 'gearbox_model.pkl'))
    with open(os.path.join(MODEL_DIR, 'feature_columns.json')) as f:
        feature_cols = json.load(f)
    with open(os.path.join(MODEL_DIR, 'model_metrics.json')) as f:
        metrics = json.load(f)
    return model, feature_cols, metrics


@st.cache_data
def load_training_features():
    return pd.read_csv(os.path.join(MODEL_DIR, 'training_features.csv'))


model, feature_cols, metrics = load_model_and_metadata()
training_df = load_training_features()

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Gearbox PdM")
st.sidebar.caption("UCT Internship — Project 8")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Predict", "Explore Dataset", "Prediction History"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model status:** :green[Loaded]  \n"
    f"**Test accuracy:** {metrics['test_accuracy']*100:.1f}%  \n"
    f"**Training samples:** {metrics['total_samples']}"
)

# ==============================================================================
# PAGE: OVERVIEW
# ==============================================================================
if page == "Overview":
    st.title("Predictive Maintenance of Gearbox Using Vibration Sensors")
    st.caption("Detecting gear tooth faults from vibration signals using machine learning")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Test Accuracy", f"{metrics['test_accuracy']*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Training Samples", metrics['total_samples'])
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Sensors Used", 4)
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Load Levels Covered", "0–90%")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("### Confusion Matrix (Held-Out Test Set)")
    cm = np.array(metrics['confusion_matrix'])
    labels = metrics['confusion_matrix_labels']
    fig = px.imshow(
        cm, text_auto=True, x=labels, y=labels,
        color_continuous_scale='Oranges',
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    fig.update_layout(
        paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
        font_color='#E6E6E6', height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### About This Project")
    st.markdown("""
    This dashboard uses vibration data from SpectraQuest's Gearbox Fault Diagnostics
    Simulator to classify a gearbox as **Healthy** or having a **Broken Tooth**.

    **Pipeline:** raw vibration signal → windowed statistical feature extraction
    (mean, std, RMS, kurtosis, skewness, crest factor, peak-to-peak, across 4 sensors)
    → Random Forest classifier → prediction logged to database.
    """)

# ==============================================================================
# PAGE: PREDICT
# ==============================================================================
elif page == "Predict":
    st.title("🔍 Run a Prediction")
    st.caption("Upload a raw vibration file (tab-separated .txt, 4 sensor columns) to classify gearbox condition")

    uploaded_file = st.file_uploader("Upload vibration file", type=['txt'])

    demo_col1, demo_col2 = st.columns(2)
    with demo_col1:
        use_demo_healthy = st.button("▶ Try a demo Healthy sample")
    with demo_col2:
        use_demo_broken = st.button("▶ Try a demo Broken Tooth sample")

    filepath_to_use = None
    source_label = None

    if uploaded_file is not None:
        filepath_to_use = uploaded_file
        source_label = uploaded_file.name
    elif use_demo_healthy:
        filepath_to_use = 'gbdata/Healthy Data/h30hz50.txt'
        source_label = 'demo_healthy_h30hz50.txt'
    elif use_demo_broken:
        filepath_to_use = 'gbdata/BrokenTooth Data/b30hz50.txt'
        source_label = 'demo_broken_b30hz50.txt'

    if filepath_to_use is not None:
        with st.spinner("Extracting features and running model..."):
            raw_df = load_raw_file(filepath_to_use)

            # Whole-file feature summary -> single prediction for this upload
            feat_row = extract_whole_file_features(raw_df)
            X = pd.DataFrame([feat_row])[feature_cols].values

            pred = model.predict(X)[0]
            proba = model.predict_proba(X)[0]
            classes = list(model.classes_)
            confidence = proba[classes.index(pred)]

            log_prediction(source_label, pred, float(confidence), feat_row)

        st.markdown("### Result")
        pill_class = "status-healthy" if pred == "healthy" else "status-broken"
        icon = "✅" if pred == "healthy" else "⚠️"
        st.markdown(
            f'<span class="status-pill {pill_class}">{icon} {pred.upper()}</span>'
            f'&nbsp;&nbsp; Confidence: **{confidence*100:.1f}%**',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### Raw Signal (Sensor 1, first 2000 samples)")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=raw_df['sensor1'].values[:2000], mode='lines',
                line=dict(color='#FF8C42', width=1),
            ))
            fig.update_layout(
                paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
                font_color='#E6E6E6', height=350,
                xaxis_title="Sample", yaxis_title="Amplitude",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Key Features (Sensor 1)")
            key_feats = {k.replace('sensor1_', ''): v for k, v in feat_row.items() if k.startswith('sensor1_')}
            st.dataframe(
                pd.DataFrame(key_feats.items(), columns=['Feature', 'Value']).round(4),
                hide_index=True, use_container_width=True,
            )
    else:
        st.info("Upload a file or click a demo button above to run a prediction.")

# ==============================================================================
# PAGE: EXPLORE DATASET
# ==============================================================================
elif page == "Explore Dataset":
    st.title("📊 Explore the Training Dataset")

    st.markdown("### Feature Distributions by Condition")
    feature_choice = st.selectbox(
        "Choose a feature to compare",
        [c for c in feature_cols if c.startswith('sensor1_')],
        format_func=lambda x: x.replace('sensor1_', '').replace('_', ' ').title(),
    )
    fig = px.box(
        training_df, x='condition', y=feature_choice, color='condition',
        color_discrete_map={'healthy': '#4ADE80', 'broken': '#F87171'},
        points='all',
    )
    fig.update_layout(
        paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
        font_color='#E6E6E6', height=450, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Feature vs. Load Level")
    fig2 = px.scatter(
        training_df, x='load_pct', y=feature_choice, color='condition',
        color_discrete_map={'healthy': '#4ADE80', 'broken': '#F87171'},
    )
    fig2.update_layout(
        paper_bgcolor='#0E1117', plot_bgcolor='#0E1117',
        font_color='#E6E6E6', height=400,
        xaxis_title="Load %", yaxis_title=feature_choice,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Raw Feature Table (sample)")
    st.dataframe(training_df.sample(min(20, len(training_df))), use_container_width=True)

# ==============================================================================
# PAGE: PREDICTION HISTORY
# ==============================================================================
elif page == "Prediction History":
    st.title("🗂️ Prediction History (from Database)")
    st.caption("Every prediction made on the 'Predict' page is logged here via SQLite.")

    history_df = fetch_history()

    if history_df.empty:
        st.info("No predictions logged yet. Go to the Predict page to run one.")
    else:
        summary = fetch_summary_counts()
        col1, col2 = st.columns([1, 2])
        with col1:
            fig = px.pie(
                summary, names='predicted_condition', values='count',
                color='predicted_condition',
                color_discrete_map={'healthy': '#4ADE80', 'broken': '#F87171'},
                hole=0.5,
            )
            fig.update_layout(
                paper_bgcolor='#0E1117', font_color='#E6E6E6', height=300,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.dataframe(history_df, use_container_width=True, height=300)

        if st.button("🗑️ Clear History"):
            clear_history()
            st.rerun()

st.markdown(
    '<div class="footer-note">Predictive Maintenance of Gearbox Using Vibration Sensors — UCT Internship Project 8</div>',
    unsafe_allow_html=True,
)
