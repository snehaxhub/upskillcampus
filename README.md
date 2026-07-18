# Predictive Maintenance of Gearbox Using Vibration Sensors

**UCT Internship — Project 8**
Author: Sneha Dhanawade

An end-to-end machine learning system that detects gearbox faults — specifically a broken gear tooth — from vibration sensor data, before they cause complete equipment failure. Built as part of an industrial internship with UniConverge Technologies (UCT) and upskill Campus.

Predictive maintenance allows manufacturers to lower maintenance costs, extend equipment life, reduce downtime, and improve production quality by addressing problems before they cause equipment failures.

---

## Overview

The dataset comes from SpectraQuest's Gearbox Fault Diagnostics Simulator: vibration signals recorded by **4 sensors** placed in four different directions, under **load levels from 0% to 90%**, in two conditions — **Healthy** and **Broken Tooth**.

This project takes that raw sensor data all the way to a live, interactive prediction tool:

```
Raw vibration signals  →  Feature extraction  →  Model training  →  Streamlit dashboard  →  Database logging
```

**Result:** a Random Forest classifier that distinguishes Healthy vs. Broken Tooth gearboxes with **99.1% accuracy** on a held-out test set, wrapped in a dark, industrial-themed dashboard for live predictions.

---

## How it works

### 1. Feature Extraction
Each raw file has ~88,000 samples per sensor — too long to feed a model directly. Instead, each signal is split into non-overlapping **windows of 2048 samples**, and 7 statistical features are computed per window, per sensor:

| Feature | What it captures |
|---|---|
| `mean` | Average signal value |
| `std` | Spread / energy of the signal |
| `rms` | Root mean square — overall vibration energy |
| `kurtosis` | "Peakiness" — spikes from impacts raise this sharply |
| `skewness` | Asymmetry of the signal |
| `crest_factor` | Peak ÷ RMS — high for impulsive, spiky faults |
| `peak_to_peak` | Max − min — overall signal swing |

With 4 sensors × 7 features, each window becomes a row of 28 features. Windowing turns 20 raw files into **978 samples** — enough to train and evaluate a model properly.

### 2. Model Training
A `RandomForestClassifier` (200 trees) is trained on these features. Critically, the train/test split is **grouped by original file** (`GroupShuffleSplit`), so windows from the same recording never appear in both the training and test sets — this prevents the model from "cheating" and gives an honest accuracy estimate.

> An earlier whole-file-only approach (20 samples, no windowing) scored a **suspicious 100% accuracy** — a classic overfitting red flag from too little data. Windowing + grouped splitting fixed this and gave a trustworthy **99.1%** result instead.

### 3. Interactive Dashboard
A Streamlit app (`GearboxPredictiveMaintenance.py`) lets you:
- **Overview** — see model accuracy and confusion matrix
- **Predict** — upload a raw vibration file (or use a demo sample) and get an instant Healthy/Broken Tooth prediction with confidence
- **Explore Dataset** — interactive plots comparing healthy vs. faulty feature distributions across load levels
- **Prediction History** — every prediction is logged to a SQLite database and viewable as a table + chart

---

## Project Structure

```
├── GearboxPredictiveMaintenance.py   # Main Streamlit dashboard app
├── feature_utils.py                   # Shared feature extraction (used by training AND the app)
├── db_utils.py                        # SQLite database layer (prediction logging)
├── train_and_save_model.py            # Builds features, trains model, saves to models/
├── models/
│   ├── gearbox_model.pkl              # Trained Random Forest model
│   ├── feature_columns.json           # Exact feature order the model expects
│   └── model_metrics.json             # Accuracy, confusion matrix, classification report
├── requirements.txt
└── README.md
```

---

## Setup & Run

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. (Optional — a trained model is already included) To retrain from scratch, place the dataset as:
   ```
   gbdata/Healthy Data/*.txt
   gbdata/BrokenTooth Data/*.txt
   ```
   then run:
   ```bash
   python train_and_save_model.py
   ```
3. Launch the dashboard:
   ```bash
   streamlit run GearboxPredictiveMaintenance.py
   ```
   Opens at `http://localhost:8501`.

---

## Results

| Metric | Value |
|---|---|
| Test accuracy | **99.13%** |
| Training samples | 978 (windowed) |
| Train / Test split | 747 / 231 (grouped by file, no leakage) |
| Healthy precision / recall | 1.00 / 0.99 |
| Broken Tooth precision / recall | 0.98 / 1.00 |

The most influential features were `sensor1_peak_to_peak`, `sensor1_rms`, and `sensor1_std` — consistent with the physical intuition that a broken tooth causes a sharp, high-amplitude impact once per rotation.

---

## Database

Every prediction made through the "Predict" page is logged to a local **SQLite** database (`gearbox_predictions.db`) — no server setup required. See `db_utils.py` for the schema and query functions. The code is structured so swapping SQLite for PostgreSQL/MySQL later only requires changing the connection logic in `db_utils.py`, not the rest of the app.

---

## Future Work

- Add frequency-domain (FFT) features around the gear-mesh frequency and harmonics
- Compare against SVM / Gradient Boosting / XGBoost
- Test generalization across shaft speeds and additional fault types
- Migrate to a production database for multi-user deployment
- Deploy the dashboard publicly (e.g., Streamlit Community Cloud)

---

## Acknowledgements
This project was completed as part of an Industrial Internship Program by upskill Campus and The IoT Academy, in collaboration with UniConverge Technologies Pvt Ltd (UCT).
This project was completed as part of an Industrial Internship Program by **upskill Campus** and **The IoT Academy**, in collaboration with **UniConverge Technologies Pvt Ltd (UCT)**.
