"""
feature_utils.py
------------------
Shared feature-extraction logic, used by BOTH the training script and
the Streamlit app. Keeping this in one place guarantees the app
extracts features in exactly the same way the model was trained on.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

SENSOR_COLUMNS = ['sensor1', 'sensor2', 'sensor3', 'sensor4']
WINDOW_SIZE = 2048
FEATURE_NAMES = ['mean', 'std', 'rms', 'kurtosis', 'skewness', 'crest_factor', 'peak_to_peak']


def load_raw_file(filepath_or_buffer):
    """Load a raw tab-separated vibration file into a 4-column DataFrame.
    Works with a filesystem path OR an in-memory uploaded file object
    (e.g. from Streamlit's file_uploader)."""
    df = pd.read_csv(filepath_or_buffer, sep='\t', header=None)
    df = df.dropna(axis=1, how='all')
    df.columns = SENSOR_COLUMNS[:df.shape[1]]
    return df


def extract_features_from_signal(signal: np.ndarray) -> dict:
    """Compute the 7 statistical features for one 1D sensor signal."""
    rms = np.sqrt(np.mean(signal ** 2))
    peak = np.max(np.abs(signal))
    return {
        'mean': float(np.mean(signal)),
        'std': float(np.std(signal)),
        'rms': float(rms),
        'kurtosis': float(kurtosis(signal)),
        'skewness': float(skew(signal)),
        'crest_factor': float(peak / rms) if rms != 0 else 0.0,
        'peak_to_peak': float(np.max(signal) - np.min(signal)),
    }


def extract_windowed_features(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> pd.DataFrame:
    """Split a raw signal DataFrame into windows and extract features from each.
    Returns one row of features per window (same schema used in training)."""
    n_windows = len(df) // window_size
    rows = []
    for w in range(n_windows):
        start, end = w * window_size, (w + 1) * window_size
        row = {'window': w}
        for sensor in SENSOR_COLUMNS:
            if sensor not in df.columns:
                continue
            segment = df[sensor].values[start:end]
            feats = extract_features_from_signal(segment)
            for feat_name, value in feats.items():
                row[f'{sensor}_{feat_name}'] = value
        rows.append(row)
    return pd.DataFrame(rows)


def extract_whole_file_features(df: pd.DataFrame) -> dict:
    """Extract ONE feature row summarizing the entire file (used for a quick
    single-shot prediction rather than per-window predictions)."""
    row = {}
    for sensor in SENSOR_COLUMNS:
        if sensor not in df.columns:
            continue
        feats = extract_features_from_signal(df[sensor].values)
        for feat_name, value in feats.items():
            row[f'{sensor}_{feat_name}'] = value
    return row


def get_feature_column_order():
    """The exact column order the model expects. Must match training."""
    cols = []
    for sensor in SENSOR_COLUMNS:
        for feat in FEATURE_NAMES:
            cols.append(f'{sensor}_{feat}')
    return cols
