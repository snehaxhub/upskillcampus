"""
train_and_save_model.py
--------------------------
Builds the windowed feature dataset from the raw SpectraQuest gearbox
files, trains a Random Forest classifier (with a group-aware
train/test split so no file's windows leak between train and test),
and saves:
  - models/gearbox_model.pkl        (the trained model)
  - models/feature_columns.json     (exact feature order the model expects)
  - models/model_metrics.json       (accuracy + confusion matrix, shown in the app)

Run this once (or whenever you re-train) BEFORE launching the Streamlit app:
    python train_and_save_model.py
"""

import glob
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from feature_utils import load_raw_file, extract_windowed_features, get_feature_column_order

HEALTHY_DIR = 'gbdata/Healthy Data'
BROKEN_DIR = 'gbdata/BrokenTooth Data'
MODEL_DIR = 'models'


def build_dataset():
    all_rows = []
    file_id = 0

    for filepath in sorted(glob.glob(os.path.join(HEALTHY_DIR, '*.txt'))):
        fname = os.path.basename(filepath)
        load_pct = int(fname.replace('h30hz', '').replace('.txt', ''))
        df = load_raw_file(filepath)
        feats = extract_windowed_features(df)
        feats['condition'] = 'healthy'
        feats['load_pct'] = load_pct
        feats['file_id'] = file_id
        all_rows.append(feats)
        file_id += 1

    for filepath in sorted(glob.glob(os.path.join(BROKEN_DIR, '*.txt'))):
        fname = os.path.basename(filepath)
        load_pct = int(fname.replace('b30hz', '').replace('.txt', ''))
        df = load_raw_file(filepath)
        feats = extract_windowed_features(df)
        feats['condition'] = 'broken'
        feats['load_pct'] = load_pct
        feats['file_id'] = file_id
        all_rows.append(feats)
        file_id += 1

    return pd.concat(all_rows, ignore_index=True)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print('Building windowed feature dataset...')
    df = build_dataset()
    df.to_csv(os.path.join(MODEL_DIR, 'training_features.csv'), index=False)
    print(f'Total samples: {len(df)}')

    feature_cols = get_feature_column_order()
    X = df[feature_cols].values
    y = df['condition'].values
    groups = df['file_id'].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print('Training Random Forest...')
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=['healthy', 'broken']).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f'Test accuracy: {acc:.4f}')

    # Re-train on ALL data for the final deployed model (common practice once you've
    # validated performance on the held-out split above)
    final_model = RandomForestClassifier(n_estimators=200, random_state=42)
    final_model.fit(X, y)

    joblib.dump(final_model, os.path.join(MODEL_DIR, 'gearbox_model.pkl'))
    with open(os.path.join(MODEL_DIR, 'feature_columns.json'), 'w') as f:
        json.dump(feature_cols, f)
    with open(os.path.join(MODEL_DIR, 'model_metrics.json'), 'w') as f:
        json.dump({
            'test_accuracy': acc,
            'confusion_matrix': cm,
            'confusion_matrix_labels': ['healthy', 'broken'],
            'classification_report': report,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'total_samples': len(X),
        }, f, indent=2)

    print('Saved model, feature columns, and metrics to models/')


if __name__ == '__main__':
    main()
