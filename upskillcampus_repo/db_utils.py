"""
db_utils.py
------------
Very lightweight database layer using SQLite (built into Python, no
server needed - perfect for an internship project / demo app).

Every time the app makes a prediction, we log it here: filename,
timestamp, predicted condition, confidence, and the key feature
values. This gives you a persistent "prediction history" table you
can show your mentor, and is a realistic stand-in for how a real
industrial monitoring system would log sensor readings + model
verdicts to a database over time.

To swap this for a real production database (PostgreSQL/MySQL) later,
you would only need to change `get_connection()` below to use a
different driver (e.g. psycopg2) - the rest of the code (INSERT/SELECT
calls) stays almost the same since we use plain SQL.
"""

import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'gearbox_predictions.db')


def get_connection():
    return sqlite3.connect(DB_PATH)


def init_db():
    """Create the predictions table if it doesn't already exist. Safe to call every app run."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            source_file TEXT,
            predicted_condition TEXT NOT NULL,
            confidence REAL NOT NULL,
            sensor1_rms REAL,
            sensor1_kurtosis REAL,
            sensor1_peak_to_peak REAL,
            load_pct_estimate TEXT
        )
    ''')
    conn.commit()
    conn.close()


def log_prediction(source_file, predicted_condition, confidence, feature_row: dict, load_pct_estimate="N/A"):
    """Insert one prediction record into the database."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO predictions
            (timestamp, source_file, predicted_condition, confidence,
             sensor1_rms, sensor1_kurtosis, sensor1_peak_to_peak, load_pct_estimate)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(timespec='seconds'),
        source_file,
        predicted_condition,
        confidence,
        feature_row.get('sensor1_rms'),
        feature_row.get('sensor1_kurtosis'),
        feature_row.get('sensor1_peak_to_peak'),
        load_pct_estimate,
    ))
    conn.commit()
    conn.close()


def fetch_history(limit: int = 100) -> pd.DataFrame:
    """Return the most recent N predictions as a DataFrame, for display in the app."""
    conn = get_connection()
    df = pd.read_sql_query(
        'SELECT * FROM predictions ORDER BY id DESC LIMIT ?', conn, params=(limit,)
    )
    conn.close()
    return df


def fetch_summary_counts() -> pd.DataFrame:
    """Count of healthy vs broken predictions logged so far - for a summary chart."""
    conn = get_connection()
    df = pd.read_sql_query(
        'SELECT predicted_condition, COUNT(*) as count FROM predictions GROUP BY predicted_condition', conn
    )
    conn.close()
    return df


def clear_history():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('DELETE FROM predictions')
    conn.commit()
    conn.close()
