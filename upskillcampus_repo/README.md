# Gearbox Predictive Maintenance — Streamlit Dashboard

A dark, industrial-themed dashboard for the Project 8 gearbox fault
detection model: upload a vibration file, get an instant Healthy /
Broken Tooth prediction, explore the training data, and review a
logged history of past predictions (backed by SQLite).

## Architecture — how the pieces connect

```
┌─────────────────────┐
│  Raw vibration data  │   gbdata/Healthy Data/*.txt, gbdata/BrokenTooth Data/*.txt
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│  feature_utils.py    │   Shared feature extraction (used by BOTH training and the app,
│                       │   so the app always extracts features exactly like training did)
└──────────┬───────────┘
           │
    ┌──────┴───────┐
    ▼              ▼
┌─────────┐   ┌──────────────────┐
│ train_   │   │   app.py          │
│ and_save │   │  (Streamlit UI)   │
│ _model.py│   └────────┬──────────┘
└────┬─────┘            │
     │ saves             │ loads
     ▼                  ▼
┌─────────────────────────────┐
│ models/gearbox_model.pkl      │  <- trained Random Forest, via joblib
│ models/feature_columns.json   │  <- exact feature order the model expects
│ models/model_metrics.json     │  <- accuracy/confusion matrix, shown on Overview page
└─────────────────────────────┘

app.py also talks to:
┌─────────────────────┐
│    db_utils.py        │  <- SQLite wrapper (init_db, log_prediction, fetch_history)
└──────────┬───────────┘
           ▼
┌─────────────────────┐
│ gearbox_predictions.db│  <- created automatically on first run
└─────────────────────┘
```

**Key idea:** `feature_utils.py` is imported by both `train_and_save_model.py`
and `app.py`. This is important — if the app extracted features
differently than training did, predictions would be silently wrong.
Having one shared module guarantees consistency.

## Setup & Run

1. Place the dataset so it looks like this:
   ```
   gbdata/Healthy Data/*.txt
   gbdata/BrokenTooth Data/*.txt
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Train the model (only needs to be done once, or whenever you
   change the feature extraction / retrain):
   ```
   python train_and_save_model.py
   ```
   This creates `models/gearbox_model.pkl` and related files.
4. Launch the dashboard:
   ```
   streamlit run app.py
   ```
   It opens at `http://localhost:8501`.

## Pages
- **Overview** — project summary, model accuracy, confusion matrix.
- **Predict** — upload a raw `.txt` vibration file (or click a demo
  button) to get a live prediction with confidence, a signal plot,
  and key feature values. Every prediction is saved to the database.
- **Explore Dataset** — interactive plots comparing feature
  distributions between healthy and broken-tooth samples, across
  load levels.
- **Prediction History** — table + pie chart of everything logged to
  the database so far, with a button to clear it.

## Database Integration

The app uses **SQLite** (`db_utils.py`) — a file-based database
built into Python, so there's nothing extra to install or host. Every
prediction made on the "Predict" page is written to a `predictions`
table (timestamp, filename, predicted condition, confidence, and key
feature values). The "Prediction History" page reads this table back.

### Upgrading to a real production database later
If you eventually want this backed by PostgreSQL/MySQL (e.g. for a
multi-user deployment), you only need to change `db_utils.py`:
- Replace `sqlite3.connect(...)` in `get_connection()` with, e.g.,
  `psycopg2.connect(host=..., dbname=..., user=..., password=...)`
  for PostgreSQL.
- The `CREATE TABLE`, `INSERT`, and `SELECT` statements stay almost
  identical (minor SQL dialect differences, e.g. `SERIAL` instead of
  `AUTOINCREMENT` for PostgreSQL primary keys).
- Everything else (`app.py`, `feature_utils.py`) stays unchanged,
  since they only ever call the functions in `db_utils.py`, never SQL
  directly.

## Notes for the Project Report
- The "Predict" page currently extracts ONE whole-file feature
  summary per upload (matches the simple approach from Week 1). The
  underlying model was trained on **windowed** features (978 samples)
  for better generalization — see `train_and_save_model.py`.
- `st.cache_resource` / `st.cache_data` are used so the model and
  training data are loaded from disk only once, not on every page
  interaction — keeps the app responsive.
