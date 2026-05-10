"""
config.py — Centralized configuration for all modules.
Domain: Software Development & Operations
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH   = os.path.join(BASE_DIR, "dataset.csv")
MODEL_PATH     = os.path.join(BASE_DIR, "framing_bias_model.pkl")
REPORTS_DIR    = os.path.join(BASE_DIR, "reports")
SECURITY_LOG   = os.path.join(BASE_DIR, "security_log.json")
RESULTS_CSV    = os.path.join(BASE_DIR, "model_comparison_results.csv")

# ─── Server ───────────────────────────────────────────────────────────────────
API_HOST       = "0.0.0.0"
API_PORT       = int(os.environ.get("PORT", 5000))
DEBUG_MODE     = False

# ─── Model ────────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 5000
TEST_SIZE          = 0.2
RANDOM_STATE       = 42

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FILE  = os.path.join(BASE_DIR, "app.log")

# ─── Streaming ────────────────────────────────────────────────────────────────
STREAM_DELAY_SECONDS = 0.5

# ─── Frames ───────────────────────────────────────────────────────────────────
FRAME_LABELS = ["Economic", "Political", "Social", "Security", "Environment"]

# ─── LLM ──────────────────────────────────────────────────────────────────────
GROQ_MODEL       = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS   = 1200
LLM_TEMPERATURE  = 0.3
NEWS_MAX_RESULTS = 4

# ─── Continuous Learning ──────────────────────────────────────────────────────
USER_SUBMISSIONS_CSV    = os.path.join(BASE_DIR, "user_submissions.csv")
RETRAINING_LOG          = os.path.join(BASE_DIR, "retraining_log.json")
MODEL_BACKUP_DIR        = os.path.join(BASE_DIR, "model_backups")
MODELS_DIR               = os.path.join(BASE_DIR, "models")
DATA_VERSIONS_DIR        = os.path.join(BASE_DIR, "data_versions")
MIN_SAMPLES_FOR_RETRAIN = 50
RETRAIN_CONFIDENCE_UPPER = 0.85
MIN_TEXT_WORDS          = 20
MAX_TEXT_CHARS           = 5000
EXPLANATIONS_LOG        = os.path.join(BASE_DIR, "logs", "explanations.json")
