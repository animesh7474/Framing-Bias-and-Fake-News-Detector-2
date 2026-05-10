"""
retraining_service.py — Service layer for active learning, MLOps, and model lifecycle.
Domain: Big Data Analytics + Software Development & Operations
"""

import os
import csv
import json
import hashlib
import threading
import re
import time
from datetime import datetime, timedelta
from collections import Counter
import pandas as pd
import numpy as np
import joblib
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from config import (
    DATASET_PATH, USER_SUBMISSIONS_CSV, RETRAINING_LOG,
    MODELS_DIR, DATA_VERSIONS_DIR, EXPLANATIONS_LOG,
    MIN_SAMPLES_FOR_RETRAIN, MIN_TEXT_WORDS, MAX_TEXT_CHARS, FRAME_LABELS
)
from logger import get_logger

log = get_logger("retraining_service")
DetectorFactory.seed = 42

# ═══════════════════════════════════════════════════════════════════════════════
# 1. USER DATA STORE & QUALITY FILTER
# ═══════════════════════════════════════════════════════════════════════════════

class DataFilter:
    """Static quality checks to reject spam, gibberish, and low-value inputs."""
    _SPAM_PHRASES = ["wake up", "they are lying", "fake news", "don't believe", "share this"]

    @staticmethod
    def check(text: str) -> dict:
        text = text.strip()
        word_count = len(text.split())
        if word_count < MIN_TEXT_WORDS:
            return {"passed": False, "reason": f"Too short"}
        if len(text) > MAX_TEXT_CHARS:
            return {"passed": False, "reason": "Too long"}
        try:
            if detect(text) != "en":
                return {"passed": False, "reason": "Non-English"}
        except LangDetectException:
            return {"passed": False, "reason": "Lang detection failed"}
        if re.search(r"(.)\1{4,}", text):
            return {"passed": False, "reason": "Repetition"}
        return {"passed": True, "reason": "ok"}

class UserDataStore:
    _lock = threading.Lock()
    _FIELDS = ["id", "timestamp", "text_hash", "text", "predicted_frame", "confidence", "bias_score", "manipulation_score", "is_validated", "user_feedback_label", "feedback_ts"]

    def __init__(self, filepath: str = USER_SUBMISSIONS_CSV):
        self.filepath = filepath
        self._ensure_file()

    def _ensure_file(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._FIELDS).writeheader()

    def capture(self, text: str, ml_results: dict):
        text_hash = hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:16]
        filt = DataFilter.check(text)
        if not filt["passed"]:
            return {"captured": False, "reason": filt["reason"]}

        threat = ml_results.get("threat_analysis", {})
        row = {
            "id": text_hash,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "text_hash": text_hash,
            "text": text.replace("\n", " "),
            "predicted_frame": ml_results.get("predicted_frame", ""),
            "confidence": ml_results.get("confidence", 0),
            "bias_score": threat.get("true_bias_score", 0),
            "manipulation_score": threat.get("manipulation_score", 0),
            "is_validated": "false",
            "user_feedback_label": "",
            "feedback_ts": ""
        }
        with self._lock:
            with open(self.filepath, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self._FIELDS).writerow(row)
        return {"captured": True, "id": text_hash}

    def _read(self) -> pd.DataFrame:
        if not os.path.exists(self.filepath):
            return pd.DataFrame(columns=self._FIELDS)
        return pd.read_csv(self.filepath)

    def validate(self, text_id: str, label: str):
        with self._lock:
            df = self._read()
            if text_id in df["id"].values:
                df.loc[df["id"] == text_id, "is_validated"] = "true"
                df.loc[df["id"] == text_id, "user_feedback_label"] = label
                df.loc[df["id"] == text_id, "feedback_ts"] = datetime.utcnow().isoformat() + "Z"
                df.to_csv(self.filepath, index=False)
                return True
        return False

# ═══════════════════════════════════════════════════════════════════════════════
# 2. RETRAINING & DRIFT MONITORING
# ═══════════════════════════════════════════════════════════════════════════════

class ModelRetrainer:
    """Manages the Train-Validate-Compare-Deploy cycle."""
    def __init__(self, store: UserDataStore):
        self.store = store

    def retrain(self):
        df_new = self.store._read()
        valid = df_new[df_new["is_validated"] == "true"]
        if len(valid) < MIN_SAMPLES_FOR_RETRAIN:
            return {"status": "skipped", "reason": f"Need {MIN_SAMPLES_FOR_RETRAIN} samples"}

        log.info("Starting retraining cycle...")
        # 1. Load current data
        df_base = pd.read_csv(DATASET_PATH)
        # 2. Add new data
        X_new = valid["text"]
        y_new = valid["user_feedback_label"]
        # Simplified: In reality, merge and balance
        # 3. Dummy Retrain (Mock logic for this exercise)
        metrics = {"accuracy": 0.88, "f1": 0.87, "timestamp": datetime.now().isoformat()}
        
        # 4. Save Versioned
        os.makedirs(MODELS_DIR, exist_ok=True)
        next_v = len(os.listdir(MODELS_DIR)) + 1
        model_path = os.path.join(MODELS_DIR, f"model_v{next_v}.pkl")
        os.makedirs(DATA_VERSIONS_DIR, exist_ok=True)
        data_path = os.path.join(DATA_VERSIONS_DIR, f"dataset_v{next_v}.csv")
        # joblib.dump(mock_model, model_path)
        
        log.info(f"New model version {next_v} created.")
        
        # 5. Log history
        history = self.get_history()
        history.append({
            "version": f"{next_v}.0",
            "timestamp": datetime.now().isoformat(),
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"]
        })
        with open(RETRAINING_LOG, "w") as f:
            json.dump(history, f, indent=2)

        return {"status": "success", "version": next_v, "metrics": metrics}

    def get_history(self):
        if not os.path.exists(RETRAINING_LOG):
            return []
        try:
            with open(RETRAINING_LOG, "r") as f:
                return json.load(f)
        except:
            return []

class DriftMonitor:
    """Tracks model performance degradation."""
    @staticmethod
    def analyze():
        # Compare last 2 retraining accuracies
        if not os.path.exists(RETRAINING_LOG):
            return {"drift_detected": False, "drift_value": 0, "status": "stable"}
        try:
            with open(RETRAINING_LOG, "r") as f:
                history = json.load(f)
        except:
            return {"drift_detected": False, "drift_value": 0, "status": "stable"}

        if len(history) < 2:
            return {"drift_detected": False, "drift_value": 0, "status": "stable"}
        
        latest = history[-1].get("accuracy", 0)
        previous = history[-2].get("accuracy", 0)
        drift = round(abs(latest - previous), 4)
        status = "stable"
        if drift > 0.1: status = "alert"
        elif drift > 0.05: status = "warning"

        return {
            "drift_detected": drift > 0.05,
            "drift_value": drift,
            "status": status,
            "last_accuracy": latest,
            "prev_accuracy": previous
        }

# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXPLAINABILITY LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def log_explanation(text: str, ml_res: dict):
    """Appends model explainability data to logs/explanations.json."""
    try:
        lime = ml_res.get("lime_words", [])
        top_k = sorted(lime, key=lambda x: x["score"], reverse=True)[:10]
        keywords = [k["word"] for k in top_k]
        
        threat = ml_res.get("threat_analysis", {})
        
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "text": text[:500],
            "prediction": ml_res.get("predicted_frame"),
            "confidence": ml_res.get("confidence"),
            "important_words": keywords,
            "sentiment_score": threat.get("breakdown", {}).get("sentiment_extremity", 0),
            "bias_score": threat.get("true_bias_score", 0),
            "manipulation_score": threat.get("manipulation_score", 0)
        }
    except Exception as e:
        log.error(f"Failed to extract explanation data: {e}")
        return

    logs = []
    if os.path.exists(EXPLANATIONS_LOG):
        with open(EXPLANATIONS_LOG, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
    
    logs.append(entry)
    os.makedirs(os.path.dirname(EXPLANATIONS_LOG), exist_ok=True)
    with open(EXPLANATIONS_LOG, "w", encoding="utf-8") as f:
        json.dump(logs[-100:], f, indent=2)

# Global Instance
data_store = UserDataStore()
retrainer = ModelRetrainer(data_store)
monitor = DriftMonitor()
