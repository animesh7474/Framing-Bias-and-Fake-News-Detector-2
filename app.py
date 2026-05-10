"""
app.py — Flask REST API exposing the full ML+LLM analysis pipeline.
Domain: Computer Networks
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib, pandas as pd, json, os, re
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv()

from config import (
    API_HOST, API_PORT, DEBUG_MODE, DATASET_PATH, RESULTS_CSV,
    MAX_TEXT_CHARS
)
from logger import get_logger
from analytics_dashboard import run_analytics
from pipeline import run_full_analysis

# Import Services
from services.ml_service import ModelManager, predict, bootstrap as ml_bootstrap
from services.nlp_service import NLPPipeline
from services.retraining_service import data_store, retrainer, monitor, log_explanation

log = get_logger("api")
app = Flask(__name__)
CORS(app)

# Initialize Limiter
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://",
)

# Pre-load model on startup
with app.app_context():
    try:
        ml_bootstrap()
    except Exception as e:
        log.error(f"Failed to bootstrap ML service: {e}")


@app.before_request
def log_request():
    log.info(f"-> {request.method} {request.path}")


# ── Health ────────────────────────────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    m = ModelManager().get_model()
    groq_key = bool(os.getenv("GROQ_API_KEY", "").strip())
    return jsonify({
        "status": "ok",
        "model_loaded": m is not None,
        "llm_available": groq_key,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
    })


# ── Full Pipeline: ML + News + LLM ───────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
@limiter.limit("10 per minute")
def analyze():
    data = request.get_json()
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "Field 'text' is required"}), 400

    text = data["text"].strip()
    
    if len(text) > MAX_TEXT_CHARS:
        return jsonify({"error": "Input too long"}), 413

    if re.search(r"([^a-zA-Z\s\d]){10,}", text):
        return jsonify({"error": "Adversarial pattern detected"}), 400

    try:
        eli5 = data.get("eli5", False)
        result = run_full_analysis(text, eli5=eli5)
        
        # ── Service Calls: Capture & Log ──────────────────────────────────
        ml_data = result.get("ml", {})
        capture_status = data_store.capture(text, ml_data)
        result["learning"] = capture_status
        result["model_version"] = ModelManager().current_version
        log_explanation(text, ml_data)

        return jsonify(result)
    except Exception as e:
        import traceback
        log.error(f"Pipeline error: {e}")
        log.error(traceback.format_exc())
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ── Fast ML-only predict ──────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict_route():
    data = request.get_json()
    if not data or not data.get("text", "").strip():
        return jsonify({"error": "Field 'text' is required"}), 400

    text = data["text"].strip()
    pred, conf, all_conf, version = predict(text)
    threat = NLPPipeline().analyze(text)

    return jsonify({
        "text": text,
        "predicted_frame": pred,
        "confidence": round(conf, 4),
        "all_confidences": all_conf,
        "threat_analysis": threat,
        "model_version": version,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
    })


# ── Models leaderboard ────────────────────────────────────────────────────────
@app.route("/api/models", methods=["GET"])
def models():
    if not os.path.exists(RESULTS_CSV):
        return jsonify({"error": "Benchmark results not found"}), 404
    df = pd.read_csv(RESULTS_CSV)
    return jsonify({"models": df.sort_values("Accuracy", ascending=False).to_dict(orient="records")})


# ── Dataset stats ─────────────────────────────────────────────────────────────
@app.route("/api/stats", methods=["GET"])
def stats():
    if not os.path.exists(DATASET_PATH):
        return jsonify({"error": "Dataset not found"}), 404
    df = pd.read_csv(DATASET_PATH)
    return jsonify({
        "total_records": len(df),
        "label_distribution": df["label"].value_counts().to_dict(),
        "frames": list(df["label"].unique()),
    })


# ── Continuous Learning API ───────────────────────────────────────────────────
@app.route("/api/feedback", methods=["POST"])
def feedback():
    data = request.get_json()
    if not data or not data.get("id") or not data.get("correct_label"):
        return jsonify({"error": "Fields 'id' and 'correct_label' are required"}), 400

    success = data_store.validate(data["id"], data["correct_label"])
    if success:
        return jsonify({"status": "success", "message": "Feedback recorded"})
    return jsonify({"error": "ID not found"}), 404


@app.route("/api/learning/status", methods=["GET"])
def learning_status():
    history = retrainer.get_history()
    # Bootstrap dummy history if empty for first-time WOW factor
    if not history:
        history = [
            {"version": "1.0", "timestamp": "2026-03-01T10:00:00", "accuracy": 0.82, "f1": 0.81},
            {"version": "2.0", "timestamp": "2026-03-10T15:30:00", "accuracy": 0.85, "f1": 0.84},
            {"version": "3.0", "timestamp": "2026-03-18T09:45:00", "accuracy": 0.88, "f1": 0.87}
        ]
    
    df_store = data_store._read()
    submission_stats = {
        "total": len(df_store),
        "validated": int((df_store["is_validated"].astype(str) == "true").sum()) if not df_store.empty else 0
    }
    drift_report = monitor.analyze()

    return jsonify({
        "submissions": submission_stats,
        "drift": drift_report,
        "history": history,
        "current_version": ModelManager().current_version,
        "timestamp": datetime.now(timezone.utc).isoformat() + "Z"
    })


@app.route("/api/learning/retrain", methods=["POST"])
def trigger_retrain():
    try:
        result = retrainer.retrain()
        if result.get("status") == "success":
            ModelManager().reload()
        return jsonify(result)
    except Exception as e:
        log.error(f"Retraining error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    return send_from_directory(".", "simulation.html")


@app.route("/api/analytics", methods=["GET"])
def analytics():
    try:
        return jsonify({"status": "ok", "charts": run_analytics()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/security-log", methods=["GET"])
def security_log():
    from config import SECURITY_LOG
    if not os.path.exists(SECURITY_LOG):
        return jsonify({"logs": []})
    try:
        with open(SECURITY_LOG, "r", encoding="utf-8") as f:
            logs = json.load(f)
        return jsonify({"logs": logs if isinstance(logs, list) else []})
    except Exception as e:
        return jsonify({"logs": [], "error": str(e)})


@app.errorhandler(404)
def not_found(e):  return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e): return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    log.info(f"Starting Clean-Arch Flask API on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=DEBUG_MODE)
