"""
pipeline.py — 4-stage analysis pipeline orchestrator.
Domain: NLP + Big Data + Computer Networks + AI Cybersecurity
"""

from datetime import datetime, timezone
from logger import get_logger

# Import Services
from services.ml_service import predict, get_explainable_keywords
from services.nlp_service import NLPPipeline
from services.news_service import fetch_related_news
from services.llm_service import LLMService

log = get_logger("pipeline")

# Initialize persistent service instances
_threat_analyzer = NLPPipeline()
_llm_service = LLMService()

def run_full_analysis(text: str, eli5: bool = False) -> dict:
    """
    Full 4-stage pipeline (Sync Orchestration):
      Stage 1 — ML Service: classification + confidence + keywords
      Stage 2 — News Service: sync DDG search for context
      Stage 3 — NLP Service: deep threat analysis (manipulation/bias)
      Stage 4 — LLM Service: sync comparative report
    """
    started = datetime.now(timezone.utc)
    log.info(f"Pipeline [SERVICE-LAYER] started: {text[:60]}")

    # ── Stage 1: ML Inference ────────────────────────────────────────────────
    pred, conf, all_conf, model_version = predict(text)
    lime_words = get_explainable_keywords(text, pred)

    # ── Stage 2: NLP Threat Analysis ─────────────────────────────────────────
    threat = _threat_analyzer.analyze(text)

    ml_result = {
        "predicted_frame":  pred,
        "confidence":       round(conf, 4),
        "all_confidences":  all_conf,
        "lime_words":       lime_words,
        "threat_analysis":  threat,
    }

    # ── Stage 3: News Context (SYNC) ─────────────────────────────────────────
    log.info("[Stage 3] Fetching news context...")
    articles = fetch_related_news(text, max_results=4)

    # ── Stage 4: LLM Analysis (SYNC) ─────────────────────────────────────────
    log.info("[Stage 4] Generating LLM comparison report...")
    llm_result = _llm_service.analyze_article(text, pred, conf, articles, eli5=eli5)

    # ── Final Composition ───────────────────────────────────────────────────
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    log.info(f"Pipeline [SERVICE-LAYER] complete in {elapsed:.2f}s")

    return {
        "text":          text,
        "timestamp":     started.isoformat() + "Z",
        "elapsed_s":     round(elapsed, 2),
        "ml":            ml_result,
        "related_news":  articles,
        "llm_analysis":  llm_result,
    }
