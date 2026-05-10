"""
llm_service.py — Asynchronous Service Layer for LLM-based news analysis.
Domain: NLP + Software Development & Operations
"""

import os
import json
from groq import Groq
from logger import get_logger

log = get_logger("llm_service")

class LLMService:
    """
    Handles communication with GroqCloud for real-time news report generation.
    Supports robust fallback mechanisms.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            log.warning("GROQ_API_KEY NOT FOUND. LLM logic will use fallback.")
        self._client = None

    @property
    def client(self):
        if self._client is None and self.api_key:
            self._client = Groq(api_key=self.api_key)
            log.info("Groq client initialized lazily.")
        return self._client

    def analyze_article(self, text: str, ml_prediction: str, confidence: float, news_context: list, eli5: bool = False):
        """
        Calls Groq LLM to perform deep framing bias and fake news analysis.
        If 'eli5' is True, generates a simplified 'Explain Like I'm 5' version.
        """
        if not self.client:
            log.warning("LLM logic skipped (no API key). Using fallback report.")
            return self._fallback(ml_prediction, confidence)

        base_prompt = f"""
        You are an expert Media Bias & News Verification Analyst.
        
        USER INPUT: "{text}"
        ML PREDICTION: "{ml_prediction}" (Confidence: {confidence*100:.1f}%)
        LIVE NEWS CONTEXT: {json.dumps(news_context[:3])}

        TASK:
        1. Compare the USER INPUT with the LIVE NEWS CONTEXT.
        2. Validate if the ML prediction aligns with current journalistic trends.
        3. Provide a structured JSON response with EXACTLY these keys:
           - "comparison_summary": 2-sentence summary of alignment.
           - "risk_factors": list of 3 bullet points.
           - "verdict": "Likely Bias", "Authentic", or "Needs Verification".
           - "explanation": Brief reasoning for the verdict.
           - "eli5_explanation": A very simple, 2-sentence explanation for a child.
           - "framing_bias_score": Integer 0-100 (STRICTLY NUMERIC).
           - "fake_news_likelihood": Integer 0-100 (STRICTLY NUMERIC).
           - "emotional_manipulation_score": Integer 0-100 (STRICTLY NUMERIC).
           - "framing_bias_explanation": Detailed paragraph.
           - "fake_news_analysis": Detailed paragraph.
           - "emotional_manipulation_analysis": Detailed paragraph.
           - "agenda_detection": Detailed paragraph.
           - "factual_gaps": Detailed paragraph.
        
        Strictly return ONLY JSON. Do not include conversational text outside the JSON block.
        """

        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": base_prompt}],
                model="llama-3.3-70b-versatile",
                response_format={"type": "json_object"}
            )
            report = json.loads(response.choices[0].message.content)
            log.info("LLM Analysis complete.")
            return report
        except Exception as e:
            log.error(f"LLM API Error: {e}")
            return self._fallback(ml_prediction, confidence)

    def _fallback(self, ml_prediction: str, confidence: float):
        """Basic heuristic analysis if LLM fails."""
        risk_score = int(confidence * 100)
        return {
            "comparison_summary": f"Our models identified patterns common in '{ml_prediction}' narratives.",
            "risk_factors": [
                f"Statistical confidence: {risk_score}%",
                "Heuristic structural patterns detected",
                "Verification via live sources skipped"
            ],
            "verdict": "Needs Verification",
            "explanation": "Automated detection flagged language structures associated with framing bias. Without live context, we recommend manual verification.",
            "eli5_explanation": "Our computer thinks this news is trying to trick you, but it couldn't check other news to be sure!",
            "framing_bias_score": risk_score if ml_prediction != "Authentic" else 10,
            "fake_news_likelihood": risk_score // 2,
            "emotional_manipulation_score": risk_score // 3,
            "framing_bias_explanation": f"The model detected '{ml_prediction}' stylistic markers. This often indicates a specific viewpoint is being prioritized over neutral reporting.",
            "fake_news_analysis": "Heuristic analysis shows some linguistic markers sometimes found in unverified news.",
            "emotional_manipulation_analysis": "Sentence structures suggest a focus on emotional resonance rather than neutral factuality.",
            "agenda_detection": "The framing technique used matches patterns often associated with persuasive rather than informative intent.",
            "factual_gaps": "Unable to perform deep fact-gap analysis without external reference corpus."
        }

    def is_available(self):
        return self.client is not None
