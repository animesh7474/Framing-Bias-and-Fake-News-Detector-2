# 🛡️ Hybrid AI Intelligence Dashboard — Framing Bias & Fake News Detector

A production-grade, multi-layered AI system for detecting **Framing Bias**, **Fake News**, and **Narrative Manipulation** in real-time. Built as a B.Tech Capstone Project.

---

## 🚀 Live Demo

> Deploy on [Render](https://render.com) for free — see [Deployment](#-deployment) below.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **4-Stage Hybrid AI Pipeline** | ML Classification → NLP Threat Scoring → Live News Context → LLM Comparative Verdict |
| **Local ML Model** | Logistic Regression + TF-IDF for high-speed frame classification (83.2% accuracy) |
| **Cloud LLM Integration** | LLaMA 3.3 70B via GroqCloud for deep semantic reasoning |
| **Live Context Retrieval** | Async DuckDuckGo news search to validate claims against real-world events |
| **Explainable AI (XAI)** | LIME-based keyword highlighting for transparent decision-making |
| **Security Intelligence Log** | Persistent threat logging for forensic audit trails |
| **Interactive Dashboard** | Responsive dark-mode UI with real-time analysis visualizations |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────┐
│              Web Dashboard (UI)              │
│         HTML5 / CSS3 / Vanilla JS            │
└──────────────────┬───────────────────────────┘
                   │  REST API (JSON)
┌──────────────────▼───────────────────────────┐
│           Flask Backend (app.py)             │
│         Pipeline Orchestrator                │
└──┬───────┬───────────┬───────────┬───────────┘
   │       │           │           │
   ▼       ▼           ▼           ▼
┌──────┐ ┌──────┐ ┌─────────┐ ┌──────────┐
│  ML  │ │ NLP  │ │  News   │ │   LLM    │
│Model │ │Score │ │ Search  │ │ LLaMA3.3 │
└──────┘ └──────┘ └─────────┘ └──────────┘
```

---

## 🛠️ Tech Stack

- **Backend:** Python 3.11, Flask 3.0, Gunicorn
- **Frontend:** HTML5, CSS3, Vanilla JavaScript, Chart.js, Feather Icons
- **ML/NLP:** Scikit-learn, LIME, VADER, TextBlob, spaCy
- **Cloud AI:** Groq SDK (LLaMA 3.3 70B)
- **News API:** DuckDuckGo Search SDK

---

## ⚡ Quick Start (Local)

```bash
# 1. Clone the repository
git clone https://github.com/animesh7474/Framing-Bias-and-Fake-News-Detector-2.git
cd Framing-Bias-and-Fake-News-Detector-2

# 2. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Set up environment variables
# Create a .env file with:
# GROQ_API_KEY=your_groq_api_key_here

# 5. Generate dataset & train the model
python dataset_generator.py
python framing_bias_detector.py

# 6. Run the server
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## 🌐 Deployment

This project is deployment-ready for **Render (Free Tier)**:

1. Push this repo to GitHub.
2. Go to [Render.com](https://render.com) → **New > Web Service** → Connect this repo.
3. Render auto-detects the `Dockerfile` and builds.
4. Add `GROQ_API_KEY` in **Environment Variables**.
5. Deploy — your live URL will be `https://your-app.onrender.com`.

---

## 📂 Project Structure

```
project/
├── app.py                  # Flask REST API
├── pipeline.py             # 4-Stage Orchestrator
├── config.py               # Centralized Configuration
├── services/
│   ├── ml_service.py       # Logistic Regression Classifier
│   ├── nlp_service.py      # Lexical Threat Scorer
│   ├── news_service.py     # Live News Context Fetcher
│   ├── llm_service.py      # LLaMA 3.3 Integration
│   └── retraining_service.py  # Active Learning Loop
├── simulation.html         # Interactive Dashboard UI
├── Dockerfile              # Production Container
└── requirements.txt        # Python Dependencies
```

---

## 📊 Results

| Metric | Value |
|---|---|
| **Accuracy** | 83.2% |
| **F1-Score** | 0.831 |
| **Precision** | 0.832 |
| **End-to-End Latency** | ~3.4 seconds |

---

## 👤 Author

**Animesh Hole**
- GitHub: [@animesh7474](https://github.com/animesh7474)

---

## 📄 License

This project is for academic and educational purposes.
