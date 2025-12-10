# voice-agent-tool-router-using-spaCy-Library

This repository contains a **Python-based intelligent tool-routing system** for a voice assistant, primarily designed for **automotive and infotainment use cases**.

The system converts natural language commands into **ranked tool selections and executable task plans** using a combination of:
- Text embeddings
- Cosine similarity
- Learning-to-Rank (LTR)
- Real Named Entity Recognition (spaCy)
- Rule-based fallbacks

---

## 🚗 What This Project Does

Given a user query like:

> "Navigate from Delhi to Agra and tell me the weather there"

The system:
1. Understands the intent
2. Identifies relevant tools
3. Ranks them using LTR
4. Extracts entities (locations, contacts, etc.)
5. Generates a structured execution plan

---

## 🧠 Core Architecture

**Stage 1: Tool Selection**
- Query is converted into an embedding
- Compared against stored tool embeddings
- Ranked using an LTR model (XGBoost or mock fallback)

**Stage 2: Task Planning**
- Uses **real NER via spaCy**
- Applies regex and heuristics as fallback
- Generates an executable tool invocation plan

---

## 🔧 Supported Tool Categories

- GPS Navigation
- Weather Forecasting
- Hands-Free Calling
- Music Playback
- Vehicle Diagnostics
- Traffic Reporting
- Climate Control
- Emergency Services
- EV & Battery Monitoring

(Tool list is fully extensible.)

---

## 🧪 Example Queries

```text
Navigate to Chennai
Call Rahul
Play Arijit Singh songs
What is the weather in Bangalore?
Route from Delhi to Jaipur

⚙️ How to Run
1️⃣ Install Dependencies
pip install numpy requests
pip install spacy
pip install xgboost   # optional
python -m spacy download en_core_web_sm

2️⃣ Set API Key

Update this line in the code:

API_KEY = "YOUR_GOOGLE_GEMINI_API_KEY"

3️⃣ Run the Program
python main.py

➕ Adding New Tools

In interactive mode, type:

add


You can dynamically add new tools.
Their embeddings are automatically generated and stored.

💾 Persistent Storage

Tool metadata and embeddings are saved in:

tools_db_simulated_final.json


This avoids recomputing embeddings on every run.
