# 🚀 VIORA — Production Deployment Guide (Vercel, Render, Railway, Hugging Face)

This guide details how to deploy VIORA to cloud platforms including **Vercel**, **Render**, **Railway**, and **Hugging Face Spaces**.

---

## 📊 1. Pre-Deployment Repository Audit

- **Total Repository Size**: **38.68 MB** (Excludes `.venv` and raw datasets via `.gitignore`).
- **Model Artifact Sizes**:
  - `models/voice_emotion/expanded_model.pkl`: **5.41 MB**
  - `models/text_emotion/tfidf_model.pkl`: **9.91 MB**
- **Environment Requirements**: Python 3.10+ with `librosa`, `scikit-learn`, `joblib`, `soundfile`, `scipy`, `numpy`, `pandas`.

---

## ⚡ 2. Deploying on Vercel

Vercel is fully supported using Vercel Python Serverless Functions (`@vercel/python` engine configured in [`vercel.json`](../vercel.json) and [`api/index.py`](../api/index.py)).

### Option A: Pure Vercel (Frontend + Serverless Python API)

1. Install the Vercel CLI or link via GitHub:
   ```bash
   npx vercel
   ```
2. Vercel automatically detects `vercel.json`:
   - Static HUD Frontend served from `ui/`
   - Python ML API served from `api/index.py` (`/api/predict` & `/api/multimodal`)

> [!NOTE]
> **Vercel Memory & Timeout Consideration**: On Vercel's free hobby tier, serverless function execution is capped at **10 seconds max duration**. Text predictions execute in ~0.1s. For heavy audio feature extraction (`librosa` YIN algorithm on long WAV files), Render or Railway provide dedicated non-serverless RAM/CPU resources.

### Option B: Hybrid Vercel (Frontend on Vercel + Backend on Render/Railway)

1. Deploy the Python ML server (`server.py`) to Render or Railway (`https://viora-backend.onrender.com`).
2. Deploy the static UI (`ui/`) to Vercel.
3. Update `ui/app.js` API fetch URL to point to your deployed backend URL.

---

## ☁️ 3. Deploying on Render (Recommended for Full Python Servers)

1. **Connect GitHub Repository**:
   - Log into [Render Dashboard](https://dashboard.render.com).
   - Click **New +** $\rightarrow$ **Web Service**.
   - Select your `VIORA` GitHub repository.

2. **Configure Web Service**:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python server.py`
   - **Instance Type**: Free Tier.

3. **Verify Deployment**:
   - Render serves the complete web HUD & API at `https://viora-app.onrender.com`.

---

## 🚂 4. Deploying on Railway

1. Log into [Railway.app](https://railway.app).
2. Click **New Project** $\rightarrow$ **Deploy from GitHub repo**.
3. Set start command to `python server.py`.

---

## 🎯 5. Deploying on Hugging Face Spaces

1. Create a Docker or Streamlit Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Push your repository to HF Spaces.

---

## 🧪 6. Post-Deployment Verification Checklist

Verify:
- [x] `https://<deployed-url>/` loads Sci-Fi HUD dashboard.
- [x] Text message predictions return 8-class probabilities & empathetic response.
- [x] Audio file upload / live mic recording processes multimodal fusion.
- [x] "How VIORA Understood You" breakdown displays Voice, Text, and Fused badges cleanly.
