# 🚀 VIORA — Production Deployment Guide

This guide details how to deploy VIORA (Frontend HUD + Backend Multimodal ML Engine) to cloud platforms such as **Render**, **Railway**, or **Hugging Face Spaces**.

---

## 📊 1. Pre-Deployment Repository Audit

- **Total Repository Size**: **38.68 MB** (Excludes `.venv` and raw datasets via `.gitignore`).
- **Model Artifact Sizes**:
  - `models/voice_emotion/expanded_model.pkl`: **5.41 MB**
  - `models/text_emotion/tfidf_model.pkl`: **9.91 MB**
- **Environment Requirements**: Python 3.10+ with `librosa`, `scikit-learn`, `joblib`, `soundfile`, `scipy`, `numpy`, `pandas`.

---

## ☁️ 2. Option A: Deploying on Render (Recommended)

1. **Connect GitHub Repository**:
   - Log into [Render Dashboard](https://dashboard.render.com).
   - Click **New +** $\rightarrow$ **Web Service**.
   - Select your `VIORA` GitHub repository.

2. **Configure Web Service**:
   - **Environment**: `Python 3`
   - **Build Command**:
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     python server.py
     ```
   - **Instance Type**: Free Tier (512 MB RAM / 0.1 CPU is sufficient).

3. **Verify Deployment**:
   - Render will deploy the web server at `https://viora-app.onrender.com`.
   - Access the interactive HUD in any web browser!

---

## 🚂 3. Option B: Deploying on Railway

1. **New Project**:
   - Log into [Railway.app](https://railway.app).
   - Click **New Project** $\rightarrow$ **Deploy from GitHub repo**.
   - Select `VIORA`.

2. **Environment Variables & Start Command**:
   - Set start command to `python server.py`.
   - Railway will auto-detect Python and install dependencies from `requirements.txt`.

---

## 🎯 4. Option C: Deploying on Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces) selecting the **Docker** or **Streamlit/Gradio** SDK.
2. Push your repository to HF Spaces.
3. Your deployment link `https://huggingface.co/spaces/<user>/viora` will serve the multimodal HUD.

---

## 🧪 5. Post-Deployment Verification Checklist

Verify:
- [x] `https://<deployed-url>/` loads Sci-Fi HUD dashboard.
- [x] Text message predictions return 8-class probabilities & empathetic response.
- [x] Audio file upload / live mic recording processes multimodal fusion.
- [x] "How VIORA Understood You" breakdown displays Voice, Text, and Fused badges cleanly.
