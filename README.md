# 🎙️ VIORA — Multimodal Emotion-Aware Conversational AI

VIORA is an end-to-end **Multimodal Emotion-Aware Conversational System** that perceives emotion from both **Human Speech (Voice Audio)** and **Text Inputs**, fuses probability distributions, and responds with supportive, non-medical empathetic dialogue.

---

## 🌟 Key Features

- 🎧 **Speech Emotion Recognition (SER)**: 342 V2 acoustic features (fast YIN pitch $F_0$, MFCC $\Delta/\Delta^2$, RMS energy, spectral contrast/centroid/rolloff, chroma) classified by a Champion RBF SVM trained on 2,080 real audio clips (**RAVDESS + CREMA-D**).
- 📝 **Text Emotion Classification**: TF-IDF ($100\text{k}$ n-grams) + Logistic Regression trained on 61,627 samples (**dair-ai/emotion + Google GoEmotions**), achieving **68.99% validation accuracy**.
- 🧮 **Probability-Level Multimodal Fusion**: Combines voice and text probability distributions seamlessly across shared emotion space (`angry`, `calm`, `disgust`, `fearful`, `happy`, `neutral`, `sad`, `surprised`). Supports `voice+text`, `voice_only`, and `text_only` modes without crashes.
- 💬 **Empathetic Response Generator**: Generates supportive, non-medical conversational dialogue ("It sounds like you may be feeling..."). Integrates Google Gemini API with a robust offline fallback response engine.
- 💻 **Sci-Fi HUD Web Dashboard**: Modern dark-theme web dashboard (`http://localhost:8000`) with audio visualizer, live microphone recording, text prompt input, radar chart visualization, and real-time response box.

---

## 📊 Performance Benchmarks

### 1. Speech Emotion Recognition (Held-Out RAVDESS Actors 19–24, 360 Untouched Clips)

| Model Variant | Dataset | Features | Accuracy | Macro F1 | Balanced Acc |
| :--- | :--- | :---: | :---: | :---: | :---: |
| V1 Baseline | Real RAVDESS (1,080 clips) | 162 | 53.89% | 53.90% | 53.89% |
| V2 Representation | Real RAVDESS (1,080 clips) | 342 | 54.44% | 52.70% | 52.92% |
| Tuned Champion | Real RAVDESS (1,080 clips) | 342 | 53.06% | 51.45% | 51.56% |
| **Final Expanded Model** | **RAVDESS + CREMA-D (2,080 clips)** | **342** | **54.17%** | **53.69%** | **53.39%** |

### 2. Text Emotion Recognition (Held-Out Test Set, 61,627 Samples)

| Metric | Score |
| :--- | :---: |
| **Validation Accuracy** | **68.99%** |
| **Validation Macro F1** | **64.11%** |

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.10+
- Virtual environment (`.venv`)

### Installation Steps

```bash
# 1. Activate virtual environment
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 💻 How to Run VIORA

### 1. Run Demo Multimodal CLI Pipeline
```bash
python main.py --text "I am so happy that I finally passed my exam!"
```

### 2. Launch Web HUD Server
```bash
python main.py --server
# or
python server.py
```
Open **`http://localhost:8000`** in your browser to access the interactive sci-fi HUD dashboard.

---

## 🧪 Running Unit Tests

Run the full unit test suite (14 passing tests):

```bash
python -m unittest discover tests
```

---

## 📂 Project Architecture & Documentation

- [VIORA Architecture Documentation](docs/VIORA_Architecture.md)
- [Official Benchmark Results Report](docs/VIORA_Results.md)
- [Text Emotion System Analysis](text_emotion_zip_analysis_report.md)
