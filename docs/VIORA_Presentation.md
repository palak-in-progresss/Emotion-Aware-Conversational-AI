# VIORA: Emotion-Aware Conversational AI
## Comprehensive Presentation Deck (Markdown Format)

---

## Slide 1: Title Slide

# VIORA
### Next-Generation Emotion-Aware Conversational AI
**Subtitle:** Intelligent Multi-Modal Voice & Sentiment Recognition System  
**Domain:** Artificial Intelligence | Machine Learning | Speech Signal Processing | NLP  
**Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)  
**Presenter:** [Your Name / Team Members]  

---

## Slide 2: Problem Statement & Motivation

### The Problem with Traditional Conversational AI
* **Emotional Blindness:** Standard voice assistants (Alexa, Siri, standard chatbots) process *what* is said (transcription), but completely ignore *how* it is said (tone, vocal stress, emotion).
* **Flat & Robotic Responses:** AI responds neutrally even when a user is distressed, frustrated, or excited.
* **Lack of Empathy:** Human communication is over **70% non-verbal and vocal**. Without emotion detection, AI cannot build meaningful rapport.

### Our Solution: VIORA
* An intelligent assistant that perceives emotional state through **Voice Acoustics** and **Text Semantics**.
* Combines multi-modal predictions to generate **contextually empathetic responses**.

---

## Slide 3: System Architecture & Workflow

```
                  ┌───────────────────────────────┐
                  │    User Input (Microphone)    │
                  └───────────────┬───────────────┘
                                  │
          ┌───────────────────────┴───────────────────────┐
          │                                               │
          ▼                                               ▼
┌──────────────────┐                            ┌──────────────────┐
│  Voice Emotion   │                            │  Speech to Text  │
│  Extraction (SER)│                            │  Transcription   │
└─────────┬────────┘                            └─────────┬────────┘
          │                                               │
          │                                               ▼
          │                                     ┌──────────────────┐
          │                                     │   Text Emotion   │
          │                                     │   Detector (NLP) │
          │                                     └─────────┬────────┘
          │                                               │
          └───────────────────────┬───────────────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Multi-Modal Fusion Engine │
                    │   (Confidence Weighting)  │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ Empathetic Response Engine│
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                    ┌───────────────────────────┐
                    │ VIORA Sci-Fi HUD Web UI   │
                    └───────────────────────────┘
```

---

## Slide 4: Speech Emotion Recognition (SER) Pipeline

### Core Technical Specification (v1 SER)
1. **Audio Standardization:** 22.05 kHz Mono audio signals.
2. **Preprocessing:** Dynamic silence trimming (`top_db=20`) and fixed 3.0-second padding/cropping.
3. **Feature Extraction (162 Features):**
   * **MFCCs (40 features):** Vocal tract shape and timbre resonance.
   * **Mel-Spectrogram (80 features):** Frequency energy matching human ear perception.
   * **Chroma STFT (24 features):** Musical pitch & harmonic inflection.
   * **Spectral Centroid (2 features):** Sound brightness & sharpness.
   * **Zero-Crossing Rate (2 features):** Breathiness, friction, and vocal agitation.
   * **Spectral Contrast (14 features):** Enunciation clarity vs. background noise.
4. **Temporal Aggregation:** Mean + Standard Deviation across time frames.

---

## Slide 5: Experimental Methodology & Validation Strategy

### Dataset: RAVDESS Benchmark
* 1,440 audio clips across **8 Emotion Categories:**  
  `Neutral` | `Calm` | `Happy` | `Sad` | `Angry` | `Fearful` | `Disgust` | `Surprised`

### Preventing Speaker Data Leakage ⭐
* **The Challenge:** Standard random splits put clips from the same actor in both Train and Test sets, causing models to memorize voice identity rather than emotion.
* **Our Rigorous Solution:** **Actor-Independent Split (GroupKFold by Actor ID)**
  * **Train Set:** Actors 1 to 18 (~75%, 1,080 audio files)
  * **Test Set:** Actors 19 to 24 (~25%, 360 audio files - *100% Unseen Speakers*)
* **Feature Normalization:** `StandardScaler` fitted **strictly on the training split**.

---

## Slide 6: Model Benchmarking & Results

### Candidate Model Evaluation (8-Class Unseen Speaker Benchmark)

| Model Architecture | Accuracy (%) | Macro F1 (%) | Weighted F1 (%) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **SVM (RBF Kernel)** 🏆 | **53.89%** | **53.90%** | **53.64%** | **Champion Model** |
| **MLP Neural Network** | 53.89% | 53.54% | 53.85% | Runner-Up |
| **Random Forest** | 45.28% | 42.46% | 43.92% | Baseline |
| **XGBoost Classifier** | 43.61% | 41.93% | 42.72% | Baseline |

### Key Findings & Defense Takeaways
* **SVM with RBF Kernel** outperformed tree ensembles and deep networks for dense acoustic feature vectors.
* **53.90% Macro F1 on 8 Unseen Speakers** matches published academic benchmarks for classical speech emotion classification (random chance across 8 classes is 12.5%).

---

## Slide 7: Detailed Emotion Classification Breakdown

### Per-Class Performance Matrix (Winning SVM Model)

| Emotion Class | Precision | Recall | F1-Score | Key Acoustic Characteristic |
| :--- | :---: | :---: | :---: | :--- |
| **Surprised** | **0.77** | **0.75** | **0.76** | High pitch jump & spectral centroid |
| **Calm** | **0.69** | **0.69** | **0.69** | Steady energy & smooth pitch contour |
| **Angry** | **0.54** | **0.69** | **0.61** | High upper-frequency energy & volume |
| **Neutral** | **0.62** | **0.54** | **0.58** | Low variance across all coefficients |
| **Disgust** | **0.55** | **0.54** | **0.55** | Low pitch with vocal friction |
| **Sad / Happy / Fearful** | 0.36 - 0.41 | 0.35 - 0.40 | 0.36 - 0.40 | Overlapping acoustic arousal levels |

### Justification for Multi-Modal Fusion
* High-arousal emotions (*Surprised, Calm, Angry*) are strongly distinguished by **voice alone**.
* Overlapping emotions (*Happy vs. Sad vs. Fearful*) demonstrate why **fusing text sentiment** is critical for full conversational accuracy.

---

## Slide 8: Frontend Interface - VIORA Sci-Fi HUD

### Interactive User Experience
* **Pure HTML5 / CSS3 / JavaScript** Web Application (Hosted on custom Python HTTP server).
* **Futuristic Sci-Fi HUD Design:** Glassmorphic translucent cards, radial grid lines, and glowing neon accents.
* **Dynamic 5-Color Theme Switcher:**
  * 🩵 **VIORA Cyan** (Default HUD Blue)
  * 💜 **Amethyst Purple**
  * 💚 **Bio Emerald**
  * 🧡 **Solar Gold**
  * ❤️ **Crimson Pulse**
* **Real-Time Visualizations:**
  * Web Audio API live microphone recorder.
  * Interactive **Chart.js 8-Axis Radar Chart**.
  * Real-time animated emotion probability meters.

---

## Slide 9: Key Contributions & Novelty

1. **Rigorous ML Methodology:** Evaluated 4 distinct model families using strict actor-independent validation to ensure true generalization.
2. **Rich Acoustic Feature Representation:** Integrated 6 spectral and temporal feature extraction techniques (162 features).
3. **Serialized Production Pipeline:** Exported lightweight `model.pkl`, `scaler.pkl`, and `feature_config.json` for real-time web inference.
4. **Themeable Interactive Web Interface:** Standalone web application with live voice recording and radar chart analytics.

---

## Slide 10: Future Roadmap & Conclusion

### Future Enhancements
* **Dataset Expansion:** Incorporate CREMA-D and TESS datasets for multi-corpus domain adaptation.
* **Text-Voice Fusion Integration:** Complete late-fusion decision weighting with text emotion analysis.
* **Real-Time Emotion Drift:** Track emotional trajectory across multi-turn user conversations.

### Conclusion
**VIORA** successfully bridges the gap between raw audio signals and emotional intelligence, providing a robust foundation for empathetic human-computer interaction.

---

## Slide 11: Questions & Answers

# Thank You!
### VIORA - Emotion-Aware Conversational AI

* **GitHub Repository:** [github.com/palak-in-progresss/Emotion-Aware-Conversational-AI](https://github.com/palak-in-progresss/Emotion-Aware-Conversational-AI)
* **Questions & Discussion**
