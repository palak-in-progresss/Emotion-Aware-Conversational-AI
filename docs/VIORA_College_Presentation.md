# Emotion-Aware Conversational AI: VIORA
## Complete Presentation Deck (College Project Defense & Proposal)

---

## Slide 1: Title Slide

# Emotion-Aware Conversational AI
### Moving Beyond Words to Understand Human Emotion
**Subtitle:** A Multi-Modal Architecture for Emotion Perception, Reasoning, and Adaptive Voice Response  
**Domain:** Artificial Intelligence | Machine Learning | Natural Language Processing | Speech Processing  
**Presenter:** [Your Name & Teammate Name / Team Members]  
**Institution:** [College / University Name]  

---

## Slide 2: Problem Statement — The Core Gap

> **"Current conversational AI understands what users say, but fails to understand how they feel."**

* **Intent Without Emotion:** Traditional assistants (Alexa, Siri, standard LLM chatbots) follow a rigid pipeline: **Text $\rightarrow$ Intent Extraction $\rightarrow$ Standard Response**.
* **Loss of Critical Signals:** Text-only analysis discards vocal tone, pitch volatility, hesitation, and emotional stress.
* **Robotic Insensitivity:** A user saying *"I had a terrible day"* in a trembling voice receives the exact same flat, neutral response as a user asking for the weather.
* **Static Speech Synthesis (TTS):** Modern TTS sounds clear, but lacks **emotional adaptation**—it cannot adjust its pitch, pace, or warm tone to match the situation.

---

## Slide 3: Project Goal — What Are We Building?

### Big-Picture Objective
Build an **Emotion-Aware Conversational AI** capable of perceiving a user's emotional state from both **Speech Acoustics & Text Semantics**, reasoning through dialogue context, and expressing responses with **adaptive tone and speech modulation**.

```text
                                  USER
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
              TEXT INPUT                          VOICE INPUT
                  │                                   │
          Text Emotion (dair-ai)              Speech-to-Text
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                         Multi-Modal Emotion Fusion
                         (Voice + Text Intelligence)
                                    │
                                    ▼
                       Conversational Brain (LLM)
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
             TEXT OUTPUT                         TTS OUTPUT
          (Empathetic Text)                 (Emotional Speech)
                                                      │
                                           • Calm → Soothing / Slower
                                           • Excited → Energetic / Bright
                                           • Sad → Gentle / Warm
                                           • Frustrated → Reassuring
```

---

## Slide 4: Dual-Stream Perception Datasets & Models

Our team divides perception into two dedicated Machine Learning streams:

```
                               DUAL PERCEPTION ENGINE
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
     VOICE EMOTION STREAM                            TEXT EMOTION STREAM
  (Acoustic Feature Processing)                    (NLP & Text Classification)
                 │                                               │
  • Dataset: RAVDESS (1,440 clips)                 • Dataset: dair-ai/emotion (~20k samples)
  • Source: Zenodo Benchmark                      • Source: HuggingFace Hub
  • Features: 162 Acoustic Features               • Format: Plain Text Sentences
    (MFCCs, Chroma, Mel, ZCR, Contrast)           • Model: Single-Label NLP Classifier
  • Champion Model: SVM (RBF Kernel)              • Classes: sadness, joy, love, 
  • Classes: neutral, calm, happy, sad,                      anger, fear, surprise
    angry, fearful, disgust, surprised
```

---

## Slide 5: Multi-Modal Schema Mapping & Fusion

### Harmonizing Voice & Text Emotion Categories

To fuse both streams seamlessly into a unified emotional decision:

| Emotion Concept | Voice Model Label (RAVDESS) | Text Model Label (`dair-ai/emotion`) | Unified Fused State |
| :--- | :--- | :--- | :--- |
| **Joy / Happiness** | `happy` | `joy` | **High Positive Arousal** |
| **Sadness** | `sad` | `sadness` | **Low Negative Arousal** |
| **Anger** | `angry` | `anger` | **High Negative Intensity** |
| **Fear / Anxiety** | `fearful` | `fear` | **Apprehensive State** |
| **Surprise** | `surprised` | `surprise` | **Unanticipated Event** |
| **Affection / Warmth**| `calm` / `happy` | `love` | **Empathetic Warmth** |
| **Neutral Baseline** | `neutral` / `calm` | Baseline Text | **Balanced State** |

#### Late Decision-Level Fusion Formula:
$$\text{Fused Probability} = w_{\text{voice}} \cdot P_{\text{voice}}(\text{Emotion}) + w_{\text{text}} \cdot P_{\text{text}}(\text{Emotion})$$
* Dynamic confidence weighting automatically adjusts if audio is noisy or text is sarcastic.

---

## Slide 6: System Framework — Core Tagline

> ### **Perceive $\rightarrow$ Understand $\rightarrow$ Reason $\rightarrow$ Adapt $\rightarrow$ Express**

```
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ 1. PERCEIVE     │   │ 2. UNDERSTAND   │   │ 3. REASON       │   │ 4. ADAPT        │   │ 5. EXPRESS      │
│                 │   │                 │   │                 │   │                 │   │                 │
│ • Speech-to-Text│   │ • Multi-Modal   │   │ • Context       │   │ • Determine     │   │ • Empathetic    │
│ • Voice Stream  │──>│   Fusion        │──>│   Memory        │──>│   Response      │──>│   Text          │
│   (RAVDESS)     │   │ • Emotion       │   │ • Intent &      │   │   Style         │   │ • Emotion-Aware │
│ • Text Stream   │   │   Confidence    │   │   Safety        │   │ • Voice Tone    │   │   TTS Output    │
│   (dair-ai)     │   │                 │   │                 │   │                 │   │                 │
└─────────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘   └─────────────────┘
```

---

## Slide 7: Current Progress & Team Implementation

#### 1. Voice Emotion Pipeline (Acoustics & ML)
* ✅ Extracted 162 acoustic features (MFCCs, Mel-Spectrogram, Chroma, ZCR, Spectral Centroid/Contrast).
* ✅ Benchmarked 4 candidate model families (**SVM, Random Forest, XGBoost, MLP**) on 8 RAVDESS categories.
* ✅ **Champion SVM Model achieved 53.90% Macro F1** on **100% Unseen Speakers** (Actor-Independent Validation).

#### 2. Text Emotion Pipeline (NLP & Corpus)
* ✅ Integrated HuggingFace **`dair-ai/emotion`** dataset (~20,000 text samples).
* ✅ Configured 6-class text emotion classification (*sadness, joy, love, anger, fear, surprise*).

#### 3. Interactive Interface & Backend Server
* ✅ Built custom HTML5/CSS3/JS **VIORA Sci-Fi HUD** frontend featuring Web Audio API live microphone recorder.
* ✅ Dynamic **5-Color Theme Switcher** (VIORA Cyan, Amethyst Purple, Bio Emerald, Solar Gold, Crimson Pulse).
* ✅ Real-time **Chart.js 8-Axis Emotion Radar Chart** & probability progress meters.
* ✅ Python HTTP backend server connecting UI to trained prediction artifacts (`model.pkl`, `scaler.pkl`, `feature_config.json`).

---

## Slide 8: Future Scope & Roadmap

#### Stage 1: Near-Term Enhancements
* **End-to-End Fusion Integration:** Connect text classifier (`dair-ai/emotion`) with voice classifier (RAVDESS) in backend `fusion/emotion_fusion.py`.
* **Emotion-Aware Speech Synthesis:** Implement pitch, speed, and volume modulation in TTS output based on predicted emotion.
* **Expanded Datasets:** Incorporate CREMA-D audio dataset for broader speaker diversity.

#### Stage 2: Next Stage (Conversational Depth)
* **Long-Term Dialogue Memory:** Track emotional shifts over multiple turns (e.g., *Frustrated $\rightarrow$ Calm*).
* **Personalized User Profiles:** Learn individual baseline vocal & text characteristics.

#### Stage 3: Advanced Research Scope
* **Tri-Modal Emotion Perception:** Integrate **Facial Expression Analysis** (Video/Webcam + Voice + Text).
* **Real-Time Streaming Emotion Detection:** Continuous emotion tracking during ongoing speech without requiring manual audio stopping.

---

## Slide 9: Final Vision

> ### **"From conversational AI that understands words..."**
> ### **"→ to conversational AI that understands people."**

#### Summary of Impact:
* **Human-Centered Interaction:** Replaces rigid command execution with natural, empathetic dialogue.
* **Dual-Stream Perception:** Combines acoustic signals (RAVDESS) + NLP semantics (`dair-ai/emotion`) for unmatched emotional accuracy.
* **Modular Engineering:** Cleanly separated perception, intelligence, and expression layers ready for real-world deployment.

---

## Slide 10: Questions & Discussion

# Thank You!
### VIORA - Emotion-Aware Conversational AI

* **Project Repository:** [github.com/palak-in-progresss/Emotion-Aware-Conversational-AI](https://github.com/palak-in-progresss/Emotion-Aware-Conversational-AI)
