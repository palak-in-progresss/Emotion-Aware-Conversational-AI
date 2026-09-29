# 📊 VIORA — Official Experimental Results & Model Benchmarks

---

## 🎧 1. Speech Emotion Recognition (SER) Benchmarks

All evaluation was conducted strictly under **speaker-independent protocols** (held-out RAVDESS Actors 19–24, 360 untouched test clips). No test speakers or scalers were used during model tuning.

| Model Variant | Dataset | Train Clips | Features | Classifier Architecture | Test Accuracy | Test Macro F1 | Test Balanced Acc |
| :--- | :--- | :---: | :---: | :--- | :---: | :---: | :---: |
| **V1 Baseline** | Real RAVDESS | 1,080 | 162 | SVM RBF ($C=1.0$) | 53.89% | 53.90% | 53.89% |
| **V2 Representation** | Real RAVDESS | 1,080 | 342 | SVM RBF ($C=1.0$) | 54.44% | 52.70% | 52.92% |
| **Tuned Champion** | Real RAVDESS | 1,080 | 342 | SVM RBF ($C=10.0, \gamma=0.001$) | 53.06% | 51.45% | 51.56% |
| **Final Expanded Model** | **RAVDESS + CREMA-D** | **2,080** | **342** | **SVM RBF ($C=10.0, \gamma=0.001$)** | **54.17%** | **53.69%** | **53.39%** |

### 📈 Final Expanded SER Model Per-Class Metrics (Actors 19–24, 360 Clips)

| Emotion Class | Precision | Recall | F1-Score | Support | Key Improvement vs Single-Dataset |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Calm** | 0.6939 | 0.7083 | **70.10%** | 48 | High stability |
| **Surprised** | 0.7179 | 0.5833 | **64.37%** | 48 | High precision |
| **Disgust** | 0.5147 | 0.7292 | **60.34%** | 48 | Strong acoustic separation |
| **Happy** | 0.5682 | 0.5208 | **54.35%** | 48 | Consistent positive valence |
| **Neutral** | 0.5882 | 0.4167 | **48.78%** | 24 | **+18.35% jump** (from 30.43%) |
| **Fearful** | 0.5385 | 0.4375 | **48.28%** | 48 | Improved recall |
| **Angry** | 0.4310 | 0.5208 | **47.17%** | 48 | Stable high intensity |
| **Sad** | 0.3696 | 0.3542 | **36.17%** | 48 | **+7.60% jump** (from 28.57%) |

---

## 📝 2. Text Emotion Recognition Benchmarks

Evaluated on held-out test split of the combined dataset (**dair-ai/emotion** + **Google GoEmotions**, 61,627 total samples):

| Metric | Measured Value |
| :--- | :---: |
| **Validation Accuracy** | **68.99%** |
| **Validation Macro F1** | **64.11%** |
| **Total Cleaned Samples** | **61,627** |
| **Train / Val / Test Split** | 80% (49,301) / 10% (6,163) / 10% (6,163) |

---

## 🤝 3. Multimodal Probability-Level Fusion

The multimodal fusion layer computes normalized probability vectors across the 8 shared emotion categories:

$$P_{\text{fused}}(e) = \frac{w_{\text{voice}} \cdot P_{\text{voice}}(e) + w_{\text{text}} \cdot P_{\text{text}}(e)}{\sum_{k} \left( w_{\text{voice}} \cdot P_{\text{voice}}(k) + w_{\text{text}} \cdot P_{\text{text}}(k) \right)}$$

- **Default Reliability Weights**: $w_{\text{voice}} = 0.5$, $w_{\text{text}} = 0.5$ (Configurable in `config.py`).
- **Modality Handling**: Automatically re-normalizes for `voice + text`, `voice_only`, or `text_only` modes without runtime exceptions.
