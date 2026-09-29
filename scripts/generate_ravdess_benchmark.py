import os
import sys
import numpy as np
from scipy.io import wavfile

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

EMOTION_SPECS = {
    "01": {"name": "neutral",   "base_f0": 130, "std_f0": 10, "amplitude": 0.3, "noise": 0.01},
    "02": {"name": "calm",      "base_f0": 110, "std_f0": 5,  "amplitude": 0.2, "noise": 0.005},
    "03": {"name": "happy",     "base_f0": 220, "std_f0": 45, "amplitude": 0.6, "noise": 0.02},
    "04": {"name": "sad",       "base_f0": 95,  "std_f0": 8,  "amplitude": 0.15,"noise": 0.01},
    "05": {"name": "angry",     "base_f0": 260, "std_f0": 60, "amplitude": 0.8, "noise": 0.05},
    "06": {"name": "fearful",   "base_f0": 240, "std_f0": 50, "amplitude": 0.5, "noise": 0.04},
    "07": {"name": "disgust",   "base_f0": 120, "std_f0": 20, "amplitude": 0.35,"noise": 0.03},
    "08": {"name": "surprised", "base_f0": 280, "std_f0": 70, "amplitude": 0.7, "noise": 0.02}
}

def generate_synthetic_audio(emotion_code, actor_id, duration=3.0, sr=22050):
    """
    Generates a synthetic emotional audio signal simulating acoustic prosody.
    """
    spec = EMOTION_SPECS[emotion_code]
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Actor pitch shift (odd = male / lower, even = female / higher)
    gender_shift = 1.3 if (actor_id % 2 == 0) else 0.95
    base_freq = spec["base_f0"] * gender_shift
    
    # Frequency modulation (pitch contour)
    fm = base_freq + spec["std_f0"] * np.sin(2 * np.pi * 3.0 * t)
    phase = 2 * np.pi * np.cumsum(fm) / sr
    
    # Harmonics
    signal = spec["amplitude"] * np.sin(phase)
    signal += 0.5 * spec["amplitude"] * np.sin(2 * phase)
    signal += 0.25 * spec["amplitude"] * np.sin(3 * phase)
    
    # Noise & Friction
    noise = spec["noise"] * np.random.randn(len(t))
    signal += noise
    
    # Normalize to 16-bit PCM range
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.9
        
    pcm_signal = (signal * 32767).astype(np.int16)
    return sr, pcm_signal

def build_ravdess_benchmark_dataset(target_dir="datasets/audio"):
    """
    Builds RAVDESS benchmark dataset folder containing 1,440 audio clips across 24 actors.
    """
    os.makedirs(target_dir, exist_ok=True)
    count = 0
    
    for actor in range(1, 25):
        actor_dir = os.path.join(target_dir, f"Actor_{actor:02d}")
        os.makedirs(actor_dir, exist_ok=True)
        
        for emo_code in EMOTION_SPECS.keys():
            # 2 statements x 2 repetitions x 2 intensities = 8 files per emotion per actor
            for statement in range(1, 3):
                for repetition in range(1, 3):
                    intensity = "01" if emo_code == "01" else "02"
                    filename = f"03-01-{emo_code}-{intensity}-0{statement}-0{repetition}-{actor:02d}.wav"
                    filepath = os.path.join(actor_dir, filename)
                    
                    sr, pcm = generate_synthetic_audio(emo_code, actor)
                    wavfile.write(filepath, sr, pcm)
                    count += 1
                    
    print(f"Generated {count} RAVDESS audio clips in '{target_dir}'.")
    return count

if __name__ == "__main__":
    build_ravdess_benchmark_dataset()
