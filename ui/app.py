import os
import sys
import tempfile
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.voice_emotion.predict import VoiceEmotionPredictor

# Page Configuration
st.set_page_config(
    page_title="VIORA - Emotion-Aware AI",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS Stylesheet
CSS_PATH = os.path.join(PROJECT_ROOT, "ui", "assets", "style.css")
if os.path.exists(CSS_PATH):
    with open(CSS_PATH, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Theme Palette Map
THEMES = {
    "🩵 VIORA Cyan (Default)": {"accent": "#00f3ff", "glow": "rgba(0, 243, 255, 0.4)"},
    "💜 Amethyst Purple": {"accent": "#d946ef", "glow": "rgba(217, 70, 239, 0.4)"},
    "💚 Bio Emerald": {"accent": "#10b981", "glow": "rgba(16, 185, 129, 0.4)"},
    "🧡 Solar Gold": {"accent": "#f59e0b", "glow": "rgba(245, 158, 11, 0.4)"},
    "❤️ Crimson Pulse": {"accent": "#ef4444", "glow": "rgba(239, 68, 68, 0.4)"}
}

# Sidebar Theme Selector
st.sidebar.markdown("### 🎨 VIORA HUD Customizer")
selected_theme_name = st.sidebar.selectbox("Choose Interface Color Accent", list(THEMES.keys()))
theme = THEMES[selected_theme_name]

# Apply Dynamic CSS Variables based on selected theme
dynamic_theme_css = f"""
<style>
:root {{
    --viora-accent: {theme['accent']} !important;
    --viora-glow: {theme['glow']} !important;
}}
</style>
"""
st.markdown(dynamic_theme_css, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
predictor = VoiceEmotionPredictor()

if predictor.is_loaded:
    st.sidebar.success("✅ Voice Emotion Model Loaded (SVM 8-Class)")
else:
    st.sidebar.warning("⚠️ Model running on acoustic fallback. Place model.pkl in models/voice_emotion/")

st.sidebar.markdown("---")
st.sidebar.markdown("**VIORA v1.0** • Multi-Modal Emotion Assistant")

# Header Section
st.markdown("""
<div class="viora-header">
    <div class="viora-title">V I O R A</div>
    <div class="viora-subtitle">INTUITIVE VOICE & EMOTION INTELLIGENCE SYSTEM</div>
    <div style="margin-top: 15px;">
        <span class="status-pill">● SYSTEM ONLINE & READY</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Main 2-Column Layout
col_input, col_viz = st.columns([1, 1], gap="large")

with col_input:
    st.markdown("### 🎙️ Audio Input Controls")
    
    input_method = st.radio("Choose Input Method", ["Upload WAV Audio File", "Record Live Microphone"], horizontal=True)
    
    audio_bytes = None
    file_path = None
    
    if input_method == "Upload WAV Audio File":
        uploaded_file = st.file_uploader("Drop an audio file (.wav, .mp3, .ogg)", type=["wav", "mp3", "ogg"])
        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                file_path = tmp.name
    else:
        recorded_audio = st.audio_input("Record your voice (3-5 seconds)")
        if recorded_audio is not None:
            audio_bytes = recorded_audio.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_bytes)
                file_path = tmp.name

    if audio_bytes is not None:
        st.markdown("#### 🎵 Audio Playback")
        st.audio(audio_bytes, format="audio/wav")

    st.markdown("---")
    analyze_btn = st.button("🚀 Analyze Voice Emotion", type="primary", use_container_width=True)

with col_viz:
    st.markdown("### 🧠 Voice Emotion Analysis")
    
    if analyze_btn and file_path is not None:
        with st.spinner("Extracting 162 acoustic features & classifying emotion..."):
            result = predictor.predict(file_path)
            
            # Remove temp file
            if os.path.exists(file_path):
                os.remove(file_path)
                
            dominant = result["dominant_emotion"].upper()
            confidence = result["confidence"] * 100
            probs = result["probabilities"]
            
            # Display Dominant Emotion Badge
            st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <div style="color: var(--viora-subtext); letter-spacing: 2px; font-size: 0.9rem;">DOMINANT VOICE EMOTION</div>
                <div class="emotion-badge" style="margin-top: 10px;">{dominant}</div>
                <div style="margin-top: 10px; color: var(--viora-accent); font-weight: 600;">{confidence:.1f}% Confidence Score</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Plotly Radar Chart
            emotions = list(probs.keys())
            values = list(probs.values())
            
            # Close the radar loop
            radar_emotions = emotions + [emotions[0]]
            radar_values = values + [values[0]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=radar_values,
                theta=radar_emotions,
                fill='toself',
                fillcolor=theme['glow'],
                line=dict(color=theme['accent'], width=3),
                name='Voice Probabilities'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                    bgcolor='rgba(15, 23, 42, 0.6)'
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#f1f5f9', family='Outfit'),
                margin=dict(l=40, r=40, t=20, b=20),
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed Probability Meters
            st.markdown("#### 📊 Emotion Probability Meters")
            for emo, p in probs.items():
                col_name, col_bar = st.columns([1, 3])
                with col_name:
                    st.write(f"**{emo.capitalize()}**")
                with col_bar:
                    st.progress(float(p))

    elif analyze_btn and file_path is None:
        st.error("Please upload or record an audio clip first!")
    else:
        st.info("👈 Upload or record an audio clip on the left, then click 'Analyze Voice Emotion'.")
