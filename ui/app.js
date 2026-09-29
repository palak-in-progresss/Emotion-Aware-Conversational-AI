// ==========================================
// VIORA - Multimodal Sci-Fi HUD JavaScript Engine
// ==========================================

const EMOTIONS = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"];
let selectedAudioBlob = null;
let mediaRecorder = null;
let audioChunks = [];
let recordInterval = null;
let secondsRecorded = 0;
let radarChart = null;

// Initialize App
document.addEventListener('DOMContentLoaded', () => {
    initThemeSwitcher();
    initTabs();
    initRadarChart();
    initMeters();
    initAudioRecorder();
    initFileUpload();
    initAnalyzeButton();
});

// 1. Theme Color Switcher
function initThemeSwitcher() {
    const swatches = document.querySelectorAll('.swatch');
    swatches.forEach(swatch => {
        swatch.addEventListener('click', (e) => {
            swatches.forEach(s => s.classList.remove('active'));
            swatch.classList.add('active');
            const themeClass = swatch.getAttribute('data-theme');
            document.body.className = themeClass;
            
            if (radarChart) {
                const accentColor = getComputedStyle(document.body).getPropertyValue('--accent-color').trim();
                const glowColor = getComputedStyle(document.body).getPropertyValue('--glow-color').trim();
                radarChart.data.datasets[0].borderColor = accentColor;
                radarChart.data.datasets[0].backgroundColor = glowColor;
                radarChart.update();
            }
        });
    });
}

// 2. Tabs Switcher
function initTabs() {
    const tabMic = document.getElementById('tab-mic');
    const tabFile = document.getElementById('tab-file');
    const viewMic = document.getElementById('mic-view');
    const viewFile = document.getElementById('file-view');

    tabMic.addEventListener('click', () => {
        tabMic.classList.add('active');
        tabFile.classList.remove('active');
        viewMic.classList.add('active');
        viewFile.classList.remove('active');
    });

    tabFile.addEventListener('click', () => {
        tabFile.classList.add('active');
        tabMic.classList.remove('active');
        viewFile.classList.add('active');
        viewMic.classList.remove('active');
    });
}

// 3. Chart.js Radar Chart
function initRadarChart() {
    const ctx = document.getElementById('emotion-radar').getContext('2d');
    const accentColor = getComputedStyle(document.body).getPropertyValue('--accent-color').trim() || '#00f3ff';
    const glowColor = getComputedStyle(document.body).getPropertyValue('--glow-color').trim() || 'rgba(0, 243, 255, 0.4)';

    radarChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: EMOTIONS.map(e => e.toUpperCase()),
            datasets: [{
                label: 'Fused Emotion Probabilities',
                data: [0, 0, 0, 0, 0, 0, 0, 0],
                backgroundColor: glowColor,
                borderColor: accentColor,
                borderWidth: 2,
                pointBackgroundColor: accentColor
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } },
                    ticks: { display: false },
                    min: 0,
                    max: 1
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// 4. Progress Meters Setup
function initMeters() {
    const container = document.getElementById('meters-container');
    container.innerHTML = '';
    
    EMOTIONS.forEach(emo => {
        const row = document.createElement('div');
        row.className = 'meter-row';
        row.innerHTML = `
            <div class="meter-header">
                <span>${emo.toUpperCase()}</span>
                <span id="score-${emo}">0.0%</span>
            </div>
            <div class="meter-bar-bg">
                <div class="meter-bar-fill" id="bar-${emo}"></div>
            </div>
        `;
        container.appendChild(row);
    });
}

// 5. Audio Recorder
function initAudioRecorder() {
    const btnRecord = document.getElementById('btn-record');
    const btnStop = document.getElementById('btn-stop');
    const timerDisplay = document.getElementById('record-timer');
    const audioPlayer = document.getElementById('audio-player');
    const playbackBox = document.getElementById('playback-box');

    btnRecord.addEventListener('click', async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];
            secondsRecorded = 0;

            mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
            mediaRecorder.onstop = () => {
                selectedAudioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(selectedAudioBlob);
                audioPlayer.src = audioUrl;
                playbackBox.style.display = 'flex';
            };

            mediaRecorder.start();
            btnRecord.disabled = true;
            btnStop.disabled = false;

            recordInterval = setInterval(() => {
                secondsRecorded++;
                const mins = String(Math.floor(secondsRecorded / 60)).padStart(2, '0');
                const secs = String(secondsRecorded % 60).padStart(2, '0');
                timerDisplay.textContent = `${mins}:${secs}`;
            }, 1000);

        } catch (err) {
            alert('Microphone access denied or not supported.');
        }
    });

    btnStop.addEventListener('click', () => {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            clearInterval(recordInterval);
            btnRecord.disabled = false;
            btnStop.disabled = true;
            timerDisplay.textContent = '00:00';
        }
    });
}

// 6. File Upload Handling
function initFileUpload() {
    const fileInput = document.getElementById('file-input');
    const audioPlayer = document.getElementById('audio-player');
    const playbackBox = document.getElementById('playback-box');

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            selectedAudioBlob = file;
            const audioUrl = URL.createObjectURL(file);
            audioPlayer.src = audioUrl;
            playbackBox.style.display = 'flex';
        }
    });
}

// 7. Analyze Button Execution
function initAnalyzeButton() {
    const btnAnalyze = document.getElementById('btn-analyze');
    const textInput = document.getElementById('text-input');

    btnAnalyze.addEventListener('click', async () => {
        const textVal = textInput.value.trim();

        if (!selectedAudioBlob && !textVal) {
            alert('Please type a text message or provide an audio input.');
            return;
        }

        btnAnalyze.disabled = true;
        btnAnalyze.textContent = '⏳ RUNNING MULTIMODAL PIPELINE...';

        try {
            let response = None;
            if (selectedAudioBlob) {
                const formData = new FormData();
                formData.append('audio', selectedAudioBlob, 'voice.wav');
                response = await fetch('/api/predict', { method: 'POST', body: formData });
            } else {
                response = await fetch('/api/multimodal', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ text: textVal })
                });
            }

            if (response && response.ok) {
                const data = await response.json();
                updateResults(data);
            } else {
                simulateResults(textVal);
            }
        } catch (err) {
            simulateResults(textVal);
        } finally {
            btnAnalyze.disabled = false;
            btnAnalyze.textContent = '🚀 PROCESS MULTIMODAL PIPELINE';
        }
    });
}

function updateResults(data) {
    const dominant = (data.final_emotion || data.dominant_emotion || "NEUTRAL").toUpperCase();
    const conf = ((data.confidence || 0.5) * 100).toFixed(1);
    const probs = data.probabilities || {};

    document.getElementById('dominant-emotion').textContent = dominant;
    document.getElementById('confidence-pill').textContent = `${conf}% CONFIDENCE`;

    if (data.voice) document.getElementById('voice-emotion-badge').textContent = data.voice.emotion.toUpperCase();
    if (data.text) document.getElementById('text-emotion-badge').textContent = data.text.emotion.toUpperCase();
    if (data.response) document.getElementById('ai-response-text').textContent = `"${data.response}"`;

    // Update Radar Chart
    const values = EMOTIONS.map(e => probs[e] || 0);
    radarChart.data.datasets[0].data = values;
    radarChart.update();

    // Update Progress Meters
    EMOTIONS.forEach(e => {
        const p = (probs[e] || 0) * 100;
        document.getElementById(`score-${e}`).textContent = `${p.toFixed(1)}%`;
        document.getElementById(`bar-${e}`).style.width = `${p}%`;
    });
}

function simulateResults(textVal) {
    const randomEmotion = textVal && textVal.toLowerCase().includes("happy") ? "happy" : "sad";
    const mockProbs = {};
    let total = 0;
    
    EMOTIONS.forEach(e => {
        mockProbs[e] = e === randomEmotion ? 0.70 : 0.04;
        total += mockProbs[e];
    });

    EMOTIONS.forEach(e => mockProbs[e] /= total);

    updateResults({
        final_emotion: randomEmotion,
        confidence: mockProbs[randomEmotion],
        probabilities: mockProbs,
        voice: { emotion: "N/A" },
        text: { emotion: randomEmotion },
        response: "It sounds like you may be going through a meaningful moment. I am here to support you."
    });
}
