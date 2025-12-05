# 🎙️ EmotiVoice - AI Speech Emotion Recognition Platform

A production-ready, industry-standard **Speech Emotion Recognition (SER)** platform powered by **HuBERT** and **Whisper** AI, featuring modern UI/UX, real-time streaming, and granular emotion analysis.

![EmotiVoice Banner](https://img.shields.io/badge/AI-Speech%20Emotion%20Recognition-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-green?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge)

---

## ✨ Key Features

### 🎯 **Comprehensive Emotion Recognition**
- **6 Base Emotions**: Anger, Fear, Happiness, Neutral, Sadness, Surprise
- **Granular Emotions**: 20+ detailed emotional states (Frustration, Anxiety, Joy, Contentment, etc.)
- **Complex Emotions**: Fusion detection (Elation, Bitterness, Desperation, etc.)
- **Confidence Scoring**: Accurate probability scores for all detected emotions

### 📝 **Multi-Language Transcription**
- **90+ Languages** supported via OpenAI Whisper
- **Automatic Language Detection**
- **High Accuracy** transcription with noise robustness
- **Large-v3 Model** for best-in-class performance

### 🚀 **Dual Analysis Modes**

#### 1. **Batch Analysis** 
- Upload audio files (WAV, MP3, M4A, FLAC, OGG, WEBM)
- Automatic silence-based segmentation
- Comprehensive analysis with detailed reports
- Support for files up to 100MB

#### 2. **Live Streaming**
- Real-time recording and analysis
- Instant emotion and transcription feedback
- Continuous monitoring during speech
- Auto-save complete recordings

### 🎨 **Modern, Professional UI/UX**
- Beautiful gradient-based design
- Intuitive landing page with features showcase
- Separate interfaces for different use cases
- Real-time progress indicators and visualizations
- Responsive design for all devices

---

## 📸 Screenshots

### Landing Page
Professional landing page with features, use cases, and clear CTAs

### Batch Analysis Interface
Clean, organized interface for uploading and analyzing audio files

### Live Streaming Interface
Real-time emotion monitoring with instant feedback

---

## 🏗️ Project Structure

```
├── app.py                          # Main Flask application
├── config/
│   └── settings.py                 # Configuration and constants
├── data/
│   ├── uploaded_files/             # Uploaded audio files
│   ├── recorded_files/             # Recorded audio files
│   └── temp_files/                 # Temporary processing files
├── logs/
│   └── app.log                     # Application logs
├── models/
│   ├── hubert/                     # HuBERT model files
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── preprocessor_config.json
│   └── whisper/                    # Whisper model files
│       └── large-v3.pt
├── services/
│   ├── audio_analyzer.py           # Combined analysis service
│   ├── emotion_predictor.py        # HuBERT emotion prediction
│   └── transcriber.py              # Whisper transcription
├── templates/
│   ├── index.html                  # Landing page
│   ├── batch_analysis.html         # Batch analysis interface
│   └── live_streaming.html         # Live streaming interface
├── utils/
│   ├── audio_utils.py              # Audio processing utilities
│   └── logging_util.py             # Logging configuration
├── .env                            # Environment variables
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1️⃣ **Prerequisites**

- Python 3.12+
- FFmpeg (for audio processing)
- CUDA (optional, for GPU acceleration)

### 2️⃣ **Installation**

```bash
# Clone the repository
git clone <repository-url>
cd EmotiVoice

# Create conda environment
conda create -n emotivoice python=3.12
conda activate emotivoice

# Install FFmpeg
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows (via Chocolatey)
choco install ffmpeg

# Install Python dependencies
pip install -r requirements.txt
```

### 3️⃣ **Download Models**

Download the pre-trained models and place them in the `models/` directory:

**HuBERT Model** (facebook/hubert-base-ls960):
- Download from [Hugging Face](https://huggingface.co/facebook/hubert-base-ls960)
- Place in `models/hubert/`

**Whisper Model** (openai/whisper-large-v3):
- Download from [OpenAI](https://github.com/openai/whisper)
- Place `large-v3.pt` in `models/whisper/`

Expected structure:
```
models/
├── hubert/
│   ├── config.json
│   ├── model.safetensors
│   └── preprocessor_config.json
└── whisper/
    └── large-v3.pt
```

### 4️⃣ **Configuration**

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` to configure:

```env
# Server
HOST=localhost
PORT=2024
DEBUG=True
SECRET_KEY=your-secret-key-here

# Device (cpu, cuda, or mps)
DEVICE=cpu

# Emotion mode (base, granular, or both)
EMOTION_MODE=both

# Logging
LOG_LEVEL=INFO
```

### 5️⃣ **Run the Application**

```bash
python app.py
```

Visit `http://localhost:2024` in your browser.

---

## 🎯 Usage Guide

### **Batch Analysis**

1. Navigate to **Batch Analysis** from the landing page
2. Upload an audio file (drag & drop or click)
3. Select emotion analysis mode:
   - **Base**: Fast, 6 core emotions
   - **Granular**: Detailed emotional states
   - **Both**: Comprehensive analysis (recommended)
4. Click **Start Analysis**
5. View results with:
   - Transcription and language detection
   - Emotion scores with percentages
   - Granular emotion breakdowns
   - Complex emotion detection

### **Live Streaming**

1. Navigate to **Live Streaming** from the landing page
2. Click **Start** to begin recording
3. Speak naturally into your microphone
4. View real-time results:
   - Instant transcription
   - Live emotion detection
   - Chunk-by-chunk analysis
5. Click **Stop** when finished
6. Recording is automatically saved

---

## 🔧 Advanced Configuration

### **Emotion Analysis Modes**

Configure in `config/settings.py`:

```python
# Base emotions only (faster)
EMOTION_MODE = 'base'

# Granular emotions with detailed states
EMOTION_MODE = 'granular'

# Both base and granular (comprehensive)
EMOTION_MODE = 'both'
```

### **Granular Emotion Customization**

Modify `GRANULAR_EMOTION_MAP` in `config/settings.py` to add or adjust granular emotions:

```python
GRANULAR_EMOTION_MAP = {
    "Happiness": {
        "primary": ["Happiness", "Joy", "Delight"],
        "secondary": ["Contentment", "Amusement", "Excitement"],
        "threshold": 0.3
    },
    # Add more emotions...
}
```

### **Complex Emotion Fusion**

Define emotion combinations in `COMPLEX_EMOTIONS`:

```python
COMPLEX_EMOTIONS = {
    ("Happiness", "Surprise"): "Elation",
    ("Sadness", "Anger"): "Bitterness",
    # Add more combinations...
}
```

### **Audio Processing Settings**

Adjust in `config/settings.py`:

```python
SAMPLE_RATE = 16000              # Audio sample rate
SILENCE_THRESHOLD = -40          # dBFS for silence detection
MIN_SILENCE_LENGTH = 500         # ms
TARGET_CHUNK_LENGTH = 10 * 1000  # 10 seconds
```

---

## 🧠 Models & Architecture

### **HuBERT (Emotion Recognition)**
- **Model**: facebook/hubert-base-ls960
- **Task**: 6-class emotion classification
- **Input**: 16kHz mono audio
- **Output**: Probability distribution over emotions

### **Whisper (Speech Recognition)**
- **Model**: OpenAI Whisper Large-v3
- **Task**: Multi-language transcription
- **Languages**: 90+ supported
- **Features**: Automatic language detection, noise robustness

### **Architecture Flow**

```
Audio Input
    ↓
Audio Preprocessing (PyDub, FFmpeg)
    ↓
Silence Detection & Segmentation
    ↓
Parallel Processing:
    ├→ HuBERT → Base Emotions → Granular Mapping → Complex Fusion
    └→ Whisper → Transcription → Language Detection
    ↓
Result Aggregation & Formatting
    ↓
Real-time Display via WebSocket
```

---

## 📊 Use Cases

- **Customer Service**: Monitor customer emotions for quality improvement
- **Mental Health**: Track emotional states in therapy sessions
- **Education**: Assess student engagement in online learning
- **Content Creation**: Analyze audience emotional responses
- **Market Research**: Gauge genuine reactions to products
- **Voice Assistants**: Build empathetic AI interactions

---

## 🛠️ Development

### **Adding New Features**

1. **Custom Emotions**: Modify `config/settings.py`
2. **New Services**: Add to `services/` directory
3. **UI Components**: Update templates in `templates/`
4. **Utilities**: Add to `utils/` directory

### **Testing**

```bash
# Test emotion prediction
python -c "from services.emotion_predictor import get_emotion_predictor; \
           predictor = get_emotion_predictor(); \
           print(predictor.predict('path/to/audio.wav'))"

# Test transcription
python -c "from services.transcriber import get_transcriber; \
           transcriber = get_transcriber(); \
           print(transcriber.transcribe('path/to/audio.wav'))"
```

### **Logging**

Logs are stored in `logs/app.log`. Configure log level in `.env`:

```env
LOG_LEVEL=DEBUG  # DEBUG, INFO, WARNING, ERROR
```

---

## 📝 License

MIT [License](LICENSE) © 2025 

---

## 🙏 Acknowledgments

- **HuBERT**: Facebook AI Research
- **Whisper**: OpenAI
- **Flask & SocketIO**: Community
- **UI Inspiration**: Modern web design trends

---

## Author

Satyaki Mitra | Data Scientist | AI Practitioner

---

**Built with ❤️ for understanding human emotions through voice**