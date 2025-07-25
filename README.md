# 🎙️ HuBERT-Whisper Based Speech Emotion Analyzer

A production-ready **Speech Emotion Recognition (SER)** system powered by **HuBERT** (for emotion detection) and **Whisper** (for transcription), wrapped in a clean and interactive **Flask web app**.

This app allows users to either **upload** or **record audio** in real time, and returns both:
- **Detected Emotion**
- **Transcribed Text**

---

## 📦 Key Features

- 🎯 Emotion recognition from raw audio using HuBERT
- 📝 Transcription of speech using Whisper
- 🎙️ Live recording + file upload interface
- ⚙️ Offline usage after model download
- 🔍 Segmentation, cleaning & preprocessing of audio
- 🧪 Sample voices for testing and demo

---

## 🧠 Models Used

### 🔹 [HuBERT](https://arxiv.org/abs/2106.07447)
Self-supervised model for speech representation learning, using **offline clustering** and **masked prediction loss**.

### 🔹 [Whisper](https://openai.com/research/whisper)
OpenAI’s multilingual speech recognition model for **robust ASR**, handling noise and multiple accents effectively.

---

## 🧠  Model Details

### 🔹 HuBERT (Hidden-Unit BERT)
Developed by Meta AI, HuBERT is a **self-supervised model** that learns speech representations by predicting masked regions in input audio based on pseudo-labels from clustering.

### 🔹 Whisper
Whisper is a general-purpose speech recognition system by OpenAI designed to be **noise-robust**, **multilingual**, and **open-domain**.

---

## 🧰 Features

- Emotion detection from speech using HuBERT
- Transcription using Whisper
- Record audio directly from browser or upload a `.wav` file
- Simple Flask-based user interface
- Fully offline support after model files are downloaded

---

## 📁 Project Structure

```bash
hubert-whisper-ser/
├── app.py                      # Flask entrypoint
├── config.py                   # Configuration variables
├── flask_app/                  # Core backend logic
│   ├── audio_analyzer.py       # Emotion + ASR analysis
│   ├── hubert_predictor.py     # HuBERT inference
│   ├── transcriber.py          # Whisper transcription
│   ├── audio_preprocessing.py  # Audio cleaning/splitting
│   └── helper.py               # Shared utilities
├── templates/
│   └── index.html              # UI HTML page
├── static/
│   └── silence-detector.js     # Client-side silence recording logic
├── data/
│   ├── app_data/               # Uploaded, recorded, and segmented files
│   └── voice_samples/          # Sample audios for testing
├── local_model_files/          # Offline HuBERT + Whisper models
├── requirements.txt
└── README.md
```
---

## 🚀 Quickstart Guide

Follow these steps to set up and run the application locally.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/satyaki-mitra/HuBERT-Whisper-Based-Speech-Emotion-Analyzer.git
cd hubert-whisper-ser
```

### 2️⃣ Create & Activate Conda Environment
```bash
conda create -n speech_emotion_env python=3.12.3
conda activate speech_emotion_env
```
### 3️⃣ Install Dependencies
Install FFmpeg (for audio handling) and Python dependencies:

```bash
# For macOS
brew install ffmpeg

# For Ubuntu/Debian
sudo apt install ffmpeg

# Python libraries
pip install -r requirements.txt
```
### 4️⃣ Download Pretrained Model Files
Download the pretrained models for offline usage from this Google Drive link.

Unzip the downloaded file.

Place the folder named local_model_files directly inside the project directory.

Final structure should look like:
```markdown
hubert-whisper-ser/
│
├── app.py
├── audio_analyzer.py
├── hubert_predictor.py
├── transcriber.py
├── ...
└── local_model_files/
    ├── hubert/
    ├── whisper/
    └── ...
```

### 5️⃣ Run the Flask App
```bash
python app.py
```

Then open your browser and go to: http://localhost:2024/

---

## 🖥️ App Usage

- Record Audio: Press and hold the microphone icon to record audio.

- Upload Audio: Upload an existing .wav file.

- Analyze: Click on Analyze Audio to view:

  - Predicted Emotion

  - Speech Transcription
---

## 📁 Project Structure Overview


---

## 🧪 Sample Output
"Analyzed: neutral emotion with transcription: 'I'm feeling alright today but it's been a long week.'"

## 📌 Notes

- Works best with clean, clear voice recordings.

- Extendable to multi-language or gender-based emotion analysis.

- Currently optimized for English.

---

## 📄 License
MIT License © 2025 — [Satyaki Mitra]

