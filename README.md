# 🎙️ HuBERT-Whisper Based Speech Emotion Analyzer

This project implements a robust **Speech Emotion Recognition (SER)** system by leveraging the power of **HuBERT** and **Whisper** models. The application allows users to **upload or record audio**, and analyzes it for **emotional content** and **transcription**, all within a lightweight **Flask web application**.

---

## 🧠 About the Models

### 🔹 HuBERT (Hidden-Unit BERT)
HuBERT is a **self-supervised speech representation learning model** developed by Meta AI. It extends BERT-style masked prediction to audio using pseudo-labels generated via offline clustering, learning acoustic and linguistic representations simultaneously.

### 🔹 Whisper
OpenAI's Whisper is a general-purpose speech recognition model that delivers high-quality **ASR (Automatic Speech Recognition)** performance across multiple languages and noise conditions.

---

## 🧰 Features

- Emotion detection from speech using HuBERT
- Transcription using Whisper
- Record audio directly from browser or upload a `.wav` file
- Simple Flask-based user interface
- Fully offline support after model files are downloaded

---

## 🚀 Quickstart Guide

Follow these steps to set up and run the application locally.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/hubert-whisper-ser.git
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

