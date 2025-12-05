# 🔌 EmotiVoice API Documentation

Complete REST API and WebSocket documentation for programmatic access.

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [REST API Endpoints](#rest-api-endpoints)
4. [WebSocket Events](#websocket-events)
5. [Data Models](#data-models)
6. [Code Examples](#code-examples)
7. [Rate Limits](#rate-limits)
8. [Error Handling](#error-handling)

---

## Overview

### Base URL
```
http://localhost:2024
```

### Supported Formats
- **Request**: `multipart/form-data`, `application/json`
- **Response**: `application/json`
- **WebSocket**: Socket.IO protocol

### API Version
```
Version: 1.0
```

---

## Authentication

Currently, the API operates without authentication for local development. For production:

```python
# Future implementation
headers = {
    'Authorization': 'Bearer YOUR_API_KEY',
    'Content-Type': 'application/json'
}
```

---

## REST API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check API health status

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0",
  "models": {
    "hubert": "loaded",
    "whisper": "loaded"
  },
  "uptime": 3600
}
```

---

### 2. Upload Audio File

**Endpoint:** `POST /upload`

**Description:** Upload an audio file for batch processing

**Content-Type:** `multipart/form-data`

**Parameters:**
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | File | Yes | Audio file (WAV, MP3, M4A, etc.) |

**Example Request:**
```bash
curl -X POST http://localhost:2024/upload \
  -F "file=@recording.wav"
```

**Success Response (200):**
```json
{
  "success": true,
  "filepath": "/path/to/uploaded/file.wav",
  "filename": "recording_abc123.wav"
}
```

**Error Response (400):**
```json
{
  "error": "Invalid file type"
}
```

---

### 3. Analyze Audio (Synchronous)

**Endpoint:** `POST /analyze`

**Description:** Analyze uploaded audio file synchronously

**Content-Type:** `application/json`

**Request Body:**
```json
{
  "filepath": "/path/to/audio.wav",
  "emotion_mode": "both",
  "include_timestamps": false
}
```

**Parameters:**
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| filepath | string | Yes | - | Path to uploaded file |
| emotion_mode | string | No | "both" | "base", "granular", or "both" |
| include_timestamps | boolean | No | false | Include word-level timestamps |

**Success Response (200):**
```json
{
  "status": "success",
  "transcription": {
    "text": "Hello, how are you doing today?",
    "language": "English",
    "language_code": "en",
    "duration": 3.5
  },
  "emotions": {
    "base": [
      {"label": "Neutral", "score": "85.23%"},
      {"label": "Happiness", "score": "10.45%"},
      {"label": "Sadness", "score": "2.15%"},
      {"label": "Anger", "score": "1.23%"},
      {"label": "Fear", "score": "0.52%"},
      {"label": "Surprise", "score": "0.42%"}
    ],
    "primary": {
      "emotions": ["Neutral", "Calm"],
      "confidence": "85.23%"
    },
    "secondary": {
      "emotions": ["Indifference", "Composure"],
      "confidence": "59.66%"
    },
    "complex": []
  },
  "metadata": {
    "processing_time": 1.23,
    "audio_duration": 3.5,
    "model_versions": {
      "hubert": "base-ls960",
      "whisper": "large-v3"
    }
  }
}
```

---

### 4. Get Emotion Models Info

**Endpoint:** `GET /models/emotions`

**Description:** Get information about available emotion models

**Success Response (200):**
```json
{
  "base_emotions": ["Anger", "Fear", "Happiness", "Neutral", "Sadness", "Surprise"],
  "granular_emotions": {
    "Anger": ["Rage", "Fury", "Frustration", "Irritation", "Annoyance", "Resentment"],
    "Fear": ["Terror", "Panic", "Anxiety", "Worry", "Nervousness", "Dread"],
    "Happiness": ["Joy", "Delight", "Contentment", "Amusement", "Excitement", "Enthusiasm"],
    "Neutral": ["Calm", "Indifference", "Composure", "Serenity"],
    "Sadness": ["Grief", "Sorrow", "Melancholy", "Disappointment", "Despair", "Loneliness"],
    "Surprise": ["Amazement", "Shock", "Wonder", "Astonishment", "Disbelief"]
  },
  "complex_emotions": [
    "Elation",
    "Bitterness",
    "Desperation",
    "Contentment",
    "Panic",
    "Resignation"
  ]
}
```

---

### 5. Get Supported Languages

**Endpoint:** `GET /languages`

**Description:** Get list of supported languages for transcription

**Success Response (200):**
```json
{
  "count": 99,
  "languages": [
    {"code": "en", "name": "English"},
    {"code": "es", "name": "Spanish"},
    {"code": "fr", "name": "French"},
    {"code": "de", "name": "German"},
    {"code": "zh", "name": "Chinese"},
    ...
  ]
}
```

---

## WebSocket Events

### Connection

**Connect to WebSocket:**
```javascript
const socket = io('http://localhost:2024');

socket.on('connect', () => {
  console.log('Connected:', socket.id);
});
```

---

### Event: `analyze_batch`

**Description:** Start batch analysis of uploaded file

**Emit:**
```javascript
socket.emit('analyze_batch', {
  filepath: '/path/to/audio.wav',
  emotion_mode: 'both'
});
```

**Listen for Progress:**
```javascript
socket.on('status', (data) => {
  console.log(data.message);  // "Analyzing segment 1/5..."
  console.log(data.progress);  // 20
});
```

**Listen for Chunk Results:**
```javascript
socket.on('chunk_result', (data) => {
  console.log('Chunk:', data.chunk_index);
  console.log('Result:', data.result);
});
```

**Listen for Completion:**
```javascript
socket.on('analysis_complete', (data) => {
  console.log('Analysis finished:', data.message);
});
```

---

### Event: `streaming_chunk`

**Description:** Send audio chunk for real-time analysis

**Emit:**
```javascript
socket.emit('streaming_chunk', {
  audio_data: audioBlob,  // Blob or ArrayBuffer
  chunk_index: 0
});
```

**Listen for Results:**
```javascript
socket.on('streaming_result', (data) => {
  console.log('Real-time result:', data.result);
});
```

---

### Event: `save_recording`

**Description:** Save complete recording after streaming

**Emit:**
```javascript
socket.emit('save_recording', {
  recording_data: fullRecordingBlob
});
```

**Listen for Confirmation:**
```javascript
socket.on('save_complete', (data) => {
  console.log('Saved at:', data.filepath);
});
```

---

### Event: `error`

**Description:** Error notifications

**Listen:**
```javascript
socket.on('error', (data) => {
  console.error('Error:', data.message);
});
```

---

## Data Models

### EmotionResult

```typescript
interface EmotionResult {
  base: Array<{
    label: string;
    score: string;  // Percentage format: "85.23%"
  }>;
  primary?: {
    emotions: string[];
    confidence: string;
  };
  secondary?: {
    emotions: string[];
    confidence: string;
  };
  complex?: Array<{
    name: string;
    components: string[];
    confidence: string;
  }>;
}
```

### TranscriptionResult

```typescript
interface TranscriptionResult {
  text: string;
  language: string;
  language_code: string;
  duration?: number;
  word_timestamps?: Array<{
    word: string;
    start: number;
    end: number;
  }>;
}
```

### AnalysisResult

```typescript
interface AnalysisResult {
  status: 'success' | 'error';
  transcription: TranscriptionResult;
  emotions: EmotionResult;
  metadata?: {
    processing_time: number;
    audio_duration: number;
    model_versions: {
      hubert: string;
      whisper: string;
    };
  };
  message?: string;  // For errors
}
```

---

## Code Examples

### Python Client

```python
import requests
from pathlib import Path

class EmotiVoiceClient:
    def __init__(self, base_url='http://localhost:2024'):
        self.base_url = base_url
    
    def upload_file(self, file_path):
        """Upload audio file"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f'{self.base_url}/upload',
                files=files
            )
        return response.json()
    
    def analyze(self, filepath, emotion_mode='both'):
        """Analyze audio file"""
        data = {
            'filepath': filepath,
            'emotion_mode': emotion_mode
        }
        response = requests.post(
            f'{self.base_url}/analyze',
            json=data
        )
        return response.json()
    
    def analyze_file(self, file_path, emotion_mode='both'):
        """Upload and analyze in one call"""
        # Upload
        upload_result = self.upload_file(file_path)
        if not upload_result.get('success'):
            raise Exception(f"Upload failed: {upload_result}")
        
        # Analyze
        filepath = upload_result['filepath']
        return self.analyze(filepath, emotion_mode)

# Usage
client = EmotiVoiceClient()
result = client.analyze_file('recording.wav', emotion_mode='both')

print("Transcription:", result['transcription']['text'])
print("Dominant emotion:", result['emotions']['base'][0]['label'])
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');
const FormData = require('form-data');
const fs = require('fs');

class EmotiVoiceClient {
  constructor(baseURL = 'http://localhost:2024') {
    this.baseURL = baseURL;
  }
  
  async uploadFile(filePath) {
    const form = new FormData();
    form.append('file', fs.createReadStream(filePath));
    
    const response = await axios.post(
      `${this.baseURL}/upload`,
      form,
      { headers: form.getHeaders() }
    );
    
    return response.data;
  }
  
  async analyze(filepath, emotionMode = 'both') {
    const response = await axios.post(
      `${this.baseURL}/analyze`,
      { filepath, emotion_mode: emotionMode }
    );
    
    return response.data;
  }
  
  async analyzeFile(filePath, emotionMode = 'both') {
    // Upload
    const uploadResult = await this.uploadFile(filePath);
    if (!uploadResult.success) {
      throw new Error(`Upload failed: ${uploadResult.error}`);
    }
    
    // Analyze
    return await this.analyze(uploadResult.filepath, emotionMode);
  }
}

// Usage
(async () => {
  const client = new EmotiVoiceClient();
  const result = await client.analyzeFile('recording.wav', 'both');
  
  console.log('Transcription:', result.transcription.text);
  console.log('Dominant emotion:', result.emotions.base[0].label);
})();
```

### WebSocket Streaming (JavaScript)

```javascript
const socket = io('http://localhost:2024');
let mediaRecorder;
let chunkIndex = 0;

// Start recording
async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  
  mediaRecorder = new MediaRecorder(stream);
  
  mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
      // Send chunk for analysis
      socket.emit('streaming_chunk', {
        audio_data: event.data,
        chunk_index: chunkIndex++
      });
    }
  };
  
  // Record in 2-second chunks
  mediaRecorder.start(2000);
}

// Listen for results
socket.on('streaming_result', (data) => {
  console.log('Chunk', data.chunk_index);
  console.log('Transcription:', data.result.transcription);
  console.log('Emotions:', data.result.emotions);
});

// Stop recording
function stopRecording() {
  mediaRecorder.stop();
}
```

---

## Rate Limits

### Current Limits (Development)

- **Requests per minute**: Unlimited
- **File size**: 100MB max
- **Concurrent WebSocket connections**: 10

### Production Limits (Recommended)

- **Requests per minute**: 60
- **File size**: 50MB max
- **Concurrent connections**: 5 per user
- **Processing queue**: 100 requests

**Rate Limit Headers:**
```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1640995200
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid parameters |
| 404 | Not Found | Resource not found |
| 413 | Payload Too Large | File size exceeds limit |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

### Error Response Format

```json
{
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "filepath",
    "reason": "File not found"
  }
}
```

### Common Error Codes

| Code | Description | Solution |
|------|-------------|----------|
| `FILE_NOT_FOUND` | Uploaded file not found | Re-upload file |
| `INVALID_FILE_TYPE` | Unsupported audio format | Use WAV, MP3, or M4A |
| `FILE_TOO_LARGE` | File exceeds size limit | Compress or split audio |
| `PROCESSING_ERROR` | Analysis failed | Check audio quality |
| `MODEL_NOT_LOADED` | AI model unavailable | Restart server |

---

## Best Practices

### 1. Audio File Preparation

```python
# Recommended audio specifications
specs = {
    'format': 'WAV',
    'sample_rate': 16000,
    'channels': 1,  # Mono
    'bit_depth': 16,
    'duration': '<5 minutes per file'
}
```

### 2. Chunking Long Audio

```python
def split_long_audio(audio_path, chunk_duration=60):
    """Split audio into manageable chunks"""
    from pydub import AudioSegment
    
    audio = AudioSegment.from_file(audio_path)
    chunks = []
    
    for i in range(0, len(audio), chunk_duration * 1000):
        chunk = audio[i:i + chunk_duration * 1000]
        chunk_path = f'chunk_{i//1000}.wav'
        chunk.export(chunk_path, format='wav')
        chunks.append(chunk_path)
    
    return chunks
```

### 3. Retry Logic

```python
import time

def analyze_with_retry(client, filepath, max_retries=3):
    """Retry analysis on failure"""
    for attempt in range(max_retries):
        try:
            return client.analyze(filepath)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 4. Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor

def batch_analyze(client, file_paths, max_workers=4):
    """Process multiple files in parallel"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(
            lambda p: client.analyze_file(p),
            file_paths
        ))
    return results
```

---

## Webhooks (Future Feature)

### Register Webhook

```python
POST /webhooks/register
{
  "url": "https://yourapp.com/webhook",
  "events": ["analysis_complete", "error"],
  "secret": "your-webhook-secret"
}
```

### Webhook Payload

```json
{
  "event": "analysis_complete",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": {
    "filepath": "/path/to/audio.wav",
    "result": { ... }
  },
  "signature": "sha256=..."
}
```

---

## Support

- **Documentation**: https://docs.emotivoice.ai
- **API Status**: https://status.emotivoice.ai
- **GitHub Issues**: https://github.com/yourrepo/issues

---

**API Version**: 1.0  
**Last Updated**: January 2025