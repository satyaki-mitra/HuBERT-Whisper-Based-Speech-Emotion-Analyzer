# EmotiVoice API Documentation

**Version:** 1.0.0  
**Base URL:** `/api/v1`

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Error Handling](#error-handling)
4. [Endpoints](#endpoints)
   - [Health & Status](#health--status)
   - [File Upload](#file-upload)
   - [Audio Analysis](#audio-analysis)
   - [Explainability](#explainability)
   - [Visualizations](#visualizations)
   - [Export](#export)
5. [WebSocket Events](#websocket-events)
6. [Data Models](#data-models)

---

## Overview

The EmotiVoice API provides comprehensive speech emotion recognition and transcription capabilities powered by HuBERT and Whisper AI models. The API supports both synchronous and asynchronous processing modes.

### Key Features

- **6 Base Emotions:** Anger, Fear, Happiness, Neutral, Sadness, Surprise
- **20+ Granular Emotions:** Detailed emotional states mapped from base emotions
- **Complex Emotion Detection:** Elation, Bitterness, Desperation, etc.
- **Multi-language Transcription:** 90+ languages supported via Whisper
- **Explainable AI:** SHAP values, LIME explanations, attention visualizations
- **Real-time Processing:** WebSocket support for streaming analysis

### Rate Limits

- **Default:** 100 requests per hour per IP
- **File Size Limit:** 100MB per upload
- **Supported Formats:** WAV, MP3, M4A, FLAC, OGG, WEBM

---

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

---

## Error Handling

### Error Response Format

```json
{
  "status": "error",
  "message": "Human-readable error message",
  "details": [
    {
      "field": "field_name",
      "message": "Detailed error description",
      "error_type": "ValidationError"
    }
  ],
  "timestamp": "2025-01-08T12:34:56.789Z"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 202 | Accepted (async processing) |
| 400 | Bad Request (validation error) |
| 404 | Resource Not Found |
| 422 | Unprocessable Entity (audio processing error) |
| 500 | Internal Server Error |
| 503 | Service Unavailable (models not loaded) |

---

## Endpoints

### Health & Status

#### `GET /api/v1/health`

Check service health and model status.

**Response:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "models_loaded": true,
  "redis_connected": true,
  "timestamp": "2025-01-08T12:34:56.789Z"
}
```

**Status Codes:**
- `200`: Service healthy
- `503`: Service unhealthy

---

#### `GET /api/v1/status/{task_id}`

Get status of an asynchronous analysis task.

**Parameters:**
- `task_id` (path): UUID of the task

**Response (Processing):**

```json
{
  "status": "processing",
  "progress": 45,
  "message": "Analyzing audio..."
}
```

**Response (Completed):**

```json
{
  "status": "completed",
  "progress": 100,
  "message": "Task completed",
  "result": {
    "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
    "transcription": "This is the transcribed text",
    "language": "English",
    "emotions": {
      "base": [
        {
          "label": "Happiness",
          "score": 75.32,
          "percentage": "75.32%"
        }
      ]
    },
    "metadata": {
      "processing_time": 2.45
    }
  }
}
```

---

### File Upload

#### `POST /api/v1/upload`

Upload an audio file for analysis.

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (form-data): Audio file (required)

**Request Example:**

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@audio.wav"
```

**Response:**

```json
{
  "success": true,
  "filepath": "/app/data/uploaded_files/audio_a1b2c3d4.wav",
  "filename": "audio_a1b2c3d4.wav"
}
```

**Status Codes:**
- `200`: Upload successful
- `400`: No file provided or invalid file type
- `500`: Upload failed

---

### Audio Analysis

#### `POST /api/v1/analyze`

Analyze a single audio file (asynchronous).

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "filepath": "/app/data/uploaded_files/audio.wav",
  "emotion_mode": "both",
  "language": null
}
```

**Parameters:**
- `filepath` (string, required): Path to uploaded audio file
- `emotion_mode` (string, optional): `"base"`, `"granular"`, or `"both"` (default: `"both"`)
- `language` (string, optional): Target language code (e.g., `"en"`, `"es"`) or `null` for auto-detect

**Response:**

```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "pending",
  "created_at": "2025-01-08T12:34:56.789Z",
  "task_id": "celery-task-uuid",
  "status_url": "/api/v1/status/celery-task-uuid"
}
```

**Status Codes:**
- `202`: Analysis queued
- `400`: Invalid request
- `404`: File not found

---

### Explainability

#### `GET /api/v1/explain/{analysis_id}`

Get explainability results for a completed analysis.

**Parameters:**
- `analysis_id` (path): UUID of the analysis

**Response:**

```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "available": true,
  "visualization_urls": {
    "emotion_distribution": "/api/v1/visualizations/550e8400.../emotion_distribution",
    "shap_importance": "/api/v1/visualizations/550e8400.../shap_importance",
    "lime_contributions": "/api/v1/visualizations/550e8400.../lime_contributions",
    "attention": "/api/v1/visualizations/550e8400.../attention"
  },
  "visualizations": {
    "550e8400..._emotion_distribution": "/path/to/viz.png"
  }
}
```

**Status Codes:**
- `200`: Explainability data available
- `404`: Analysis not found

---

### Visualizations

#### `GET /api/v1/visualizations/{analysis_id}`

List all available visualizations for an analysis.

**Parameters:**
- `analysis_id` (path): UUID of the analysis

**Response:**

```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "visualizations": {
    "emotion_distribution": {
      "url": "/api/v1/visualizations/550e8400.../emotion_distribution",
      "type": "emotion_distribution",
      "filename": "550e8400..._emotion_distribution.png",
      "layer_info": null
    },
    "shap_importance": {
      "url": "/api/v1/visualizations/550e8400.../shap_importance",
      "type": "shap_importance",
      "filename": "550e8400..._shap_importance.png",
      "layer_info": null
    }
  },
  "count": 4
}
```

---

#### `GET /api/v1/visualizations/{analysis_id}/{viz_type}`

Get a specific visualization image.

**Parameters:**
- `analysis_id` (path): UUID of the analysis
- `viz_type` (path): Visualization type (`emotion_distribution`, `shap_importance`, `lime_contributions`, `attention`)

**Response:**
- Content-Type: `image/png`
- Returns PNG image data

**Status Codes:**
- `200`: Image returned
- `404`: Visualization not found

---

#### `GET /api/v1/visualizations/{analysis_id}/all`

Get all visualization URLs in a single request.

**Response:**

```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "visualization_urls": {
    "emotion_distribution": "/api/v1/visualizations/550e8400.../emotion_distribution",
    "shap_importance": "/api/v1/visualizations/550e8400.../shap_importance",
    "lime_contributions": "/api/v1/visualizations/550e8400.../lime_contributions",
    "attention": "/api/v1/visualizations/550e8400.../attention"
  },
  "available_types": ["emotion_distribution", "shap_importance", "lime_contributions", "attention"]
}
```

---

### Export

#### `POST /api/v1/export`

Export analysis results in various formats.

**Content-Type:** `application/json`

**Request Body:**

```json
{
  "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
  "format": "json",
  "result_data": {}
}
```

**Parameters:**
- `analysis_id` (string, required): UUID of the analysis
- `format` (string, optional): `"json"`, `"csv"`, or `"pdf"` (default: `"json"`)
- `result_data` (object, optional): Analysis result data to export

**Response:**

```json
{
  "success": true,
  "export_path": "/app/exports/550e8400....json",
  "download_url": "/api/v1/download/550e8400....json"
}
```

---

#### `GET /api/v1/download/{filename}`

Download an exported file.

**Parameters:**
- `filename` (path): Name of the exported file

**Response:**
- File download with appropriate content type

**Status Codes:**
- `200`: File download started
- `403`: Access denied (directory traversal attempt)
- `404`: File not found

---

## WebSocket Events

### Connection

```javascript
const socket = io('http://localhost:8000');

socket.on('connect', () => {
  console.log('Connected to EmotiVoice');
});
```

---

### Batch Analysis

#### Client → Server: `analyze_batch`

Start batch analysis with automatic audio segmentation.

**Payload:**

```json
{
  "filepath": "/app/data/uploaded_files/audio.wav",
  "emotion_mode": "both"
}
```

#### Server → Client: `status`

Progress updates during analysis.

**Payload:**

```json
{
  "message": "Analyzing segment 3/10...",
  "progress": 30
}
```

#### Server → Client: `chunk_result`

Result for each audio segment.

**Payload:**

```json
{
  "chunk_index": 0,
  "total_chunks": 10,
  "result": {
    "transcription": "This is segment text",
    "language": "English",
    "emotions": {
      "base": [...]
    },
    "analysis_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

#### Server → Client: `analysis_complete`

All segments processed.

**Payload:**

```json
{
  "message": "Complete"
}
```

---

### Live Streaming

#### Client → Server: `streaming_chunk`

Send real-time audio chunk for analysis.

**Payload:**

```json
{
  "audio_data": "<binary_data>",
  "chunk_index": 0
}
```

#### Server → Client: `streaming_result`

Real-time analysis result.

**Payload:**

```json
{
  "chunk_index": 0,
  "result": {
    "transcription": "Real-time text",
    "language": "English",
    "emotions": {
      "base": [...]
    },
    "analysis_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

#### Client → Server: `save_recording`

Save complete recording after streaming session.

**Payload:**

```json
{
  "recording_data": "<binary_data>"
}
```

#### Server → Client: `save_complete`

Recording saved successfully.

**Payload:**

```json
{
  "filepath": "/app/data/recorded_files/recording_xyz.wav"
}
```

---

### Error Events

#### Server → Client: `error`

Error occurred during processing.

**Payload:**

```json
{
  "message": "Error description"
}
```

#### Server → Client: `chunk_skipped`

Audio chunk skipped due to processing error.

**Payload:**

```json
{
  "chunk_index": 5,
  "reason": "conversion_failed"
}
```

---

## Data Models

### EmotionScore

```json
{
  "label": "Happiness",
  "score": 75.32,
  "percentage": "75.32%"
}
```

### GranularEmotion

```json
{
  "emotions": ["Joy", "Delight", "Happiness"],
  "confidence": "75.32%"
}
```

### ComplexEmotion

```json
{
  "name": "Elation",
  "components": ["Happiness", "Surprise"],
  "confidence": "68.45%"
}
```

### TranscriptionResult

```json
{
  "text": "Transcribed speech text",
  "language": "English",
  "language_code": "en",
  "confidence": 0.95
}
```

### EmotionResult

```json
{
  "base": [
    {
      "label": "Happiness",
      "score": 75.32,
      "percentage": "75.32%"
    }
  ],
  "primary": {
    "emotions": ["Joy", "Delight"],
    "confidence": "75.32%"
  },
  "secondary": {
    "emotions": ["Contentment", "Amusement"],
    "confidence": "52.73%"
  },
  "complex": [
    {
      "name": "Elation",
      "components": ["Happiness", "Surprise"],
      "confidence": "68.45%"
    }
  ]
}
```

### AnalysisResult

```json
{
  "transcription": {
    "text": "Transcribed text",
    "language": "English",
    "language_code": "en"
  },
  "emotions": {
    "base": [...],
    "primary": {...},
    "secondary": {...},
    "complex": [...]
  },
  "metadata": {
    "analysis_id": "550e8400-e29b-41d4-a716-446655440000",
    "processing_time": 2.45,
    "audio_path": "/app/data/uploaded_files/audio.wav"
  },
  "processing_time": 2.45
}
```

---

## Usage Examples

### Example 1: Single File Analysis

```python
import requests
import time

# 1. Upload file
with open('audio.wav', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/upload',
        files={'file': f}
    )
upload_data = response.json()
filepath = upload_data['filepath']

# 2. Start analysis
response = requests.post(
    'http://localhost:8000/api/v1/analyze',
    json={
        'filepath': filepath,
        'emotion_mode': 'both'
    }
)
task_data = response.json()
task_id = task_data['task_id']

# 3. Poll for results
while True:
    response = requests.get(f'http://localhost:8000/api/v1/status/{task_id}')
    status_data = response.json()
    
    if status_data['status'] == 'completed':
        result = status_data['result']
        print(f"Transcription: {result['transcription']}")
        print(f"Dominant Emotion: {result['emotions']['base'][0]['label']}")
        break
    
    time.sleep(2)

# 4. Get explainability
analysis_id = result['analysis_id']
response = requests.get(f'http://localhost:8000/api/v1/explain/{analysis_id}')
explain_data = response.json()
print(f"Visualizations: {explain_data['visualization_urls']}")
```

### Example 2: WebSocket Streaming

```javascript
const socket = io('http://localhost:8000');

// Start recording
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    const mediaRecorder = new RecordRTC(stream, {
      type: 'audio',
      mimeType: 'audio/webm',
      timeSlice: 10000,
      ondataavailable: (blob) => {
        socket.emit('streaming_chunk', {
          audio_data: blob,
          chunk_index: chunkIndex++
        });
      }
    });
    
    mediaRecorder.startRecording();
  });

// Receive results
socket.on('streaming_result', (data) => {
  console.log('Transcription:', data.result.transcription);
  console.log('Emotions:', data.result.emotions.base);
});
```

---

## Notes

- All timestamps are in UTC ISO 8601 format
- File paths are server-side paths and should not be manipulated by clients
- WebSocket connections timeout after 600 seconds of inactivity
- Analysis results are stored temporarily and may be cleaned up after 24 hours
- Visualizations are cached with 1-hour TTL

---

**For support or questions, please refer to the project repository.**