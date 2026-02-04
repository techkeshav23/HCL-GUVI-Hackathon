---
title: Voice Detection API
emoji: 🎙️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# 🎙️ AI Voice Detection API

Detect AI-generated vs human voices across 5 Indian languages using Deep Learning.

## 🚀 API Usage

**Endpoint:** `POST /api/voice-detection`

**Headers:**
```
x-api-key: sk_test_123456789
Content-Type: application/json
```

**Body:**
```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<BASE64_STRING>"
}
```
