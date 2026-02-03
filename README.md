# 🎙️ AI Voice Detection API

Detect AI-generated vs human voices across 5 Indian languages.

**Problem Statement:** AI-Generated Voice Detection  
**Languages:** Tamil, English, Hindi, Malayalam, Telugu  
**Architecture:** Microservices (Node.js + Python)

---

## 🚀 Quick Start

**Terminal 1 - Start Node.js API:**
```bash
cd node-api
npm install
npm start
```

**Terminal 2 - Start Python Voice Service:**
```bash
cd python-voice-service
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Services will run at:
- **API Gateway:** http://localhost:3000
- **Voice Service:** http://localhost:5001

---

## 📡 API Usage

### Authentication
All requests require an API key:
```
x-api-key: sk_test_123456789
```

### Voice Detection Endpoint

**POST** `/api/voice-detection`

```bash
curl -X POST http://localhost:3000/api/voice-detection ^
  -H "Content-Type: application/json" ^
  -H "x-api-key: sk_test_123456789" ^
  -d "{\"language\": \"English\", \"audioFormat\": \"mp3\", \"audioBase64\": \"YOUR_BASE64_AUDIO_HERE\"}"
```

**Request Body:**

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `language` | string | Yes | `Tamil`, `English`, `Hindi`, `Malayalam`, `Telugu` |
| `audioFormat` | string | Yes | `mp3` |
| `audioBase64` | string | Yes | Base64-encoded audio |

**Success Response (200):**
```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.91,
  "explanation": "Unnatural pitch consistency and robotic speech patterns detected."
}
```

### Health Check

```bash
curl http://localhost:3000/health
```

---

## 📂 Project Structure

```
├── node-api/                    # API Gateway (Node.js/Express)
│   ├── src/
│   │   ├── index.js
│   │   ├── routes/voiceRoutes.js
│   │   ├── middleware/
│   │   │   ├── auth.js
│   │   │   ├── validator.js
│   │   │   └── errorHandler.js
│   │   └── services/
│   │       ├── voiceService.js
│   │       └── logger.js
│   └── package.json
│
├── python-voice-service/        # ML Service (Python/Flask)
│   ├── model/detector.py
│   ├── app.py
│   └── requirements.txt
│
├── convert_audio.py             # Audio to Base64 converter
└── postman_collection.json      # Postman tests
```

---

## 🧪 Testing

```bash
# Convert audio file to Base64
python convert_audio.py path/to/audio.mp3
```

Import `postman_collection.json` into Postman for interactive testing.

---

## ⚙️ Configuration

**node-api/.env:**
```env
PORT=3000
API_KEY=sk_test_123456789
VOICE_SERVICE_URL=http://localhost:5001
```

**python-voice-service/.env:**
```env
PORT=5001
LOG_LEVEL=INFO
```

---

## 🔒 Security Features

- ✅ API Key authentication
- ✅ Request validation
- ✅ Rate limiting (100 req/15min)
- ✅ Security headers (Helmet.js)
- ✅ CORS protection

---

## 📊 ML Features Analyzed

- Pitch consistency & frequency
- Spectral centroid & rolloff
- MFCC (13 coefficients)
- Zero-crossing rate
- RMS energy patterns

---

**Built for GUVI Hackathon 2026** 🚀
