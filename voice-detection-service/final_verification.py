import requests
import base64
import os
import json

# --- CONFIGURATION ---
API_URL = "https://keshav166-voice-detection-api.hf.space/api/voice-detection"
API_KEY = "sk_test_123456789"
AUDIO_FILE = "test_audio.mp3" 

def verify_api():
    print(f"🔍 Verifying API for Submission...")
    print(f"📍 URL: {API_URL}")
    
    # 1. Check File
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Error: {AUDIO_FILE} not found.")
        return

    # 2. Encode Audio
    print(f"⏳ Encoding {AUDIO_FILE}...")
    with open(AUDIO_FILE, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')
    
    # 3. Prepare Request (Exactly as per submission form)
    payload = {
        "language": "Hindi",  # Testing with Hindi
        "audioFormat": "mp3",
        "audioBase64": audio_b64
    }
    
    headers = {
        "x-api-key": API_KEY,
        "Content-Type": "application/json"
    }

    # 4. Send Request
    print(f"🚀 Sending Request... (Please wait)")
    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
        
        print("\n" + "="*30)
        print(f"STATUS CODE: {response.status_code}")
        print("="*30)
        
        if response.status_code == 200:
            print("✅ SUCCESS! API is working perfectly.")
            print("Response Data:")
            print(json.dumps(response.json(), indent=2))
        else:
            print("❌ FAILED!")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Exception: {e}")

if __name__ == "__main__":
    verify_api()
