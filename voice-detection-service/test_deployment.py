import requests
import base64
import os
import sys

# CONFIGURATION
API_URL = "https://keshav166-voice-detection-api.hf.space/api/voice-detection"
API_KEY = "sk_test_123456789"
AUDIO_FILE = "test_audio.mp3"  # Make sure this file exists!

def test_api():
    print("="*50)
    print(f"🚀 Testing API: {API_URL}")
    print("="*50)

    # 1. Check if audio file exists
    if not os.path.exists(AUDIO_FILE):
        print(f"❌ Error: '{AUDIO_FILE}' not found!")
        print("Please place a sample MP3 file in this folder and name it 'test_audio.mp3'")
        return

    # 2. Convert Audio to Base64
    print("Converting audio to Base64...")
    try:
        with open(AUDIO_FILE, "rb") as audio_file:
            encoded_string = base64.b64encode(audio_file.read()).decode('utf-8')
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return

    # 3. Prepare Payload
    payload = {
        "language": "English",
        "audioFormat": "mp3",
        "audioBase64": encoded_string
    }
    
    headers = {
        "Content-Type": "application/json",
        "x-api-key": API_KEY
    }

    # 4. Send Request
    print("📡 Sending request to Render... (This might take 30s for the first run)")
    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        
        # 5. Print Result
        print("\n" + "-"*20 + " RESPONSE " + "-"*20)
        print(f"Status Code: {response.status_code}")
        try:
            print(response.json())
        except:
            print(response.text)
            
    except Exception as e:
        print(f"❌ Connection Error: {e}")

if __name__ == "__main__":
    test_api()
