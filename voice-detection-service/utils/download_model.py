import os
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

model_id = "MelodyMachine/Deepfake-audio-detection"
save_directory = "./model/local_weights"

print(f"Downloading model: {model_id}...")

try:
    # Download and save to local directory
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    feature_extractor.save_pretrained(save_directory)
    
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.save_pretrained(save_directory)
    
    print(f"✅ Model successfully saved to {save_directory}")
except Exception as e:
    print(f"❌ Failed to download model: {e}")
    exit(1)
