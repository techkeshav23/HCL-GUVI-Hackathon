
import sys
import logging

# Configure logging to stdout
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ModelVerifier")

print("🔍 Starting Model Availability Check...")

try:
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    import torch
    import transformers
    print(f"✅ Libraries imported successfully.")
    print(f"   Torch version: {torch.__version__}")
    print(f"   Transformers version: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Critical Library Missing: {e}")
    sys.exit(1)

def test_model_load(model_id):
    print(f"\n🧪 Testing Model ID: '{model_id}'")
    try:
        print("   Downloading/Loading Feature Extractor...")
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        
        print("   Downloading/Loading Model Weights...")
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        
        print(f"✅ SUCCESS: Model '{model_id}' is available and public!")
        return True
    except Exception as e:
        print(f"❌ FAILURE: Could not load '{model_id}'")
        print(f"   Error: {str(e)}")
        return False

# 1. Test the primary model from our code
primary_model = "MelodyMachine/Deepfake-audio-detection"
success = test_model_load(primary_model)

if success:
    print("\n🎉 GREAT NEWS: The primary model is working perfectly.")
    print("   Your code will use the AI Model path (Best Case).")
else:
    print("\n⚠️  WARNING: Primary model failed.")
    print("   Your code will use the Heuristic Fallback (Worst Case).")
    
    # 2. Test a generic fallback just in case user wants to switch
    print("\n🔄 Checking an alternative popular model (just in case)...")
    fallback_model = "facebook/wav2vec2-base-960h"
    test_model_load(fallback_model)
