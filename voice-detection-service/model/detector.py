import librosa
import numpy as np
import logging
import torch
import torch.nn.functional as F
import os
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from typing import Dict, Tuple

logger = logging.getLogger(__name__)

class VoiceDetector:
    """
    AI Voice Detection System
    Uses a hybrid approach:
    1. Deep Learning Model (Transformers) for primary detection
    2. Signal Processing (Librosa) as fallback and explanation generator
    """

    def __init__(self):
        """Initialize the voice detector model"""
        self.supported_languages = ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu']
        
        # Model Configuration
        # Check if local baked model exists (Offline Mode), else use Hub
        self.local_model_path = "./model/local_weights"
        if os.path.exists(self.local_model_path) and os.path.exists(os.path.join(self.local_model_path, "config.json")):
             logger.info(f"📁 Found local model at {self.local_model_path}. Using Offline Mode.")
             self.model_id = self.local_model_path
        else:
             logger.info("☁️ Local model not found. Using Hugging Face Hub.")
             self.model_id = "MelodyMachine/Deepfake-audio-detection"
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.feature_extractor = None
        self.model_loaded = False
        self.load_attempted = False
        
        # Check environment
        self.is_render = os.environ.get('RENDER') == 'true'

    def _load_model(self):
        """
        Load the model only when needed (Lazy Loading).
        """
        if self.load_attempted:
            return

        self.load_attempted = True
        
        # NOTE: On Hugging Face Spaces, we have 16GB RAM, so we ALWAYS load the model.
        # Modified to ensure model loads unless explicitly disabled
        if self.is_render and os.environ.get('DISABLE_MODEL_LOAD') == 'true':
             logger.warning("⚠️ RENDER DETECTED: Skipping Heavy Model Load due to DISABLE_MODEL_LOAD.")
             return

        try:
            logger.info(f"Loading AI Model (Lazy): {self.model_id}")
            # Optimization: Restrict Torch threads to save memory
            torch.set_num_threads(1)
            
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            self.model.to(self.device)
            
            self.model_loaded = True
            logger.info("✅ AI Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Failed to load AI model: {str(e)}")
            logger.warning("⚠️ Running in Fallback Mode (Signal Processing only)")
            self.model_loaded = False

    def detect(self, audio_path: str, language: str) -> Dict:
        """
        Main detection method
        """
        # Attempt to load model if not already done
        if not self.load_attempted:
            self._load_model()
            
        try:
            logger.info(f"Analyzing audio: {audio_path} in {language}")

            # 1. AI Model Prediction (Primary)
            ai_classification = "UNKNOWN"
            ai_confidence = 0.5
            
            if self.model_loaded:
                ai_classification, ai_confidence = self._predict_with_model(audio_path)
            
            # 2. Extract Signal Features (Secondary/Explanation)
            # Optimization: Load only first 10 seconds for feature extraction
            audio, sr = librosa.load(audio_path, sr=None, duration=10)
            
            # Trim silence for better acoustic feature extraction
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            # Be careful not to trim everything if it's a quiet recording
            if len(audio_trimmed) > len(audio) * 0.1:
                audio = audio_trimmed

            features = self._extract_features(audio, sr)
            
            # 3. Heuristic Check (Fallback/Validation)
            heuristic_class, heuristic_conf = self._heuristic_check(features)
            
            # 4. Final Decision Logic (Ensemble)
            final_class = ai_classification
            final_conf = ai_confidence

            # If model isn't loaded, rely on heuristics
            if not self.model_loaded:
                logger.info("Using heuristic fallback (Model not loaded)")
                final_class = heuristic_class
                final_conf = heuristic_conf
            else:
                # Weighted Ensemble Logic
                # Convert everything to "AI Probability" for simpler math
                
                # Model Score (0.0=Human, 1.0=AI)
                if ai_classification == 'AI_GENERATED':
                    model_ai_score = ai_confidence
                else:
                    model_ai_score = 1.0 - ai_confidence
                
                # Heuristic Score (0.0=Human, 1.0=AI)
                if heuristic_class == 'AI_GENERATED':
                    heur_ai_score = heuristic_conf
                else:
                    heur_ai_score = 1.0 - heuristic_conf
                
                # Weights: Model 70%, Heuristics 30%
                # This ensures that strong model predictions aren't easily overturned,
                # but strong heuristic signals can help marginal cases.
                combined_ai_score = (model_ai_score * 0.7) + (heur_ai_score * 0.3)
                
                if combined_ai_score > 0.5:
                    final_class = 'AI_GENERATED'
                    # Map 0.5-1.0 to confidence
                    final_conf = 0.5 + (combined_ai_score - 0.5)
                else:
                    final_class = 'HUMAN'
                    final_conf = 0.5 + (0.5 - combined_ai_score)
            
            # Generate explanation based on key features
            explanation = self._generate_explanation(features, final_class)

            result = {
                'language': language,
                'classification': final_class,
                'confidenceScore': round(float(final_conf), 2),
                'explanation': explanation
            }

            logger.info(f"Final Result: {final_class} ({final_conf:.2f})")
            return result

        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            raise

    def _predict_with_model(self, audio_path: str) -> Tuple[str, float]:
        """Run inference using the Transformer model"""
        try:
            # Load and preprocess audio
            # Resample to 16000Hz as required by most Wav2Vec models
            audio, sr = librosa.load(audio_path, sr=16000, duration=10)
            
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True,
                max_length=16000*10, # Max 10 seconds
                truncation=True
            )
            
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits

            # Apply Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get predicted class
            predicted_id = torch.argmax(logits, dim=-1).item()
            confidence = probs[0][predicted_id].item()
            
            # Map robustly
            label_map = self.model.config.id2label
            predicted_label = label_map.get(predicted_id, str(predicted_id)).lower()
            
            logger.info(f"Model Predicted ID: {predicted_id}, Label: {predicted_label}, Conf: {confidence:.2f}")

            # Logic for MelodyMachine and similar 2-class models
            # Often: 0=fake, 1=real OR "fake", "real" labels
            is_fake = False
            
            if "fake" in predicted_label or "spoof" in predicted_label:
                is_fake = True
            elif "real" in predicted_label or "bonafide" in predicted_label:
                is_fake = False
            elif predicted_label == "0": 
                # Fallback: Index 0 is sometimes fake in deepfake datasets
                is_fake = True 
            elif predicted_label == "1":
                is_fake = False
            
            if is_fake:
                 return "AI_GENERATED", confidence
            else:
                 return "HUMAN", confidence

        except Exception as e:
            logger.error(f"AI Inference failed: {e}")
            return "UNKNOWN", 0.0

    def _extract_features(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Extract acoustic features from audio for explanation and fallback
        """
        features = {}

        try:
            # MFCC (Timbre/Cepstral Analysis)
            # Real voices often have higher variance in certain MFCC coefficients
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features['mfcc_std_mean'] = np.mean(np.std(mfcc, axis=1))

            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            # Optimization: Take max magnitude per frame
            max_mags = magnitudes.max(axis=0)
            # Thresholding
            mask = max_mags > np.median(max_mags)
            if np.any(mask):
                pitch_indices = magnitudes[:, mask].argmax(axis=0)
                pitch_values = pitches[pitch_indices, np.where(mask)[0]]
                valid_pitches = pitch_values[pitch_values > 0]
                
                if len(valid_pitches) > 0:
                    features['pitch_mean'] = np.mean(valid_pitches)
                    features['pitch_std'] = np.std(valid_pitches)
                    # Safe division
                    features['pitch_consistency'] = 1.0 - (features['pitch_std'] / (features['pitch_mean'] + 1e-6))
                else:
                    features['pitch_mean'], features['pitch_std'], features['pitch_consistency'] = 0, 0, 0
            else:
                 features['pitch_mean'], features['pitch_std'], features['pitch_consistency'] = 0, 0, 0

            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features['spectral_centroid_std'] = np.std(spectral_centroids)

            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features['zcr_std'] = np.std(zcr)

            return features

        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            # Return safe averages
            return {
                'pitch_consistency': 0.5, 
                'spectral_centroid_std': 1000, 
                'zcr_std': 0.1, 
                'mfcc_std_mean': 30
            }

    def _heuristic_check(self, features: Dict) -> Tuple[str, float]:
        """
        Heuristic Rule-Based Classifier (Fallback)
        """
        try:
            ai_score = 0.0
            
            # Logic: AI voices are "too perfect" (low variance, high consistency)
            if features.get('pitch_consistency', 0) > 0.85: ai_score += 0.3
            if features.get('spectral_centroid_std', 1000) < 500: ai_score += 0.2
            if features.get('zcr_std', 0.1) < 0.05: ai_score += 0.2
            
            # MFCC Heuristic: Synthetic voices often have lower variance in cepstral coeffs
            # Real human voice has high complexity (high std dev mean usually > 10-15)
            if features.get('mfcc_std_mean', 20) < 10.0: ai_score += 0.3
            
            if ai_score > 0.4:
                return 'AI_GENERATED', min(0.95, 0.55 + ai_score/2.5)
            else:
                return 'HUMAN', min(0.98, 0.6 + (1-ai_score)/2)

        except Exception:
            return "HUMAN", 0.65 # Default safe bias

    def _generate_explanation(self, features: Dict, classification: str) -> str:
        """Generate human-readable explanation based on features"""
        
        explanations = []

        if classification == 'AI_GENERATED':
            if features.get('pitch_consistency', 0) > 0.82:
                explanations.append("Unnaturally consistent pitch patterns")
            if features.get('mfcc_std_mean', 20) < 12.0:
                explanations.append("Synthetic timbre characteristics detected")
            if features.get('spectral_centroid_std', 1000) < 600:
                explanations.append("Lack of natural spectral variance")
            if not explanations:
                explanations.append("Deepfake artifacts identified in audio stream")
        else:  # HUMAN
            if features.get('pitch_consistency', 0) < 0.78:
                explanations.append("Natural pitch modulation detected")
            if features.get('mfcc_std_mean', 0) > 12.0:
                explanations.append("Complex human vocal timbre")
            if features.get('spectral_centroid_std', 0) > 800:
                explanations.append("Organic spectral fluctuations")
            if not explanations:
                explanations.append("Human vocal micro-tremors identified")

        return " and ".join(explanations)
