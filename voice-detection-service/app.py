from flask import Flask, request, jsonify
from flask_cors import CORS
from functools import wraps
import base64
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from model.detector import VoiceDetector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
API_KEY = os.environ.get('API_KEY', 'sk_test_123456789')
SUPPORTED_LANGUAGES = ['Tamil', 'English', 'Hindi', 'Malayalam', 'Telugu']

# Initialize detector
detector = VoiceDetector()

# Create audio directory if it doesn't exist
os.makedirs('audio/temp', exist_ok=True)


# ==================== MIDDLEWARE ====================

def require_api_key(f):
    """API Key authentication decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('x-api-key')
        
        if not api_key:
            logger.warn('Request received without API key')
            return jsonify({
                'status': 'error',
                'message': 'API key is required'
            }), 401
        
        if api_key != API_KEY:
            logger.warn(f'Invalid API key attempt: {api_key[:10]}...')
            return jsonify({
                'status': 'error',
                'message': 'Invalid API key'
            }), 403
        
        logger.info('API key validated successfully')
        return f(*args, **kwargs)
    return decorated_function


def validate_voice_request(data):
    """Validate voice detection request body"""
    errors = []
    
    language = data.get('language')
    audio_format = data.get('audioFormat')
    audio_base64 = data.get('audioBase64')
    
    # Validate language
    if not language:
        errors.append('Language is required')
    elif language not in SUPPORTED_LANGUAGES:
        errors.append(f'Language must be one of: {", ".join(SUPPORTED_LANGUAGES)}')
    
    # Validate audio format
    if not audio_format:
        errors.append('audioFormat is required')
    elif audio_format != 'mp3':
        errors.append('audioFormat must be mp3')
    
    # Validate audio base64
    if not audio_base64:
        errors.append('audioBase64 is required')
    elif not isinstance(audio_base64, str) or len(audio_base64) == 0:
        errors.append('audioBase64 must be a non-empty string')
    
    return errors


# ==================== ROUTES ====================

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'status': 'online',
        'message': 'AI Voice Detection API is running. Use POST /api/voice-detection to detect voices.',
        'documentation': 'See README.md for usage details'
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'AI Voice Detection API',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat()
    })


@app.route('/api/voice-detection', methods=['POST'])
@require_api_key
def voice_detection():
    """
    Main API endpoint for voice detection (as per problem statement)
    
    Headers Required:
        x-api-key: Your API key
    
    Request Body:
        {
            "language": "Tamil|English|Hindi|Malayalam|Telugu",
            "audioFormat": "mp3",
            "audioBase64": "base64_encoded_audio_string"
        }
    
    Response:
        {
            "status": "success",
            "language": "Tamil",
            "classification": "AI_GENERATED" or "HUMAN",
            "confidenceScore": 0.91,
            "explanation": "Reason for classification"
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Request body is required'
            }), 400
        
        # Validate request
        validation_errors = validate_voice_request(data)
        if validation_errors:
            logger.warn(f'Validation failed: {validation_errors}')
            return jsonify({
                'status': 'error',
                'message': 'Invalid API key or malformed request',
                'errors': validation_errors
            }), 400
        
        language = data.get('language')
        audio_format = data.get('audioFormat')
        audio_base64 = data.get('audioBase64')
        
        logger.info(f'Voice detection request for language: {language}')
        
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            logger.error(f'Base64 decode error: {str(e)}')
            return jsonify({
                'status': 'error',
                'message': 'Invalid API key or malformed request'
            }), 400
        
        # Save temporary audio file
        temp_file = f"audio/temp/{datetime.now().timestamp()}.{audio_format}"
        with open(temp_file, 'wb') as f:
            f.write(audio_bytes)
        
        try:
            # Perform detection
            result = detector.detect(temp_file, language)
            
            # Clean up temp file
            os.remove(temp_file)
            
            # Return response in exact format required
            return jsonify({
                'status': 'success',
                'language': result['language'],
                'classification': result['classification'],
                'confidenceScore': result['confidenceScore'],
                'explanation': result['explanation']
            }), 200
        
        except Exception as e:
            logger.error(f'Detection error: {str(e)}')
            # Clean up temp file on error
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            return jsonify({
                'status': 'error',
                'message': f'Detection failed: {str(e)}'
            }), 500
    
    except Exception as e:
        logger.error(f'Request processing error: {str(e)}')
        return jsonify({
            'status': 'error',
            'message': 'Invalid API key or malformed request'
        }), 500


# Legacy endpoint (for backward compatibility)
@app.route('/detect', methods=['POST'])
def detect_voice_legacy():
    """Legacy endpoint - redirects to main API"""
    return voice_detection()


# ==================== 404 HANDLER ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'status': 'error',
        'message': 'Endpoint not found'
    }), 404


# ==================== SERVER START ====================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    
    logger.info("=" * 50)
    logger.info("🎯 AI Voice Detection API")
    logger.info("=" * 50)
    logger.info(f"📡 API Endpoint: http://localhost:{port}/api/voice-detection")
    logger.info(f"🔑 API Key: {API_KEY[:10]}...")
    logger.info(f"🌍 Supported Languages: {', '.join(SUPPORTED_LANGUAGES)}")
    logger.info("=" * 50)
    
    app.run(host='0.0.0.0', port=port, debug=False)

