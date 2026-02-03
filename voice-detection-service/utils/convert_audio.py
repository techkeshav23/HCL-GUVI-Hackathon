#!/usr/bin/env python3
"""
Audio to Base64 Converter
Converts MP3 audio files to base64 for API testing
"""

import base64
import sys
import os

def audio_to_base64(file_path):
    """Convert audio file to base64 string"""
    try:
        with open(file_path, 'rb') as audio_file:
            audio_data = audio_file.read()
            base64_audio = base64.b64encode(audio_data).decode('utf-8')
            return base64_audio
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found")
        return None
    except Exception as e:
        print(f"Error: {str(e)}")
        return None

def save_base64_to_file(base64_string, output_file):
    """Save base64 string to a file"""
    try:
        with open(output_file, 'w') as f:
            f.write(base64_string)
        print(f"✅ Base64 saved to: {output_file}")
    except Exception as e:
        print(f"Error saving file: {str(e)}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 convert_audio.py <audio_file.mp3> [output_file.txt]")
        print("\nExample:")
        print("  python3 convert_audio.py sample.mp3")
        print("  python3 convert_audio.py sample.mp3 output.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(input_file):
        print(f"❌ Error: File '{input_file}' does not exist")
        sys.exit(1)
    
    print(f"📁 Reading file: {input_file}")
    file_size = os.path.getsize(input_file)
    print(f"📊 File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")
    
    print("🔄 Converting to base64...")
    base64_audio = audio_to_base64(input_file)
    
    if base64_audio:
        base64_length = len(base64_audio)
        print(f"✅ Conversion complete!")
        print(f"📊 Base64 length: {base64_length:,} characters")
        
        if output_file:
            save_base64_to_file(base64_audio, output_file)
        else:
            # Print first 100 and last 100 characters
            print("\n--- Base64 Preview (first 100 chars) ---")
            print(base64_audio[:100])
            print("...")
            print(base64_audio[-100:])
            print("--- End Preview ---\n")
            
            print("💡 Tip: To save to file, run:")
            print(f"   python3 convert_audio.py {input_file} output.txt")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
