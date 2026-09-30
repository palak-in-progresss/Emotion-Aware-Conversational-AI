import os
import sys
import json
import tempfile
from http.server import BaseHTTPRequestHandler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from main import process_multimodal_input

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        post_data = self.rfile.read(content_length)

        audio_path = None
        text_input = None
        voice_weight = config.DEFAULT_VOICE_WEIGHT
        text_weight = config.DEFAULT_TEXT_WEIGHT

        content_type = self.headers.get('Content-Type', '')

        if 'application/json' in content_type:
            try:
                payload = json.loads(post_data.decode('utf-8'))
                text_input = payload.get('text', '')
                voice_weight = payload.get('voice_weight', config.DEFAULT_VOICE_WEIGHT)
                text_weight = payload.get('text_weight', config.DEFAULT_TEXT_WEIGHT)
            except Exception:
                pass
        elif post_data:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(post_data)
                audio_path = tmp.name

        try:
            result = process_multimodal_input(
                audio_input=audio_path,
                text_input=text_input,
                voice_weight=voice_weight,
                text_weight=text_weight
            )
        finally:
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode('utf-8'))

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"VIORA Vercel API Endpoint Active")
