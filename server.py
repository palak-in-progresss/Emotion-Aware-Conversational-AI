import os
import sys
import json
import tempfile
from http.server import HTTPServer, SimpleHTTPRequestHandler

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from main import process_multimodal_input
from utils.logger import logger

class VioraRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        ui_directory = os.path.join(PROJECT_ROOT, "ui")
        super().__init__(*args, directory=ui_directory, **kwargs)

    def do_POST(self):
        if self.path in ['/api/predict', '/api/multimodal']:
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
                except Exception as e:
                    logger.error(f"Error parsing JSON payload: {e}")

            elif post_data:
                # Raw audio binary upload
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
        else:
            self.send_error(404, "Endpoint Not Found")

def run_server(port=config.SERVER_PORT):
    server_address = ('', port)
    httpd = HTTPServer(server_address, VioraRequestHandler)
    print("==================================================")
    print(f"[VIORA] Web HUD Server Running at: http://localhost:{port}")
    print("==================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping VIORA server...")
        httpd.server_close()

if __name__ == '__main__':
    run_server()
