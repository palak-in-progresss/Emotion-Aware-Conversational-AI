import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from main import process_multimodal_input
from text.response_generator import generate_empathetic_response

class TestMultimodalPipeline(unittest.TestCase):
    def test_text_only_pipeline(self):
        res = process_multimodal_input(text_input="I am so happy that I passed my exam!")
        self.assertEqual(res["final_emotion"], "happy")
        self.assertEqual(res["fusion"]["mode"], "text_only")
        self.assertTrue(len(res["response"]) > 0)

    def test_empathetic_response_phrasing(self):
        resp = generate_empathetic_response("I miss you so much", detected_emotion="sad")
        self.assertTrue(len(resp) > 0)
        self.assertNotIn("depressed", resp.lower())
        self.assertNotIn("medical diagnosis", resp.lower())

if __name__ == "__main__":
    unittest.main()
