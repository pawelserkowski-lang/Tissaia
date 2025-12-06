import unittest
import src.config
import src.ai_core
import src.graphics
import os

class TestTissaia(unittest.TestCase):
    def test_config_loading(self):
        self.assertIsNotNone(src.config.MODEL_DETECTION)

    def test_ai_core_functions(self):
        self.assertTrue(hasattr(src.ai_core, 'detect_rotation_strict'))
        self.assertTrue(hasattr(src.ai_core, 'clean_json_response'))

    def test_graphics_functions(self):
        self.assertTrue(hasattr(src.graphics, 'warp_perspective'))

if __name__ == "__main__":
    unittest.main()