import unittest
import src.config
import src.ai_core
import src.graphics
import os

class TestTissaia(unittest.TestCase):
    def test_config_loading(self):
        # We assume .env is set up or environment variables are mocked
        # Check for module-level constants
        self.assertIsNotNone(src.config.API_KEY)
        self.assertIsNotNone(src.config.MODEL_DETECTION)
        self.assertIsNotNone(src.config.INPUT_ZIP)

    def test_ai_core_functions(self):
        # Check if functions exist
        self.assertTrue(hasattr(src.ai_core, 'get_url'))
        self.assertTrue(hasattr(src.ai_core, 'detect_rotation_strict'))
        self.assertTrue(hasattr(src.ai_core, 'detect_corners'))
        self.assertTrue(hasattr(src.ai_core, 'restore_final'))

    def test_graphics_functions(self):
        # Check if functions exist
        self.assertTrue(hasattr(src.graphics, 'aggressive_trim_borders'))
        self.assertTrue(hasattr(src.graphics, 'apply_super_sharpen'))
        self.assertTrue(hasattr(src.graphics, 'warp_perspective'))

if __name__ == "__main__":
    unittest.main()
