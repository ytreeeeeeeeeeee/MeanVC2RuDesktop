import os
import tempfile
import time
import unittest
from unittest.mock import patch

from PySide6.QtWidgets import QApplication
import sounddevice as sd

from app.gui import MainWindow


class GuiTests(unittest.TestCase):
    def test_unsupported_audio_rate_does_not_load_model(self):
        application = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {"LOCALAPPDATA": folder}), \
                patch("app.gui.devices", return_value=[]), patch("app.gui.audio_devices", return_value=[]):
            window = MainWindow()
            window.restore(window.compute, "cpu")
            window.restore(window.rate, 96000)
            try:
                with patch.object(window, "selected_voice", return_value="reference.wav"), \
                        patch("app.audio.sd.query_devices", return_value=dict(name="Microphone", default_samplerate=48000)), \
                        patch("app.audio.sd.check_input_settings", side_effect=sd.PortAudioError("Invalid sample rate", -9997)), \
                        patch("app.gui.InferenceEngine") as inference, self.assertLogs(level="ERROR"):
                    window.start_conversion()
                    deadline = time.monotonic() + 5
                    while window.task is not None and time.monotonic() < deadline:
                        application.processEvents()
                        time.sleep(.01)
                    self.assertIsNone(window.task)
                    inference.assert_not_called()
                    self.assertIn("96000", window.errors.text())
                    self.assertIn("Автоматически", window.errors.text())
                    self.assertTrue(window.start_button.isEnabled())
            finally:
                window.close()
                application.processEvents()


if __name__ == "__main__":
    unittest.main()
