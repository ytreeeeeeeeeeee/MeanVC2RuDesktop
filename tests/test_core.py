import json
from pathlib import Path
from queue import Queue
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import sounddevice as sd
import soundfile as sf
from safetensors.torch import save_file
import torch

from app.audio import AudioEngine, AudioStats, prepare_voice, stream_format
from app.inference import choose_device, load_weights
from app.models import ModelManager
from app import settings
from app.upstream import DiT, UPSTREAM_ROOT


class CoreTests(unittest.TestCase):
    def test_automatic_sample_rate_falls_back(self):
        info = dict(name="Microphone", default_samplerate=96000)
        with patch("app.audio.sd.query_devices", return_value=info), patch("app.audio.sd.check_input_settings") as check:
            check.side_effect = [sd.PortAudioError("Invalid sample rate", -9997), None]
            self.assertEqual(stream_format(3, "input"), (48000, 1))
            self.assertEqual([c.kwargs["samplerate"] for c in check.call_args_list], [96000, 48000])

    def test_explicit_sample_rate_error_is_actionable(self):
        info = dict(name="Microphone", default_samplerate=48000)
        with patch("app.audio.sd.query_devices", return_value=info), patch("app.audio.sd.check_input_settings") as check:
            check.side_effect = sd.PortAudioError("Invalid sample rate", -9997)
            with self.assertRaisesRegex(ValueError, "96000.*Автоматически"):
                stream_format(3, "input", 96000)
            self.assertEqual(check.call_count, 1)

    def test_automatic_output_rate_and_unrelated_errors(self):
        info = dict(name="Output", default_samplerate=44100, max_output_channels=2)
        with patch("app.audio.sd.query_devices", return_value=info), patch("app.audio.sd.check_output_settings") as check:
            self.assertEqual(stream_format(4, "output"), (44100, 2))
            check.side_effect = sd.PortAudioError("Device unavailable", -9985)
            with self.assertRaises(sd.PortAudioError):
                stream_format(4, "output")
            self.assertEqual(check.call_count, 2)

    def test_automatic_sample_rate_exhausted(self):
        info = dict(name="Microphone", default_samplerate=48000)
        with patch("app.audio.sd.query_devices", return_value=info), patch("app.audio.sd.check_input_settings") as check:
            check.side_effect = sd.PortAudioError("Invalid sample rate", -9997)
            with self.assertRaisesRegex(ValueError, "Не удалось подобрать"):
                stream_format(3, "input")
            self.assertEqual(check.call_count, 4)

    def test_device_selection(self):
        with patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(choose_device("auto"), "cpu")
            self.assertEqual(choose_device("cpu"), "cpu")
            with self.assertRaisesRegex(ValueError, "CUDA"):
                choose_device("cuda")
        with patch("torch.cuda.is_available", return_value=True), patch("torch.cuda.device_count", return_value=2):
            self.assertEqual(choose_device("cuda", 1), "cuda:1")
            with self.assertRaises(ValueError):
                choose_device("cuda", 2)

    def test_settings_and_corruption(self):
        with tempfile.TemporaryDirectory() as folder, patch("app.settings.data_dir", return_value=Path(folder)):
            settings.save_settings({"voice": "Русский голос", "volume": 75})
            self.assertEqual(settings.load_settings()["volume"], 75)
            (Path(folder) / "settings.json").write_text('{"volume":null,"voice":42}', encoding="utf-8")
            self.assertEqual(settings.load_settings(), {})
            (Path(folder) / "settings.json").write_text("{", encoding="utf-8")
            with self.assertLogs(level="ERROR"):
                self.assertEqual(settings.load_settings(), {})

    def test_reference_validation_and_resampling(self):
        with tempfile.TemporaryDirectory() as folder, patch("app.audio.voices_dir", return_value=Path(folder)):
            folder = Path(folder)
            rate = 44100
            wave = np.sin(np.arange(rate * 2) * 2 * np.pi * 220 / rate).astype("float32") * .2
            sf.write(folder / "input.wav", np.column_stack((wave, wave)), rate)
            path = prepare_voice(folder / "input.wav", "../Голос")
            self.assertEqual(path.parent, folder)
            self.assertEqual(sf.info(path).samplerate, 16000)
            self.assertEqual(sf.info(path).channels, 1)
            self.assertAlmostEqual(sf.info(path).duration, 2, places=2)
            sf.write(folder / "silence.wav", np.zeros(16000), 16000)
            with self.assertRaisesRegex(ValueError, "тишину"):
                prepare_voice(folder / "silence.wav")
            sf.write(folder / "short.wav", wave[:100], rate)
            with self.assertRaises(ValueError):
                prepare_voice(folder / "short.wav")

    def test_playback_partial_blocks_underflow_and_clipping(self):
        audio = AudioEngine.__new__(AudioEngine)
        audio.outputs = Queue(maxsize=5)
        audio.outputs.put(np.array([2, .5, -.5, -2], dtype="float32"))
        audio.current = np.empty(0, dtype="float32")
        audio.position = 0
        audio.volume = 1
        audio.playing = True
        audio.queued_samples = 4
        audio.stats = AudioStats()
        data = np.empty((3, 2), dtype="float32")
        audio.playback(data, 3, None, None)
        np.testing.assert_array_equal(data[:, 0], [1, .5, -.5])
        np.testing.assert_array_equal(data[:, 1], data[:, 0])
        audio.playback(data, 3, None, None)
        np.testing.assert_array_equal(data[:, 0], [-1, 0, 0])
        self.assertEqual(audio.queued_samples, 0)
        self.assertEqual(audio.stats.output_underflows, 1)

    def test_capture_is_bounded(self):
        audio = AudioEngine.__new__(AudioEngine)
        audio.inputs = Queue(maxsize=1)
        audio.stats = AudioStats()
        audio.error = None
        for _ in range(2):
            audio.capture(np.ones((320, 1), dtype="float32"), 320, None, None)
        self.assertIsNotNone(audio.error)
        self.assertEqual(audio.inputs.qsize(), 1)

    def test_hf_download_integrity_cache_and_update(self):
        import hashlib
        from types import SimpleNamespace
        data = b"test weights"
        digest = hashlib.sha256(data).hexdigest()
        metadata = SimpleNamespace(etag=digest, size=len(data), location="https://example.test/model", commit_hash="revision")
        with tempfile.TemporaryDirectory() as folder:
            manager = ModelManager(folder)
            with self.assertRaises(FileNotFoundError):
                manager.fetch("test/repo", "model.safetensors", offline=True)
            with patch("app.models.get_hf_file_metadata", return_value=metadata), patch("app.models.requests.get") as get:
                response = get.return_value.__enter__.return_value
                response.iter_content.return_value = [data]
                path = manager.fetch("test/repo", "model.safetensors")
                self.assertEqual(path.read_bytes(), data)
                self.assertEqual(manager.fetch("test/repo", "model.safetensors", offline=True), path)
                manager.fetch("test/repo", "model.safetensors")
                self.assertEqual(get.call_count, 1)
                metadata.etag = "f" * 64
                with self.assertRaisesRegex(ValueError, "SHA256"):
                    manager.fetch("test/repo", "model.safetensors")
                self.assertTrue(path.exists())

    def test_strict_model_compatibility(self):
        config = json.loads((UPSTREAM_ROOT / "src/config/config_120ms_40ms.json").read_text())["model"]
        torch.set_num_threads(1)
        model = DiT(**config)
        weights = {k: v for k, v in model.state_dict().items() if not k.startswith("cache_embed.")}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.safetensors"
            save_file(weights, str(path), metadata={"model": "MeanVC2 120ms+40ms"})
            loaded = load_weights(path, "120ms", "cpu")
            with torch.inference_mode():
                result, cache = loaded(torch.zeros(1, 16, 80), torch.ones(1), torch.zeros(1),
                    cond=torch.zeros(1, 16, 256), spks=torch.ones(1, 256), cache=None, is_inference=True)
            self.assertEqual(tuple(result.shape), (1, 16, 80))
            self.assertTrue(torch.isfinite(result).all())
            with self.assertRaisesRegex(ValueError, "конфигурац"):
                load_weights(path, "40ms", "cpu")
            del weights["proj_out.weight"]
            save_file(weights, str(path))
            with self.assertRaisesRegex(ValueError, "Несовместимые веса"):
                load_weights(path, "120ms", "cpu")


if __name__ == "__main__":
    unittest.main()
