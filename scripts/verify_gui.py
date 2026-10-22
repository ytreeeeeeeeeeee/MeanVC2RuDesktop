"""Real Windows GUI/device integration check. Uses cached models, never mocks inference.

Run from the project root: python scripts/verify_gui.py --reference ref.wav
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtWidgets import QApplication

from app.audio import prepare_voice, record_voice, listen_voice
from app.gui import MainWindow
from app.settings import configure_logging, data_dir
import threading


def wait(application, condition, timeout=120):
    deadline = time.monotonic() + timeout
    while not condition():
        application.processEvents()
        if time.monotonic() > deadline:
            raise TimeoutError("GUI operation timed out")
        time.sleep(.02)
    application.processEvents()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--physical-input", default="Microphone", help="Substring of the physical WASAPI input name")
    parser.add_argument("--physical-output", default="", help="Substring of the physical WASAPI output name")
    args = parser.parse_args()
    configure_logging()
    settings_path = data_dir() / "settings.json"
    original_settings = settings_path.read_bytes() if settings_path.exists() else None
    application = QApplication([])
    window = MainWindow()
    window.show()
    report = {"scenarios": []}
    report_path = Path("reports/gui-integration.json")
    imported = None
    try:
        imported = prepare_voice(args.reference)
        window.refresh_voices(imported)
        window.refresh_audio(True)
        # Resolve names after enumeration: Windows indices can change on hotplug.
        mic = next(d for d in window.device_list if d["host"] == "Windows WASAPI" and d["inputs"] and args.physical_input in d["name"])
        headphones = next(d for d in window.device_list if d["host"] == "Windows WASAPI" and d["outputs"]
                          and "CABLE" not in d["name"] and args.physical_output in d["name"])
        cable_in = next(d for d in window.device_list if d["host"] == "Windows WASAPI" and d["inputs"] and "CABLE Output" in d["name"])
        cable_out = next(d for d in window.device_list if d["host"] == "Windows WASAPI" and d["outputs"] and "CABLE Input (" in d["name"])
        window.volume.setValue(0)
        window.buffer.setValue(160)
        pcm, pcm_rate = sf.read(imported, dtype="float32")
        pcm = __import__("soxr").resample(pcm, pcm_rate, 48000)
        feed_stop = threading.Event()
        def feed():
            with sd.OutputStream(device=cable_out["index"], samplerate=48000, channels=2, dtype="float32") as stream:
                while not feed_stop.is_set():
                    for start in range(0, len(pcm), 2400):
                        if feed_stop.is_set():
                            break
                        stream.write(np.repeat(pcm[start:start+2400, None], 2, axis=1))
        feed_thread = None
        for mode, input_device, output_device in (("cpu", mic, cable_out), ("cuda", cable_in, headphones)):
            print("Starting", mode, input_device["key"], output_device["key"], flush=True)
            window.restore(window.compute, mode)
            window.restore(window.input, input_device["key"])
            window.restore(window.output, output_device["key"])
            if mode == "cuda":
                feed_thread = threading.Thread(target=feed)
                feed_thread.start()
            window.start_conversion()
            wait(application, lambda: window.audio is not None or window.task is None)
            assert window.audio is not None, window.errors.text()
            wait(application, lambda: "работает" in window.status.text() or window.task is None)
            assert window.task is not None, window.errors.text()
            wait(application, lambda: window.audio.stats.blocks >= 65 or window.task is None, timeout=30)
            assert window.audio is not None and window.audio.stats.blocks >= 65, window.errors.text()
            report["scenarios"].append(dict(mode=mode, input=input_device["key"], output=output_device["key"], stats=asdict(window.audio.stats)))
            if mode == "cuda":
                assert window.audio.stats.peak_level > 1e-4
            window.stop()
            wait(application, lambda: window.task is None)
            print("Stopped", mode, flush=True)
            assert not window.errors.text(), window.errors.text()
        # Exercise the GUI recording worker using speech arriving through a real Windows endpoint.
        window.restore(window.input, cable_in["key"])
        window.record_seconds.setValue(3)
        before = set(Path(p) for p in (window.voice.itemData(i) for i in range(window.voice.count())))
        window.record()
        wait(application, lambda: window.task is None)
        assert not window.errors.text(), window.errors.text()
        new_recording = Path(window.voice.currentData())
        assert new_recording not in before and sf.info(new_recording).duration >= 2.9
        report["gui_recording_virtual_input"] = "passed"
        new_recording.unlink()
        window.refresh_voices(imported)
        feed_stop.set()
        feed_thread.join(timeout=5)
        assert not feed_thread.is_alive()
        # Recording through the selected physical microphone; silence is a valid validation rejection.
        try:
            print("Recording", flush=True)
            recording = record_voice(mic["index"], 2, threading.Event())
            report["recording"] = dict(status="passed", duration=sf.info(recording).duration)
            recording.unlink()
        except ValueError as error:
            report["recording"] = dict(status="rejected_sample", reason=str(error))
        # Feed a prepared voice into the virtual output and verify the paired Windows capture.
        samples, rate = sf.read(imported, dtype="float32")
        samples = __import__("soxr").resample(samples, rate, 48000)
        captured = []
        def callback(data, frames, times, status):
            captured.append(data.copy())
        with sd.InputStream(device=cable_in["index"], samplerate=48000, channels=1,
                            dtype="float32", callback=callback):
            with sd.OutputStream(device=cable_out["index"], samplerate=48000, channels=2, dtype="float32") as stream:
                stream.write(np.repeat(samples[:, None], 2, axis=1))
            sd.sleep(300)
        assert captured and np.max(np.abs(np.concatenate(captured))) > 1e-5
        report["virtual_cable_pcm"] = "passed"
        print("Virtual cable passed", flush=True)
        # Listening uses the selected virtual output and releases it afterward.
        listen_voice(imported, cable_out["index"], threading.Event())
        report["listen_voice"] = "passed"
        window.persist()
        window.grab().save("reports/gui-running.png")
        selection = window.voice.currentData()
        window.close()
        restored = MainWindow()
        assert restored.voice.currentData() == selection
        assert restored.compute.currentData() == "cuda"
        assert restored.input.currentData() == cable_in["key"]
        restored.close()
        report["settings_restart"] = "passed"
    finally:
        if "feed_stop" in locals():
            feed_stop.set()
        if "feed_thread" in locals() and feed_thread:
            feed_thread.join(timeout=5)
        if window.task:
            window.stop()
            wait(application, lambda: window.task is None)
        window.close()
        if imported:
            imported.unlink(missing_ok=True)
        if original_settings is None:
            settings_path.unlink(missing_ok=True)
        else:
            settings_path.write_bytes(original_settings)
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
