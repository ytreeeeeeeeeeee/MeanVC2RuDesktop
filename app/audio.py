"""PortAudio callbacks only move buffers; the calling worker performs inference."""
from dataclasses import dataclass
import logging
from pathlib import Path
from queue import Empty, Full, Queue
import re
import threading
import time
import uuid

import numpy as np
import sounddevice as sd
import soundfile as sf
import soxr

from app.settings import data_dir


def audio_devices(refresh=False):
    if refresh:
        # PortAudio caches enumeration until reinitialization; call only while idle.
        sd._terminate()
        sd._initialize()
    hosts = sd.query_hostapis()
    return [dict(index=i, name=d["name"], host=hosts[d["hostapi"]]["name"],
                 key=hosts[d["hostapi"]]["name"] + " | " + d["name"],
                 inputs=d["max_input_channels"], outputs=d["max_output_channels"])
            for i, d in enumerate(sd.query_devices())]


def stream_format(device, kind, rate=0):
    info = sd.query_devices(device, kind)
    channels = 1 if kind == "input" else min(2, int(info["max_output_channels"]))
    check = sd.check_input_settings if kind == "input" else sd.check_output_settings
    rates = [rate] if rate else [int(info["default_samplerate"]), 48000, 44100, 16000, 96000]
    for candidate in dict.fromkeys(rates):
        try:
            check(device=device, channels=channels, dtype="float32", samplerate=candidate)
            logging.info("Audio %s: %s, %s Hz, %s channels", kind, info["name"], candidate, channels)
            return candidate, channels
        except sd.PortAudioError as error:
            if len(error.args) < 2 or error.args[1] != -9997:
                raise
            if rate:
                direction = "входа" if kind == "input" else "выхода"
                raise ValueError(
                    f"Устройство {direction} «{info['name']}» не поддерживает частоту {rate} Гц. "
                    "Выберите «Автоматически (частота устройства)» в разделе «Аудио» "
                    "или другую частоту. Можно также выбрать это устройство через другой аудиодрайвер."
                ) from error
    raise ValueError(f"Не удалось подобрать частоту для «{info['name']}». "
                     "Проверьте формат устройства в настройках звука Windows или выберите другой аудиодрайвер.")


@dataclass
class AudioStats:
    level: float = 0
    peak_level: float = 0
    processing_ms: float = 0
    latency_ms: float = 0
    input_overflows: int = 0
    output_underflows: int = 0
    blocks: int = 0
    input_latency_ms: float = 0
    output_latency_ms: float = 0


class AudioEngine:
    def __init__(self, inference, input_device, output_device, rate=0, volume=1.0, buffer_ms=160):
        if not 20 <= buffer_ms <= 500:
            raise ValueError("Буфер воспроизведения: 20–500 мс")
        self.inference = inference
        self.input_device, self.output_device = input_device, output_device
        self.input_rate, self.input_channels = stream_format(input_device, "input", rate)
        self.output_rate, self.output_channels = stream_format(output_device, "output", rate)
        self.volume = volume
        self.buffer_ms = buffer_ms
        self.stats = AudioStats()
        self.error = None
        self.inputs = Queue(maxsize=25)
        self.outputs = Queue(maxsize=5)
        self.current = np.empty(0, dtype=np.float32)
        self.position = 0
        self.playing = False
        self.queued_samples = 0

    def capture(self, data, frames, times, status):
        if status:
            self.stats.input_overflows += 1
        self.stats.level = float(np.sqrt(np.mean(data[:, 0] ** 2)))
        self.stats.peak_level = max(self.stats.peak_level, self.stats.level)
        try:
            self.inputs.put_nowait((data[:, 0].copy(), time.perf_counter()))
        except Full:
            self.error = "Обработка не успевает за микрофоном. Уменьшите число шагов или выберите CUDA."

    def playback(self, data, frames, times, status):
        data.fill(0)
        if status:
            self.stats.output_underflows += 1
        if not self.playing:
            if self.queued_samples < self.output_rate * self.buffer_ms / 1000:
                return
            self.playing = True
        written = 0
        while written < frames:
            if self.position == len(self.current):
                try:
                    self.current = self.outputs.get_nowait()
                    self.position = 0
                except Empty:
                    self.stats.output_underflows += 1
                    self.playing = False
                    return
            n = min(frames - written, len(self.current) - self.position)
            data[written:written+n] = (self.current[self.position:self.position+n] * self.volume)[:, None]
            np.clip(data[written:written+n], -1, 1, out=data[written:written+n])
            self.position += n
            written += n
            self.queued_samples -= n

    def run(self, stop, ready=lambda: None):
        input_resampler = soxr.ResampleStream(self.input_rate, 16000, 1, dtype="float32", quality="LQ")
        output_resampler = soxr.ResampleStream(16000, self.output_rate, 1, dtype="float32", quality="LQ")
        pending = np.empty(0, dtype=np.float32)
        source_time = None
        self.inference.reset()
        try:
            with sd.OutputStream(device=self.output_device, samplerate=self.output_rate,
                    channels=self.output_channels, dtype="float32", blocksize=0,
                    latency="low", callback=self.playback) as output:
                with sd.InputStream(device=self.input_device, samplerate=self.input_rate,
                        channels=self.input_channels, dtype="float32", blocksize=round(self.input_rate * .02),
                        latency="low", callback=self.capture) as capture:
                    self.stats.input_latency_ms = capture.latency * 1000
                    self.stats.output_latency_ms = output.latency * 1000
                    ready()
                    while not stop.is_set():
                        if self.error:
                            raise RuntimeError(self.error)
                        if not capture.active or not output.active:
                            raise RuntimeError("Аудиоустройство отключилось. Обновите список устройств.")
                        try:
                            samples, captured = self.inputs.get(timeout=.1)
                        except Empty:
                            continue
                        if source_time is None:
                            source_time = captured
                        pending = np.concatenate((pending, input_resampler.resample_chunk(samples)))
                        while len(pending) >= self.inference.chunk_samples and not stop.is_set():
                            block, pending = pending[:self.inference.chunk_samples], pending[self.inference.chunk_samples:]
                            start = time.perf_counter()
                            converted = self.inference.process(block)
                            self.stats.processing_ms = (time.perf_counter() - start) * 1000
                            self.stats.blocks += 1
                            if len(converted):
                                converted = output_resampler.resample_chunk(converted)
                                if len(converted):
                                    self.queued_samples += len(converted)
                                    try:
                                        self.outputs.put_nowait(converted)
                                    except Full:
                                        self.queued_samples -= len(converted)
                                        raise RuntimeError("Выходное устройство не успевает воспроизводить звук")
                                # Estimate includes oldest captured input, output queue and driver latency.
                                self.stats.latency_ms = ((time.perf_counter() - source_time) * 1000
                                    + self.queued_samples / self.output_rate * 1000
                                    + self.stats.input_latency_ms + self.stats.output_latency_ms)
                            source_time += self.inference.chunk_samples / 16000
        finally:
            logging.info("Audio stopped: %s", self.stats)


def voices_dir():
    path = data_dir() / "voices"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_voice(source, label=None):
    info = sf.info(source)
    if info.format not in ("WAV", "WAVEX", "RF64") or not 1 <= info.duration <= 30 or info.channels > 8:
        raise ValueError("Нужен WAV длительностью от 1 до 30 секунд (до 8 каналов)")
    if not 8000 <= info.samplerate <= 192000:
        raise ValueError("Частота WAV должна быть от 8 до 192 кГц")
    samples, rate = sf.read(source, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if not np.isfinite(samples).all() or float(np.sqrt(np.mean(samples ** 2))) < 1e-5:
        raise ValueError("Образец содержит тишину или некорректные значения")
    samples = soxr.resample(samples, rate, 16000)
    samples = np.clip(samples, -1, 1)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", label or Path(source).stem).strip(" .")[:64] or "Голос"
    path = voices_dir() / f"{name} — {uuid.uuid4().hex[:8]}.wav"
    temporary = path.with_suffix(".tmp")
    try:
        sf.write(temporary, samples, 16000, format="WAV", subtype="PCM_16")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def record_voice(device, seconds, stop, level=lambda value: None):
    rate, channels = stream_format(device, "input")
    blocks = Queue(maxsize=100)
    overflow = threading.Event()

    def callback(data, frames, times, status):
        if status:
            overflow.set()
        try:
            blocks.put_nowait(data.copy())
        except Full:
            overflow.set()

    samples = []
    count = 0
    started = time.monotonic()
    with sd.InputStream(device=device, samplerate=rate, channels=channels, dtype="float32", callback=callback) as stream:
        while count < seconds * rate and not stop.is_set():
            if not stream.active or time.monotonic() - started > seconds + 5:
                raise RuntimeError("Микрофон перестал передавать звук. Проверьте подключение устройства.")
            try:
                block = blocks.get(timeout=.1)
            except Empty:
                continue
            samples.append(block)
            count += len(block)
            level(float(np.sqrt(np.mean(block ** 2))))
    if overflow.is_set():
        raise RuntimeError("Запись прерывалась. Повторите запись с другим устройством.")
    if count < rate:
        raise ValueError("Запись слишком короткая: нужна хотя бы 1 секунда")
    temporary = voices_dir() / (uuid.uuid4().hex + ".recording")
    try:
        sf.write(temporary, np.concatenate(samples)[:seconds * rate], rate, format="WAV")
        return prepare_voice(temporary, "Запись")
    finally:
        temporary.unlink(missing_ok=True)


def listen_voice(path, device, stop):
    rate, channels = stream_format(device, "output")
    samples, source_rate = sf.read(path, dtype="float32")
    samples = soxr.resample(samples, source_rate, rate)
    with sd.OutputStream(device=device, samplerate=rate, channels=channels, dtype="float32") as stream:
        for start in range(0, len(samples), round(rate * .02)):
            if stop.is_set():
                break
            stream.write(np.repeat(samples[start:start + round(rate * .02), None], channels, axis=1))
