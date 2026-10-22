"""Reproducible real-model and device checks, usable from Python and the EXE."""
from dataclasses import asdict
import json
from pathlib import Path
import platform
import sys
import threading
import time

import numpy as np
import soundfile as sf
import soxr
import torch

from app.audio import AudioEngine, audio_devices
from app.inference import InferenceEngine, devices


def write_json(path, value):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def diagnostics(path):
    write_json(path, dict(platform=platform.platform(), python=sys.version,
        frozen=bool(getattr(sys, "frozen", False)), executable=sys.executable,
        python_paths=sys.path, processor=platform.processor(),
        torch=torch.__version__, cuda_runtime=torch.version.cuda, gpus=devices(),
        audio_devices=audio_devices()))


def benchmark(args):
    if not args.source:
        raise ValueError("Укажите --source с речевым WAV")
    if not 1 <= args.seconds <= 3600:
        raise ValueError("Длительность проверки: 1–3600 секунд")
    start = time.perf_counter()
    engine = InferenceEngine(args.model, args.benchmark.resolve(), args.device, args.gpu, args.profile)
    initialization = time.perf_counter() - start
    samples, rate = sf.read(args.source, dtype="float32", always_2d=True)
    samples = soxr.resample(samples.mean(axis=1), rate, 16000)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Некорректный исходный WAV")
    samples = np.resize(samples, round(args.seconds * 16000))
    samples = np.pad(samples, (0, (-len(samples)) % engine.chunk_samples))
    results = []
    # Two passes verify cache reset and repeated use of the loaded models.
    for run in range(2):
        engine.reset()
        durations = []
        outputs = []
        for block in samples.reshape(-1, engine.chunk_samples):
            start = time.perf_counter()
            output = engine.process(block)
            durations.append((time.perf_counter() - start) * 1000)
            if len(output):
                outputs.append(output)
        if not outputs:
            raise AssertionError("Streaming inference produced no audio")
        output = np.concatenate(outputs)
        assert np.isfinite(output).all() and np.max(np.abs(output)) > 1e-6
        sf.write(args.report.with_suffix(f".run{run + 1}.wav"), output, 16000)
        results.append(dict(blocks=len(durations), input_seconds=len(samples) / 16000,
            output_seconds=len(output) / 16000, mean_ms=float(np.mean(durations)),
            p95_ms=float(np.percentile(durations, 95)), max_ms=float(np.max(durations)),
            rtf=sum(durations) / 1000 / (len(samples) / 16000),
            peak=float(np.max(np.abs(output)))))
    report = dict(platform=platform.platform(), torch=torch.__version__, device=engine.device,
        gpus=devices(), profile=args.profile, model=args.model, initialization_s=initialization,
        streaming_passes=results, audio_latency="Not measured: use --audio-test for an estimate; physical loopback is required for end-to-end measurement")
    if args.audio_test:
        report["audio_runs"] = []
        for run in range(2):
            stop = threading.Event()
            audio = AudioEngine(engine, args.input_device, args.output_device, volume=0)
            timer = threading.Timer(args.seconds, stop.set)
            try:
                audio.run(stop, timer.start)
            finally:
                timer.cancel()
            assert audio.stats.blocks > 0
            report["audio_runs"].append(asdict(audio.stats))
    write_json(args.report, report)
