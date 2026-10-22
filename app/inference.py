"""Small adapter around the official MeanVC2 streaming runtime."""
import json
import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from app.models import ModelManager
from app.upstream import VCRunner, MODEL_PATHS, DiT, ECAPA_TDNN_SMALL, extract_embedding, UPSTREAM_ROOT


def devices():
    return [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]


def choose_device(mode="auto", gpu=0):
    if mode not in ("auto", "cpu", "cuda"):
        raise ValueError("Неизвестное устройство вычисления")
    if mode == "cpu" or (mode == "auto" and not torch.cuda.is_available()):
        return "cpu"
    if not torch.cuda.is_available() or gpu not in range(torch.cuda.device_count()):
        raise ValueError("CUDA недоступна. Выберите CPU или установите совместимый драйвер NVIDIA и CUDA-сборку приложения.")
    return f"cuda:{gpu}"


def load_weights(path, profile, device):
    if profile not in MODEL_PATHS:
        raise ValueError("Выберите конфигурацию 120ms или 40ms")
    config = json.loads((UPSTREAM_ROOT / f"src/config/config_{profile}_40ms.json").read_text())
    model = DiT(**config["model"])
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
    declared = metadata.get("model", "")
    if ("120ms" in declared and profile != "120ms") or ("120ms" not in declared and "40ms" in declared and profile != "40ms"):
        raise ValueError(f"Модель {declared} несовместима с выбранной конфигурацией")
    state = load_file(str(path), device="cpu")
    expected = model.state_dict()
    # Upstream's legacy cache_embed is inactive: streaming always uses cache=None.
    missing = expected.keys() - state.keys()
    extra = state.keys() - expected.keys()
    if missing - {"cache_embed.weight", "cache_embed.bias"} or extra:
        raise ValueError(f"Несовместимые веса MeanVC2: отсутствуют {sorted(missing)}, лишние {sorted(extra)}")
    for key, value in state.items():
        if value.shape != expected[key].shape or not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"Несовместимый или повреждённый параметр: {key}")
    model.load_state_dict(state | {k: expected[k] for k in missing}, strict=True)
    return model.float().eval().to(device)


class _StreamingRunner(VCRunner):
    """App-owned initialization; all default streaming methods are upstream's."""

    def __init__(self, model, reference, device, profile, steps, manager):
        # Stock runtime recognizes 'cuda'; the owner selects its GPU context.
        self.device = "cuda" if device.startswith("cuda") else "cpu"
        self.steps = steps
        path = manager.model(model)
        self.vc = load_weights(path, profile, self.device)
        paths = manager.auxiliaries(profile)
        self.asr = torch.jit.load(str(paths["asr"]), map_location="cpu").eval()
        self.vocoder = torch.jit.load(str(paths["vocoder"]), map_location=self.device).eval()
        speaker = ECAPA_TDNN_SMALL(feat_dim=1024, emb_dim=256, feat_type="wavlm_large",
                                   config_path=str(Path(__file__).parent / "assets/wavlm_large_cfg.pt"))
        state = torch.load(paths["speaker"], map_location="cpu", weights_only=True)
        state = state.get("model", state)
        result = speaker.load_state_dict(state, strict=False)
        # Fine-tuned upstream checkpoints may include a training classifier only.
        if result.missing_keys or any(not k.startswith("loss_calculator.") for k in result.unexpected_keys):
            raise ValueError(f"Несовместимый WavLM/ECAPA checkpoint: {result}")
        speaker.eval().to(self.device)
        with torch.inference_mode():
            self.vc_spk_emb = extract_embedding(speaker, str(reference), device=self.device)
            self.vc_gtm_kv = self.vc.gtm(self.vc_spk_emb)
        del speaker, state
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        parameters = MODEL_PATHS[profile]
        for attr, key in (("_bn_window", "bn_window"), ("_bn_stride", "bn_stride"),
                          ("_required_cache_size", "required_cache_size"),
                          ("_asr_offset_init", "asr_offset_init"), ("_asr_offset_step", "asr_offset_step")):
            setattr(self, attr, parameters[key])
        self.chunk_size = 12 if profile == "120ms" else 4
        self.block_size = 4
        self.CHUNK = 2560
        self.vocoder_overlap = 2
        self.upsample_factor = 160
        self.vocoder_wav_overlap = 160
        self.vocoder_wav_pad = 0
        self.down_linspace = torch.linspace(1, 0, 160)
        self.up_linspace = torch.linspace(0, 1, 160)
        self._init_cache()
        logging.info("Inference ready: %s / %s / %s", path, profile, device)

    def _vc_step(self, cond):
        if self.steps == 2:
            return super()._vc_step(cond)
        return self._variable_step(cond, self.steps)

    def _variable_step(self, cond, steps):
        """Optional ODE step count; upstream hardcodes two steps."""
        dt = 1.0 / steps
        with torch.no_grad():
            x = torch.randn(1, cond.shape[1], 80, device=cond.device, dtype=cond.dtype)
            if self.noise_cache is not None:
                x[:, :self.block_size, :] = self.noise_cache
            self.noise_cache = x[:, -self.block_size:, :].clone()
            if self.vc_offset >= 4000:
                self.vc_offset = 0
                self.vc_kv_cache = None
            pre_kv_cache = self.vc_kv_cache
            for i in range(steps):
                t_val = 1.0 - i * dt
                r_val = max(0.0, t_val - dt)
                u, cache = self.vc(x, torch.full((1,), t_val, device=cond.device),
                    torch.full((1,), r_val, device=cond.device), cache=None, cond=cond,
                    spks=self.vc_spk_emb, offset=self.vc_offset, is_inference=True,
                    kv_cache=pre_kv_cache, gtm_kv=self.vc_gtm_kv)
                x = x - dt * u
                if i == steps - 1:
                    self.vc_kv_cache = cache
            self.vc_offset += self.chunk_size
            return x[:, :self.chunk_size, :].transpose(1, 2)


class InferenceEngine:
    sample_rate = 16000
    chunk_samples = 2560  # Official runtime consumes 160 ms in both profiles.

    def __init__(self, model, reference, mode="auto", gpu=0, profile="120ms", steps=2, threads=1, manager=None):
        if not 1 <= steps <= 8 or not 1 <= threads <= 32:
            raise ValueError("Шаги: 1–8; потоки CPU: 1–32")
        self.device = choose_device(mode, gpu)
        torch.set_num_threads(threads)
        torch.backends.cudnn.enabled = False  # Matches the official runtime.
        with self._device_context():
            self.runner = _StreamingRunner(model, reference, self.device, profile, steps, manager or ModelManager())
        for _ in range(3):
            self.process(np.zeros(self.chunk_samples, dtype=np.float32))
        self.reset()

    def _device_context(self):
        return torch.cuda.device(self.device) if self.device.startswith("cuda") else nullcontext()

    def reset(self):
        self.runner._init_cache()

    def process(self, samples):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (self.chunk_samples,) or not np.isfinite(samples).all():
            raise ValueError("Ожидается блок из 2560 конечных отсчётов mono 16 кГц")
        with self._device_context(), torch.inference_mode():
            result = self.runner.process_chunk(samples)
        if result is None:
            return np.empty(0, dtype=np.float32)
        if not np.isfinite(result).all():
            raise ValueError("Инференс вернул некорректное аудио")
        return result.astype(np.float32, copy=False)
