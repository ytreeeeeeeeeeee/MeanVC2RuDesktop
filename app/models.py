"""Pinned auxiliary models, HF discovery, downloads and offline cache."""
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
import requests

from app.settings import data_dir

REPO = "ytre0o0/MeanVC2-ru"
# Immutable sources; the speaker checkpoint is the same WavLM/ECAPA used upstream.
AUX = {
    "asr120ms": ("ASLP-lab/MeanVC2", "fastu2pp_160ms.pt", "39cdd19522fe896c227da691314d9a0e3b995486"),
    "asr40ms": ("ASLP-lab/MeanVC2", "fastu2pp_80ms.pt", "39cdd19522fe896c227da691314d9a0e3b995486"),
    "vocoder": ("ASLP-lab/MeanVC2", "vocos.pt", "39cdd19522fe896c227da691314d9a0e3b995486"),
    "speaker": ("lmzjms/wavlm-large", "wavlm_large_finetune.pth", "f1c40859a86883dac660f314165a2ea45a8e4dab"),
}


class ModelManager:
    def __init__(self, root=None):
        self.root = Path(root) if root else data_dir() / "models"
        self.root.mkdir(parents=True, exist_ok=True)

    def discover(self):
        names = sorted(n for n in HfApi().list_repo_files(REPO) if n.endswith(".safetensors"))
        if not names:
            raise ValueError("В репозитории нет моделей Safetensors")
        temporary = self.root / "catalog.tmp"
        temporary.write_text(json.dumps(names), encoding="utf-8")
        temporary.replace(self.root / "catalog.json")
        return names

    def catalog(self):
        try:
            names = json.loads((self.root / "catalog.json").read_text(encoding="utf-8"))
            return [n for n in names if isinstance(n, str) and n.endswith(".safetensors")]
        except (OSError, ValueError, TypeError):
            return ["meanvc2_ru.safetensors"]

    def fetch(self, repo, name, revision="main", *, offline=False, progress=None):
        folder = self.root / repo.replace("/", "--")
        folder.mkdir(parents=True, exist_ok=True)
        ref = folder / (hashlib.sha256(f"{revision}/{name}".encode()).hexdigest() + ".json")
        cached = None
        try:
            saved = json.loads(ref.read_text())
            digest = saved["sha256"]
            if len(digest) == 64 and all(c in "0123456789abcdef" for c in digest):
                candidate = folder / digest
                if candidate.stat().st_size == saved["size"]:
                    cached = candidate
        except (OSError, ValueError, KeyError, TypeError):
            pass
        if offline:
            if cached:
                return cached
            raise FileNotFoundError(f"Нет локального файла {name}. Нажмите «Скачать / обновить» при подключении к интернету.")
        metadata = get_hf_file_metadata(hf_hub_url(repo, name, revision=revision))
        expected = metadata.etag
        if cached and cached.name == expected:
            if progress:
                progress(name, metadata.size, metadata.size)
            return cached
        temporary = folder / (ref.stem + ".part")
        digest = hashlib.sha256()
        count = 0
        if progress:
            progress(name, 0, metadata.size or 0)
        with requests.get(metadata.location, stream=True, timeout=(15, 30)) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for block in response.iter_content(1024 * 1024):
                    output.write(block)
                    digest.update(block)
                    count += len(block)
                    if progress:
                        progress(name, count, metadata.size or 0)
        if count != metadata.size or (len(expected) == 64 and digest.hexdigest() != expected):
            temporary.unlink(missing_ok=True)
            raise ValueError(f"Повреждён файл {name}: не совпадают размер или SHA256")
        target = folder / digest.hexdigest()
        temporary.replace(target)
        ref_tmp = ref.with_suffix(".tmp")
        ref_tmp.write_text(json.dumps(dict(sha256=digest.hexdigest(), size=count, filename=name,
                                           commit=metadata.commit_hash)), encoding="utf-8")
        ref_tmp.replace(ref)
        return target

    def model(self, selection, *, offline=True, progress=None):
        path = Path(selection).expanduser()
        if path.is_absolute():
            if not path.is_file() or path.suffix.lower() != ".safetensors":
                raise ValueError("Выберите существующий файл .safetensors")
            return path
        if selection not in self.catalog():
            raise ValueError("Неизвестная модель. Обновите список моделей.")
        return self.fetch(REPO, selection, offline=offline, progress=progress)

    def auxiliaries(self, profile, *, offline=True, progress=None):
        return {key: self.fetch(*AUX[source], offline=offline, progress=progress)
                for key, source in (("asr", "asr" + profile), ("vocoder", "vocoder"), ("speaker", "speaker"))}

    def download(self, selection, profile, progress=None):
        path = self.model(selection, offline=False, progress=progress)
        self.auxiliaries(profile, offline=False, progress=progress)
        return path
