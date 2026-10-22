"""Import the pinned official runtime without changing its source tree."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ROOT = ROOT / "third_party/MeanVC2"
RUNTIME = UPSTREAM_ROOT / "runtime"

if not getattr(sys, "frozen", False) and not (RUNTIME / "run_rt.py").is_file():
    raise RuntimeError("MeanVC2 не загружен. Выполните git submodule update --init --recursive из корня проекта.")

# Upstream uses absolute src.* and s3prl.* imports; expose only its runtime.
sys.path[:0] = [str(RUNTIME), str(ROOT / "app/vendor")]

from run_rt import VCRunner, MODEL_PATHS
from src.dit import DiT
from src.speaker import ECAPA_TDNN_SMALL, extract_embedding
