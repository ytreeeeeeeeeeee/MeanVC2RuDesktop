# Included inference sources

* MeanVC2: https://github.com/ASLP-lab/MeanVC2 at
  `13acf84c1bf135ea5edad9c245b345289b06b33e`. ASLP-lab, Apache-2.0
  (license declared in the upstream README; a separate LICENSE is absent upstream).
  The complete, unmodified upstream repository is a pinned git submodule at
  `third_party/MeanVC2`. The app directly imports `runtime/run_rt.py` and
  `runtime/src/`. Only these imported modules and the two configurations from
  `src/config/` enter the executable; training and evaluation code are excluded.
* S3PRL 0.4.18: https://pypi.org/project/s3prl/0.4.18/ (Apache-2.0).
  Only WavLM, its wrapper, base interface and helper are included under
  `app/vendor/s3prl`. WavLM files retain Microsoft's MIT notices.
* WavLM: https://github.com/microsoft/unilm/tree/master/wavlm (MIT).
  Its original implementation is used via S3PRL, not reimplemented.

Licenses are in `app/vendor/LICENSE-*`; original S3PRL file hashes are recorded in
`app/vendor/SOURCE_HASHES.json`. S3PRL's nonempty source files are unchanged.

No MeanVC2 source files are copied or patched. `app/upstream.py` exposes the
upstream runtime and minimal S3PRL import paths. `app/inference.py` replaces
path-dependent initialization and validates weights. It selects the CUDA device
with a PyTorch device context, preserving upstream's CPU/CUDA branch unchanged.
All streaming ASR/BN, DiT KV/noise context and vocoder overlap-add are upstream.
The default two-step ODE calls upstream directly; other step counts use an
explicit adapter extension of that loop, without altering the submodule.
The unused runtime-only `cache_embed` layer keeps its original initialization;
no other missing or unexpected VC weights are allowed.

# Downloaded weights (not embedded in the executable)

* https://huggingface.co/ytre0o0/MeanVC2-ru — Russian 120ms+40ms EMA weights.
  Repository metadata determines the selected revision at download time;
  the local manifest stores its commit and SHA256.
* https://huggingface.co/ASLP-lab/MeanVC2 at
  `39cdd19522fe896c227da691314d9a0e3b995486` — FastU2++ ASR and Vocos.
* https://huggingface.co/lmzjms/wavlm-large at
  `f1c40859a86883dac660f314165a2ea45a8e4dab` — WavLM/ECAPA finetuned checkpoint,
  SHA256 `51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b`.
  This is the public mirror of the checkpoint named in MeanVC2's initialization.
* The tiny `app/assets/wavlm_large_cfg.pt` (2333 bytes) is bundled from
  https://huggingface.co/theoracleguy/MeanVC2-OpenVox-Quality/tree/main/preprocess/ckpts
  with SHA256 `d8c312a42f3a73e969894b59ae918312811d5e6f34744c0e970455af6fefcd38`.
  It contains configuration only and is read by the original speaker loader.
  All downloaded model checkpoints use restricted loading or Safetensors.

Qt/PySide6 (LGPLv3), PyTorch, PortAudio/sounddevice, libsndfile/soundfile,
libsoxr, x-transformers and their dependencies are dynamically bundled by
PyInstaller. Their installed license files accompany the distribution.
PySide6 is used through dynamically loaded DLLs. Source and replacement
instructions: https://doc.qt.io/qtforpython-6/licenses.html and
https://download.qt.io/official_releases/QtForPython/pyside6/.
