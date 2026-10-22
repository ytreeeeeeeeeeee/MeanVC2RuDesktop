# The CUDA bundle also runs on CPU; model weights live outside the distribution.
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, copy_metadata
from importlib.metadata import distributions
import argparse
from pathlib import Path, PureWindowsPath

parser = argparse.ArgumentParser()
parser.add_argument('--onefile', action='store_true')
options = parser.parse_args()

root = Path(SPECPATH)
runtime = root / 'third_party/MeanVC2/runtime'
vendor = root / 'app/vendor'
datas = collect_data_files('app', includes=['assets/*.pt', 'vendor/LICENSE*'])
datas += [(str(root / f'third_party/MeanVC2/src/config/config_{profile}_40ms.json'),
           'third_party/MeanVC2/src/config') for profile in ('120ms', '40ms')]
datas += copy_metadata('x-transformers') + copy_metadata('einx')
for distribution in distributions():
    for file in distribution.files or []:
        if any(part.lower().startswith(('license', 'copying', 'notice')) for part in file.parts):
            source = distribution.locate_file(file)
            if source.is_file():
                datas.append((str(source), 'licenses/' + distribution.metadata['Name'] + '/' + str(file.parent)))
a = Analysis(['run.py'], pathex=[SPECPATH, str(runtime), str(vendor)], datas=datas,
    module_collection_mode={'x_transformers': 'pyz+py', 'einx': 'pyz+py'},
    binaries=collect_dynamic_libs('soxr'),
    hiddenimports=['run_rt', 's3prl.upstream.wavlm.WavLM',
                   's3prl.upstream.wavlm.expert', 's3prl.upstream.interfaces'],
    excludes=['tkinter', 'matplotlib', 'scipy', 'tensorboard', 'IPython', 'pytest'])
pyz = PYZ(a.pure)
if options.onefile:
    # shortcut: upstream disables cuDNN; restore its precompiled engines if enabling cuDNN.
    binaries = [entry for entry in a.binaries
                if PureWindowsPath(entry[0]).name.lower() != 'cudnn_engines_precompiled64_9.dll']
    exe = EXE(pyz, a.scripts, binaries, a.datas, name='MeanVC2Desktop',
              console=False, debug=False, strip=False, upx=False)
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name='MeanVC2Desktop',
              console=False, debug=False, strip=False, upx=False)
    coll = COLLECT(exe, a.binaries, a.datas, name='MeanVC2Desktop', strip=False, upx=False)
