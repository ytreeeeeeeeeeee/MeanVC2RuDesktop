from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class PackagingTests(unittest.TestCase):
    def check_layout(self, onefile):
        spec = Path(__file__).resolve().parents[1] / "MeanVC2Desktop.spec"
        analysis = SimpleNamespace(pure=[], scripts=["entry"],
            binaries=[("torch\\lib\\torch_cuda.dll", "cuda", "BINARY"),
                      ("torch\\lib\\cudnn_engines_precompiled64_9.dll", "engines", "BINARY")],
            datas=["config.json"])
        executable = Mock(return_value="exe")
        collect = Mock()
        namespace = dict(SPECPATH=str(spec.parent), Analysis=Mock(return_value=analysis),
                         PYZ=Mock(return_value="pyz"), EXE=executable, COLLECT=collect)
        arguments = [str(spec), "--onefile"] if onefile else [str(spec)]
        with patch("sys.argv", arguments), patch("importlib.metadata.distributions", return_value=[]), \
                patch("PyInstaller.utils.hooks.collect_data_files", return_value=[]), \
                patch("PyInstaller.utils.hooks.collect_dynamic_libs", return_value=[]), \
                patch("PyInstaller.utils.hooks.copy_metadata", return_value=[]):
            runpy.run_path(str(spec), init_globals=namespace)
        options = namespace['Analysis'].call_args.kwargs
        self.assertEqual(options['pathex'], [str(spec.parent),
            str(spec.parent / 'third_party/MeanVC2/runtime'), str(spec.parent / 'app/vendor')])
        self.assertEqual(options['datas'], [(str(spec.parent / f'third_party/MeanVC2/src/config/config_{p}_40ms.json'),
            'third_party/MeanVC2/src/config') for p in ('120ms', '40ms')])
        self.assertIn('run_rt', options['hiddenimports'])
        if onefile:
            self.assertEqual(executable.call_args.args, ("pyz", analysis.scripts, analysis.binaries[:1], analysis.datas))
            self.assertFalse(executable.call_args.kwargs.get("exclude_binaries", False))
            collect.assert_not_called()
        else:
            self.assertTrue(executable.call_args.kwargs["exclude_binaries"])
            collect.assert_called_once_with("exe", analysis.binaries, analysis.datas,
                                            name="MeanVC2Desktop", strip=False, upx=False)

    def test_onefile_includes_runtime_in_exe(self):
        self.check_layout(True)

    def test_default_onedir_layout_is_preserved(self):
        self.check_layout(False)


if __name__ == "__main__":
    unittest.main()
