from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from app.inference import InferenceEngine, _StreamingRunner
from app.upstream import DiT, UPSTREAM_ROOT, VCRunner


class UpstreamTests(unittest.TestCase):
    def test_runtime_is_imported_from_submodule(self):
        self.assertEqual(Path(VCRunner.process_chunk.__code__.co_filename).resolve(),
                         (UPSTREAM_ROOT / 'runtime/run_rt.py').resolve())
        self.assertEqual(DiT.__module__, 'src.dit')
        for name in ('process_chunk', '_encode_chunk', '_decode_mel', '_init_cache'):
            self.assertIs(getattr(_StreamingRunner, name), getattr(VCRunner, name))

    def test_default_ode_calls_upstream(self):
        runner = _StreamingRunner.__new__(_StreamingRunner)
        runner.steps = 2
        with patch.object(VCRunner, '_vc_step', return_value='upstream') as step:
            self.assertEqual(runner._vc_step('condition'), 'upstream')
            step.assert_called_once_with('condition')

    def test_optional_ode_preserves_upstream_cache_behavior(self):
        def runner(steps):
            instance = _StreamingRunner.__new__(_StreamingRunner)
            instance.steps, instance.block_size, instance.chunk_size = steps, 4, 12
            instance.noise_cache, instance.vc_kv_cache = None, None
            instance.vc_offset = 3996
            instance.vc_spk_emb = instance.vc_gtm_kv = None
            instance.vc = Mock(side_effect=lambda x, t, r, **kw:
                (x * .25 + t.reshape(1, 1, 1), kw['offset'] + 1))
            return instance

        # Same noise, multiple blocks and an offset rollover must match exactly.
        original, extension = runner(2), runner(2)
        cond = torch.zeros(1, 16, 256)
        for seed in range(3):
            torch.manual_seed(seed)
            expected = VCRunner._vc_step(original, cond)
            torch.manual_seed(seed)
            actual = extension._variable_step(cond, 2)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(extension.noise_cache, original.noise_cache)
            self.assertEqual(extension.vc_kv_cache, original.vc_kv_cache)
            self.assertEqual(extension.vc_offset, original.vc_offset)
        for steps in (1, 3, 8):
            instance = runner(steps)
            output = instance._vc_step(cond)
            self.assertEqual(tuple(output.shape), (1, 80, 12))
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(instance.vc.call_count, steps)

    def test_process_selects_gpu_context_without_changing_upstream_device(self):
        engine = InferenceEngine.__new__(InferenceEngine)
        engine.device = 'cuda:1'
        engine.runner = SimpleNamespace(device='cuda', process_chunk=Mock(return_value=None))
        with patch('torch.cuda.device') as context:
            output = engine.process(np.zeros(2560, dtype=np.float32))
            context.assert_called_once_with('cuda:1')
            context.return_value.__enter__.assert_called_once()
            engine.runner.process_chunk.assert_called_once()
        self.assertEqual(engine.runner.device, 'cuda')
        self.assertEqual(output.size, 0)


if __name__ == '__main__':
    unittest.main()
