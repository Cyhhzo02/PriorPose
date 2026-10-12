"""Dataset-specific FPS candidate semantics and repeatability."""
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model/pointnet2'))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class FPSTests(unittest.TestCase):
    def tearDown(self):
        from pointnet2_utils import configure_fps
        configure_fps(True)

    def test_small_centered_cloud(self):
        from pointnet2_utils import configure_fps, furthest_point_sample
        xyz = torch.tensor([[[0., 0., 0.], [.01, 0., 0.],
                             [0., .02, 0.], [0., 0., .025]]], device='cuda')
        configure_fps(True)
        self.assertEqual(furthest_point_sample(xyz, 4).tolist(), [[0, 0, 0, 0]])
        configure_fps(False)
        self.assertEqual(furthest_point_sample(xyz, 4).tolist(), [[0, 3, 2, 1]])

    def test_repeat_rng_and_default_restore(self):
        from pointnet2 import _ext
        from pointnet2_utils import configure_fps, furthest_point_sample
        generator = torch.Generator().manual_seed(47)
        xyz = torch.randn(2, 128, 3, generator=generator).cuda().contiguous()
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        for skip in [True, False, True]:
            configure_fps(skip)
            expected = _ext.furthest_point_sampling(xyz, 32, skip)
            for _ in range(3):
                self.assertTrue(torch.equal(furthest_point_sample(xyz, 32), expected))
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state()))
        with self.assertRaises(ValueError):
            configure_fps('false')
