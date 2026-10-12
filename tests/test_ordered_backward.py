"""Small GPU fixtures; CPU scatter is a test oracle, not a training backend."""
import hashlib
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model/pointnet2'))
from ordered_backward import OrderedCudaBackward


def reference(grad, indices, count, operation, seed, weight=None):
    values = grad.cpu()
    if weight is not None:
        values = values.unsqueeze(-1) * weight.cpu().unsqueeze(1)
    values = values.reshape(values.shape[0], values.shape[1], -1)
    idx = indices.cpu().long().reshape(indices.shape[0], 1, -1)
    if seed is not None:
        key = f'priorpose-reduction-v1:{seed}:{operation}:{values.shape[2]}:{count}'
        local_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')
        order = torch.randperm(values.shape[2], generator=torch.Generator().manual_seed(local_seed))
        values, idx = values.index_select(2, order), idx.index_select(2, order)
    out = torch.zeros(values.shape[0], values.shape[1], count)
    return out.scatter_add_(2, idx.expand(-1, values.shape[1], -1), values)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class OrderedBackwardTests(unittest.TestCase):
    def test_exact_reference_repeat_and_rng(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(73)
        for seed in [None, 1, 2]:
            backend = OrderedCudaBackward(seed)
            for operation in ['gather', 'group', 'interpolate']:
                for fixture in ['random', 'collision', 'cancellation', 'subnormal']:
                    with self.subTest(seed=seed, operation=operation, fixture=fixture):
                        shape = (2, 101) if operation == 'gather' else (2, 37, 3)
                        idx = torch.randint(19, shape, generator=generator, dtype=torch.int32)
                        if fixture in ('collision', 'cancellation'):
                            idx.zero_()
                        gs = (2, 7, 37) if operation == 'interpolate' else (2, 7, *shape[1:])
                        grad = torch.randn(gs, generator=generator)
                        if fixture == 'cancellation':
                            grad.flatten()[::3] = 1e10
                            grad.flatten()[1::3] = -1e10
                        if fixture == 'subnormal':
                            grad *= 1e-39
                        weight = torch.rand(shape, generator=generator) if operation == 'interpolate' else None
                        expected = reference(grad, idx, 19, operation, seed, weight)
                        args = (grad, idx, weight, 19) if weight is not None else (grad, idx, 19)
                        args = tuple(v.cuda() if torch.is_tensor(v) else v for v in args)
                        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                        for _ in range(3):
                            actual = getattr(backend, operation)(*args).cpu()
                            self.assertTrue(torch.equal(expected.view(torch.int32), actual.view(torch.int32)))
                        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
                        self.assertTrue(torch.equal(gpu_rng, torch.cuda.get_rng_state()))

    def test_native_restore(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import pointnet2_utils as p
        originals = dict(p._native_backward)
        p.configure_backward('ordered_cuda', 2)
        self.assertNotEqual(p._ext.gather_points_grad, originals['gather_points_grad'])
        p.configure_backward('native')
        for name, function in originals.items():
            self.assertIs(getattr(p._ext, name), function)

    def test_nondefault_stream(self):
        backend = OrderedCudaBackward(2)
        generator = torch.Generator().manual_seed(19)
        grad = torch.randn(2, 8, 64, 16, generator=generator)
        idx = torch.randint(128, (2, 64, 16), generator=generator, dtype=torch.int32)
        expected = reference(grad, idx, 128, 'group', 2)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            result = backend.group(grad.cuda(), idx.cuda(), 128)
        stream.synchronize()
        self.assertTrue(torch.equal(expected.view(torch.int32), result.cpu().view(torch.int32)))


if __name__ == '__main__':
    unittest.main()
