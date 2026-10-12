"""Public recipe and backend-selection contracts (no GPU required)."""
import ast
from pathlib import Path
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]


class RecipeTests(unittest.TestCase):
    def test_nocs_recipe(self):
        cfg = yaml.safe_load((ROOT / 'config/nocs.yaml').read_text())
        self.assertEqual((cfg['seed'], cfg['pointnet_reduction_seed']), (1, 2))
        self.assertTrue(cfg['deterministic'])
        self.assertEqual(cfg['pointnet_backend'], 'ordered_cuda')
        self.assertEqual(cfg['epochs'] * cfg['data']['steps_per_epoch'], 120000)
        self.assertEqual(cfg['schedule']['total_steps'], 120000)
        self.assertEqual(cfg['schedule']['learning_rate'], 0.0006)

    def test_shared_architecture_and_dataset_settings(self):
        configs = [yaml.safe_load((ROOT / f'config/{n}.yaml').read_text())
                   for n in ['nocs', 'housecat6d']]
        for cfg in configs:
            self.assertEqual(cfg['model']['transformer']['blocks'], 6)
            self.assertFalse(cfg['model']['transformer']['share_parameters'])
            self.assertTrue(cfg['loss']['nocs_symmetry_adjusted_rotation'])
            self.assertTrue(cfg['loss']['camera_symmetry_adjusted_rotation'])
        housecat = configs[1]
        self.assertFalse(housecat.get('deterministic', False))
        self.assertEqual(housecat.get('pointnet_backend', 'native'), 'native')
        self.assertFalse(housecat['pointnet_fps_skip_near_origin'])
        self.assertTrue(configs[0].get('pointnet_fps_skip_near_origin', True))
        self.assertEqual(housecat['loss']['symmetry_ids'], [1, 2, 7])

    def test_portable_paths(self):
        for name in ['nocs', 'housecat6d']:
            cfg = yaml.safe_load((ROOT / f'config/{name}.yaml').read_text())
            for path in [cfg['data']['root'], cfg['data']['mean_shape_path'],
                         cfg['model']['dino_repository']]:
                self.assertFalse(Path(path).is_absolute())

    def test_python_syntax(self):
        for folder in ['model', 'provider', 'utils', 'scripts', 'tests']:
            for path in (ROOT / folder).rglob('*.py'):
                if 'build' not in path.parts:
                    ast.parse(path.read_text(), filename=str(path))

    def test_dinov2_revision(self):
        revision = 'e1277af2ba9496fbadf7aec6eba56e8d882d1e35'
        setup = (ROOT / 'scripts/setup_dinov2.sh').read_text()
        documentation = (ROOT / 'docs/RELEASE_SCOPE.md').read_text()
        self.assertIn('revision=' + revision, setup)
        self.assertIn(revision, documentation)
        self.assertIn('${destination}/.git', setup)


if __name__ == '__main__':
    unittest.main()
