"""Run with Python + torch; no simulator and no policy updates."""
import importlib.util
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('curriculum', ROOT / 'legged_gym/utils/curriculum.py')
curriculum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curriculum)


def main():
    torch.manual_seed(1)
    columns = torch.arange(4096) * 22 // 4096
    mask = curriculum.stratified_extended_mask(columns)
    for column in columns.unique():
        group = columns == column
        assert abs(float(mask[group].float().mean()) - .2) < .006
    assert mask[columns >= 20].any() and mask[columns < 4].any()
    assert not torch.equal(mask, torch.arange(4096) < 819)
    print('Every terrain column has randomly assigned approximately 20% extended-speed environments.')
    spec = importlib.util.spec_from_file_location('pebbles', ROOT / 'legged_gym/utils/pebbles.py')
    pebbles = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pebbles)
    raw = pebbles.pebble_heightfield(8., 8., .025, .001, .08, 5., np.random.default_rng(2))
    assert raw.shape == (321, 321) and 40 < raw.max() <= 80
    assert (raw[100:221, 100:221] == 0).all()  # 3 m spawn platform.
    assert not raw[[0, -1], :].any() and not raw[:, [0, -1]].any()
    assert (raw > 0).sum() > 5000
    assert np.unique(raw).size > 30  # Curved caps, not binary-height blocks.
    print('Pebble caps have curved collision heights, a clear spawn platform and seamless zero boundaries.')


if __name__ == '__main__':
    main()
