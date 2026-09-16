"""Run with Python + torch; no simulator and no policy updates."""
import importlib.util
from pathlib import Path
import torch

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


if __name__ == '__main__':
    main()
