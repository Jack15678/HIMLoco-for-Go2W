"""Offline checkpoint export; requires torch, onnx and onnxruntime, not Isaac Gym."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'rsl_rl'))
from rsl_rl.export_s10 import main


if __name__ == '__main__':
    main()
