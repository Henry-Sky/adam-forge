"""Load the repository's upstream float Gaussian PLY reader without Torch."""
import importlib.util
from pathlib import Path


def ply_module():
    path = Path(__file__).resolve().parents[1]/'third_party/asset-harvester/asset_harvester/tokengs/ply_io.py'
    spec = importlib.util.spec_from_file_location('binding_ply', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
