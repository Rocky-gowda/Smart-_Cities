import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "system: end-to-end system tests")
    config.addinivalue_line("markers", "performance: performance tests")
    config.addinivalue_line("markers", "e2e: end-to-end browser tests of the real app")


@pytest.fixture(scope="session")
def artifacts():
    """Skip model-dependent tests if the pipeline has not been run yet."""
    from src import config as C
    needed = [C.MODEL_DIR / "deployed_model.onnx", C.RESULT_DIR / "results.json", C.DATA_DIR / "test.csv"]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        pytest.skip(f"run `python run_all.py` first (missing {missing})")
    return C
