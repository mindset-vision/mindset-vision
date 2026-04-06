"""fast smoke tests for mindset-vision package."""
import shutil
from pathlib import Path


def test_package_imports():
    """verify core modules import correctly."""
    from mindset.utils.misc import DEFAULTS
    assert "canvas_size" in DEFAULTS

    from mindset.utils.drawing_utils import DrawStimuli
    assert DrawStimuli is not None

    from mindset.utils.dataset_utils import ImageDatasetAnnotations
    assert ImageDatasetAnnotations is not None


def test_generator_registry():
    """verify all 33 generators register via auto-discovery."""
    from mindset.cli import _load_registry
    from mindset.generators import list_generators

    registry = _load_registry()
    assert len(registry) == 33

    cats = list_generators()
    assert len(cats["visual_illusions"]) == 10
    assert len(cats["low_mid_vision"]) == 9
    assert len(cats["shape_recognition"]) == 14


def test_generate_ebbinghaus():
    """smoke test: generate a small ebbinghaus dataset via new path."""
    from mindset.generators.visual_illusions.ebbinghaus import generate_all

    out = Path("/tmp/mindset_ci_ebbinghaus")
    if out.exists():
        shutil.rmtree(out)

    result = generate_all(
        num_samples_scrambled=2,
        num_samples_illusory=1,
        output_folder=str(out),
    )

    assert Path(result).exists()
    assert (out / "annotation.csv").exists()
    assert len(list(out.rglob("*.png"))) >= 3
    shutil.rmtree(out)


def test_generate_legacy():
    """smoke test: verify old import path still works."""
    from mindset.generate_datasets.visual_illusions.ebbinghaus_illusion.generate_dataset import generate_all

    out = Path("/tmp/mindset_ci_legacy")
    if out.exists():
        shutil.rmtree(out)

    result = generate_all(
        num_samples_scrambled=2,
        num_samples_illusory=1,
        output_folder=str(out),
    )

    assert Path(result).exists()
    assert (out / "annotation.csv").exists()
    shutil.rmtree(out)


def test_cli_entry_point():
    """verify the CLI entry point is callable."""
    from mindset.cli import main
    assert callable(main)
