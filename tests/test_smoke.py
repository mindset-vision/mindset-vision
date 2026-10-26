"""fast smoke tests for mindset-vision package."""
import shutil
from pathlib import Path


def test_package_imports():
    """verify core modules import correctly."""
    from mindset.utils import DEFAULTS
    assert "canvas_size" in DEFAULTS

    from mindset.drawing.base import DrawStimuli
    assert DrawStimuli is not None


def test_generator_registry():
    """verify all 37 generators register via auto-discovery."""
    from mindset.cli import _load_registry
    from mindset.generators import list_generators

    registry = _load_registry()
    assert len(registry) == 37

    cats = list_generators()
    assert len(cats["visual_illusions"]) == 10
    assert len(cats["low_mid_vision"]) == 10
    assert len(cats["shape_recognition"]) == 17


def test_generate_composite_textures(tmp_path):
    """smoke test: generate a small composite_textures dataset."""
    from mindset.generators.shape_recognition.composite_textures import generate_all

    out = tmp_path / "composite-textures-smoke"

    result = generate_all(
        fg_texture_mode="lines",
        bg_texture_mode="dots",
        output_folder=str(out),
    )

    assert Path(result).exists()
    assert (out / "annotation.csv").exists()
    assert len(list(out.rglob("*.png"))) > 0





def test_generate_ebbinghaus():
    """smoke test: generate a small ebbinghaus dataset."""
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


def test_generate_ponzo_random_target_lines(tmp_path):
    """smoke test: generate ponzo scrambled samples with random targets."""
    from mindset.generators.visual_illusions.ponzo import generate_all

    out = tmp_path / "ponzo-random-target-lines"

    result = generate_all(
        num_samples_scrambled=1,
        num_samples_illusory=1,
        rnd_target_lines=True,
        output_folder=str(out),
    )

    assert Path(result).exists()
    assert (out / "annotation.csv").exists()
    assert len(list(out.rglob("*.png"))) == 3


def test_cli_entry_point():
    """verify the CLI entry point is callable."""
    from mindset.cli import main
    assert callable(main)


def test_cli_parse_generator_args():
    """verify CLI generator arguments are dynamically parsed without skipping base fields."""
    from mindset.cli import _parse_generator_args
    from mindset.generators.low_mid_vision.amodal_completion import AmodalCompletionConfig

    args = ["--num-samples", "2", "--canvas-size", "256", "256", "-o", "/tmp/test_out"]
    parsed = _parse_generator_args(AmodalCompletionConfig, args)

    assert parsed["num_samples"] == 2
    assert parsed["canvas_size"] == [256, 256]
    assert parsed["output_folder"] == "/tmp/test_out"



def test_generate_relational_vs_coordinate(tmp_path):
    """smoke test: generate a small relational vs coordinate dataset."""
    from mindset.generators.low_mid_vision.relational_vs_coordinate import generate_all

    out = tmp_path / "relational-vs-coordinate-smoke"

    result = generate_all(
        num_samples=2,
        output_folder=str(out),
    )

    assert Path(result).exists()
    assert (out / "annotation.csv").exists()
    # 2 samples * 3 conditions (basis, coord change, relation change) = 6 images
    assert len(list(out.rglob("*.png"))) == 6


def test_generate_manipulated_textures(tmp_path):
    """smoke test: generate a small manipulated_textures dataset."""
    from mindset.generators.shape_recognition.manipulated_textures import generate_all

    out = tmp_path / "manipulated-textures-smoke"

    result = generate_all(
        texture_mode="lines",
        output_folder=str(out),
    )

    assert Path(result).exists()
    assert (out / "annotation.csv").exists()
    assert len(list(out.rglob("*.png"))) > 0


def test_generate_manipulated_outlines(tmp_path):
    """smoke test: generate a small manipulated_outlines dataset."""
    from mindset.generators.shape_recognition.manipulated_outlines import generate_all

    out = tmp_path / "manipulated-outlines-smoke"

    result = generate_all(
        outline_mode="dotted",
        output_folder=str(out),
    )

    assert Path(result).exists()
    assert (out / "annotation.csv").exists()
    assert len(list(out.rglob("*.png"))) > 0


def test_jastrow_random_same_size_annotation(tmp_path, monkeypatch):
    """record the sizes passed to the drawer, not the original random sizes."""
    import csv
    import random

    from mindset.generators.visual_illusions import jastrow

    size_red, size_blue = 0.01, 0.05
    size = (size_red + size_blue) / 2
    monkeypatch.setattr(
        jastrow, "get_random_params", lambda: (50, 0.1, size_red, size_blue)
    )
    monkeypatch.setattr(jastrow.random, "uniform", random.Random(0).uniform)
    drawn_sizes = []
    draw = jastrow.DrawJastrow.generate_jastrow_illusion

    def record_sizes(self, arc, width, red, blue, top_color, type_stimulus):
        drawn_sizes.append((red, blue))
        return draw(self, arc, width, red, blue, top_color, type_stimulus)

    monkeypatch.setattr(jastrow.DrawJastrow, "generate_jastrow_illusion", record_sizes)
    out = tmp_path / "jastrow-same-size"
    jastrow.generate_all(
        jastrow.JastrowConfig(
            num_samples_illusory=0,
            num_samples_random=0,
            num_samples_aligned=0,
            num_samples_random_same_size=1,
            output_folder=str(out),
        )
    )

    with (out / "annotation.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert drawn_sizes == [(size, size)] * 2
    for row in rows:
        assert row["Type"] == "random_same_size"
        assert float(row["SizeRed"]) == float(row["SizeBlue"]) == size
        assert float(row["SizeRedMinusBlue"]) == 0
        path = Path(row["Path"])
        assert path.name.startswith(f"red{size:.2f}_blue{size:.2f}_0_")
        assert (out / path).is_file()

