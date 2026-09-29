"""unit tests for uncrowding dataset generator."""

import csv
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from mindset.generators.low_mid_vision.uncrowding import (
    CIRCLE,
    HEXAGON,
    SQUARE,
    STAR,
    DrawUncrowding,
    UncrowdingDistributionsConfig,
    UncrowdingShapesConfig,
    generate_distributions,
    generate_grid_layout,
    generate_shapes,
    generate_uncrowding_stimulus,
    get_valid_arrangements,
    serialize_grid,
)


def test_grid_layout_all_valid_dimensions():
    """verify grid layouts for all R <= C with odd R in {1,3,5} and C in {1,3,5,7}."""
    row_counts = [1, 3, 5]
    col_counts = [1, 3, 5, 7]
    grid_sizes = [
        (r, c)
        for r in row_counts
        for c in col_counts
        if r <= c
    ]
    assert len(grid_sizes) == 9

    arrangements = ["uniform", "interleaved_cols", "interleaved_rows", "interleaved_both"]

    for r, c in grid_sizes:
        r_c = r // 2
        c_c = c // 2

        for arr in arrangements:
            grid = generate_grid_layout(r, c, arr)
            assert grid.shape == (r, c)
            # Center cell surrounding vernier must ALWAYS be 0
            assert grid[r_c, c_c] == 0

            if arr == "uniform":
                assert np.all(grid == 0)
            elif arr == "interleaved_cols":
                # Each column is constant across rows
                for col_idx in range(c):
                    assert len(set(grid[:, col_idx])) == 1
                    expected_val = abs(col_idx - c_c) % 2
                    assert grid[0, col_idx] == expected_val
            elif arr == "interleaved_rows":
                # Each row is constant across columns
                for row_idx in range(r):
                    assert len(set(grid[row_idx, :])) == 1
                    expected_val = abs(row_idx - r_c) % 2
                    assert grid[row_idx, 0] == expected_val
            elif arr == "interleaved_both":
                for row_idx in range(r):
                    for col_idx in range(c):
                        expected_val = (abs(row_idx - r_c) + abs(col_idx - c_c)) % 2
                        assert grid[row_idx, col_idx] == expected_val


def test_valid_arrangements():
    """verify non-degenerate arrangement filtering."""
    assert get_valid_arrangements(1, 1) == ["uniform"]
    assert get_valid_arrangements(1, 3) == ["uniform", "interleaved_cols"]
    assert get_valid_arrangements(3, 3) == [
        "uniform",
        "interleaved_cols",
        "interleaved_rows",
        "interleaved_both",
    ]
    assert get_valid_arrangements(5, 7) == [
        "uniform",
        "interleaved_cols",
        "interleaved_rows",
        "interleaved_both",
    ]


def test_serialize_grid():
    """verify grid serialization format."""
    grid = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    serialized = serialize_grid(grid)
    assert serialized == "3x3:010/101/010"


def test_draw_shapes_and_transformations():
    """test shape rendering with rotation and scale."""
    drawer = DrawUncrowding(
        canvas_size=(512, 512),
        background=(0, 0, 0),
        bar_width=2,
        bar_height=7,
        flanker_bar_width=3,
    )
    shape_size = 64

    for shape_id in [SQUARE, CIRCLE, HEXAGON, STAR, "square", "star"]:
        patch = drawer.draw_shape(shape_id, shape_size=shape_size, rotation=25.0, scale=0.8)
        assert isinstance(patch, Image.Image)
        assert patch.size == (shape_size, shape_size)

    # Star-specific verification: 7-pointed star outline
    star_patch = drawer.draw_star_7(shape_size=shape_size, rotation=0.0, scale=1.0)
    assert star_patch.size == (shape_size, shape_size)

    # Vernier verification
    v0 = drawer.draw_vernier(vernier_type=0, offset_pixels=2)
    v1 = drawer.draw_vernier(vernier_type=1, offset_pixels=4)
    assert isinstance(v0, Image.Image)
    assert isinstance(v1, Image.Image)


def test_star_vernier_clearance_no_clipping():
    """verify that the 7-pointed star outline does NOT intersect/clip the central Vernier."""
    drawer = DrawUncrowding(
        canvas_size=(512, 512),
        background=(0, 0, 0),
        bar_width=2,
        bar_height=7,
        flanker_bar_width=3,
    )
    shape_size = 64

    # Draw star outline patch
    star_patch = drawer.draw_star_7(shape_size=shape_size, rotation=0.0, scale=1.0)
    star_arr = np.array(star_patch.convert("L")) > 0

    # Draw Vernier patch
    v_patch = drawer.draw_vernier(vernier_type=0, offset_pixels=2)
    vw, vh = v_patch.size

    # Position Vernier at center of shape patch
    v_full = Image.new("L", (shape_size, shape_size), 0)
    vx = (shape_size - vw) // 2
    vy = (shape_size - vh) // 2
    v_full.paste(v_patch.convert("L"), (vx, vy))
    v_arr = np.array(v_full) > 0

    # Star outline and Vernier lines must NOT intersect (clearance > 0)
    overlap = np.logical_and(star_arr, v_arr)
    assert np.sum(overlap) == 0, f"Star clipped Vernier at {np.sum(overlap)} pixels!"


def test_square_rotation_boundary_and_vernier_clearance():
    """verify that square corners never clip cell boundary or Vernier at any rotation."""
    drawer = DrawUncrowding(
        canvas_size=(512, 512),
        background=(0, 0, 0),
        bar_width=2,
        bar_height=7,
        flanker_bar_width=3,
    )
    shape_size = 64

    # Vernier mask
    v_patch = drawer.draw_vernier(vernier_type=0, offset_pixels=2)
    vw, vh = v_patch.size
    v_full = Image.new("L", (shape_size, shape_size), 0)
    v_full.paste(v_patch.convert("L"), ((shape_size - vw) // 2, (shape_size - vh) // 2))
    v_arr = np.array(v_full) > 0

    for rot in range(0, 360, 15):
        sq = drawer.draw_square(shape_size=shape_size, rotation=float(rot), scale=1.0)
        sq_arr = np.array(sq.convert("L")) > 0

        # Check 1: Zero overlap with Vernier
        overlap = np.sum(np.logical_and(sq_arr, v_arr))
        assert overlap == 0, f"Square at rot={rot} clipped Vernier at {overlap} pixels!"

        # Check 2: Zero pixels on the outer boundary border (row 0, row -1, col 0, col -1)
        border_pixels = (
            np.sum(sq_arr[0, :])
            + np.sum(sq_arr[-1, :])
            + np.sum(sq_arr[:, 0])
            + np.sum(sq_arr[:, -1])
        )
        assert border_pixels == 0, f"Square at rot={rot} clipped cell boundary ({border_pixels} border pixels)!"


def test_generate_uncrowding_stimulus():
    """test full stimulus generation for inside and outside conditions."""
    drawer = DrawUncrowding(
        canvas_size=(512, 512),
        background=(0, 0, 0),
        bar_width=2,
        bar_height=7,
        flanker_bar_width=3,
    )
    r, c = 3, 5
    cell_shapes = [["square"] * c for _ in range(r)]
    cell_rotations = [[0.0] * c for _ in range(r)]
    cell_scales = [[1.0] * c for _ in range(r)]

    img_inside = generate_uncrowding_stimulus(
        drawer=drawer,
        cell_shapes=cell_shapes,
        cell_rotations=cell_rotations,
        cell_scales=cell_scales,
        vernier_type=0,
        vernier_offset=2,
        vernier_in_out="inside",
        shape_size=64,
        canvas_size=(512, 512),
    )
    assert img_inside.size == (512, 512)

    img_outside = generate_uncrowding_stimulus(
        drawer=drawer,
        cell_shapes=cell_shapes,
        cell_rotations=cell_rotations,
        cell_scales=cell_scales,
        vernier_type=1,
        vernier_offset=2,
        vernier_in_out="outside",
        shape_size=64,
        canvas_size=(512, 512),
    )
    assert img_outside.size == (512, 512)


def test_uncrowding_shapes_generator(tmp_path):
    """end-to-end smoke test for uncrowding_shapes generator."""
    output_dir = tmp_path / "shapes_test"
    config = UncrowdingShapesConfig(
        shapes=["square", "star"],
        row_counts=[1, 3],
        col_counts=[1, 3],
        num_samples_vernier_inside=2,
        num_samples_vernier_outside=2,
        output_folder=str(output_dir),
    )

    result_folder = generate_shapes(config)
    assert Path(result_folder).exists()

    csv_path = output_dir / "annotation.csv"
    assert csv_path.exists()

    with open(csv_path) as f:
        reader = list(csv.DictReader(f))
        assert len(reader) > 0
        for row in reader:
            assert row["ConditionType"] == "two_shapes"
            assert row["VernierInOut"] in ("inside", "outside")
            assert row["CenterShape"] in ("square", "star")
            img_file = output_dir / row["Path"]
            assert img_file.exists()


def test_uncrowding_distributions_generator(tmp_path):
    """end-to-end smoke test for uncrowding_distributions generator with rotation."""
    output_dir = tmp_path / "distributions_test_rot"
    config = UncrowdingDistributionsConfig(
        dimension="rotation",
        loc=[0.0, 45.0],
        scale=[0.0, 5.0],
        base_shapes=["square", "star"],
        row_counts=[1, 3],
        col_counts=[1, 3],
        num_samples_vernier_inside=2,
        num_samples_vernier_outside=2,
        output_folder=str(output_dir),
    )

    result_folder = generate_distributions(config)
    assert Path(result_folder).exists()

    csv_path = output_dir / "annotation.csv"
    assert csv_path.exists()

    with open(csv_path) as f:
        reader = list(csv.DictReader(f))
        assert len(reader) > 0
        for row in reader:
            assert row["ConditionType"] == "two_distributions"
            assert row["VernierInOut"] in ("inside", "outside")
            assert row["Dimension"] == "rotation"
            assert row["BaseShape"] in ("square", "star")
            assert float(row["LocA"]) in (0.0, 45.0)
            assert float(row["ScaleA"]) in (0.0, 5.0)
            assert float(row["LocB"]) in (0.0, 45.0) if row['LocB'] != 'none' else True
            assert float(row["ScaleB"]) in (0.0, 5.0) if row['ScaleB'] != 'none' else True
            img_file = output_dir / row["Path"]
            assert img_file.exists()


def test_uncrowding_distributions_scale_generator(tmp_path):
    """end-to-end smoke test for uncrowding_distributions generator with scale."""
    output_dir = tmp_path / "distributions_test_scale"
    config = UncrowdingDistributionsConfig(
        dimension="scale",
        loc=[0.5, 0.6],
        scale=[0.5, 0.05],
        base_shapes=["hexagon"],
        row_counts=[1, 3],
        col_counts=[1, 3],
        num_samples_vernier_inside=1,
        num_samples_vernier_outside=1,
        output_folder=str(output_dir),
    )

    result_folder = generate_distributions(config)
    assert Path(result_folder).exists()

    csv_path = output_dir / "annotation.csv"
    assert csv_path.exists()

    with open(csv_path) as f:
        reader = list(csv.DictReader(f))
        assert len(reader) > 0
        for row in reader:
            assert row["Dimension"] == "scale"
            assert float(row["LocA"]) in (0.5, 0.6)
            assert float(row["ScaleA"]) in (0.5, 0.05)
            assert float(row["LocB"]) in (0.5, 0.6) if row['LocB'] != 'none' else True
            assert float(row["ScaleB"]) in (0.5, 0.05) if row['ScaleB'] != 'none' else True
            img_file = output_dir / row["Path"]
            assert img_file.exists()


def test_uncrowding_distributions_validation():
    """verify distribution config validation and normalization."""
    # Invalid dimension
    with pytest.raises(ValueError, match="dimension must be 'rotation' or 'scale'"):
        UncrowdingDistributionsConfig(dimension="invalid")

    with pytest.raises(ValueError, match="dimension must be 'rotation' or 'scale'"):
        UncrowdingDistributionsConfig(dimension="all")

    # Mismatched loc and scale lengths
    with pytest.raises(ValueError, match="loc and scale lists must have equal length"):
        UncrowdingDistributionsConfig(loc=[0.0, 45.0], scale=[0.0])

    # Multiple values
    cfg_multi = UncrowdingDistributionsConfig(loc=[15.0, 30.0], scale=[1.0, 2.0])
    assert cfg_multi.loc == [15.0, 30.0]
    assert cfg_multi.scale == [1.0, 2.0]


def test_vernier_offset_configuration_and_normalization():
    """verify vernier_offset default preservation, singleton conversion, and list handling."""
    # Shapes config
    default_shapes = UncrowdingShapesConfig()
    assert default_shapes.vernier_offset == [4]

    single_int_shapes = UncrowdingShapesConfig(vernier_offset=6)
    assert single_int_shapes.vernier_offset == [6]

    single_str_shapes = UncrowdingShapesConfig(vernier_offset="8")
    assert single_str_shapes.vernier_offset == [8]

    list_shapes = UncrowdingShapesConfig(vernier_offset=[2, 4, 8])
    assert list_shapes.vernier_offset == [2, 4, 8]

    # Distributions config
    default_dist = UncrowdingDistributionsConfig()
    assert default_dist.vernier_offset == [4]

    single_dist = UncrowdingDistributionsConfig(vernier_offset=3)
    assert single_dist.vernier_offset == [3]

    list_dist = UncrowdingDistributionsConfig(vernier_offset=[2, 5])
    assert list_dist.vernier_offset == [2, 5]


def test_uncrowding_multi_offset_generation(tmp_path):
    """verify stimulus generation across multiple offsets."""
    output_dir = tmp_path / "multi_offset_test"
    offsets = [2, 5]
    config = UncrowdingShapesConfig(
        shapes=["square"],
        row_counts=[1],
        col_counts=[1],
        num_samples_vernier_inside=1,
        num_samples_vernier_outside=1,
        vernier_offset=offsets,
        output_folder=str(output_dir),
    )

    result_folder = generate_shapes(config)
    assert Path(result_folder).exists()

    csv_path = output_dir / "annotation.csv"
    assert csv_path.exists()

    with open(csv_path) as f:
        reader = list(csv.DictReader(f))
        logged_offsets = {int(row["VernierOffset"]) for row in reader}
        assert logged_offsets == {2, 5}
        for row in reader:
            offset_val = row["VernierOffset"]
            assert f"offset{offset_val}" in row["Path"]
            img_file = output_dir / row["Path"]
            assert img_file.exists()


def test_cli_parsing_vernier_offset():
    """verify CLI parser handles single and multiple vernier offset arguments."""
    from mindset.cli import _parse_generator_args

    args_single = _parse_generator_args(UncrowdingShapesConfig, ["--vernier-offset", "4"])
    assert args_single["vernier_offset"] == [4]

    args_multi = _parse_generator_args(UncrowdingShapesConfig, ["--vernier-offset", "2", "4", "8"])
    assert args_multi["vernier_offset"] == [2, 4, 8]


def test_uncrowding_distributions_central_flanker_fixed(monkeypatch, tmp_path):
    """verify that central flanker at (r // 2, c // 2) is strictly fixed to 0.0 rot and 1.0 scale."""
    captured_calls = []

    import mindset.generators.low_mid_vision.uncrowding as uncrowd_mod
    orig_generate_stimulus = uncrowd_mod.generate_uncrowding_stimulus

    def mock_generate_stimulus(**kwargs):
        captured_calls.append(
            {
                "rotations": kwargs["cell_rotations"],
                "scales": kwargs["cell_scales"],
            }
        )
        return orig_generate_stimulus(**kwargs)

    monkeypatch.setattr(uncrowd_mod, "generate_uncrowding_stimulus", mock_generate_stimulus)

    output_dir = tmp_path / "center_fixed_test"

    # Test rotation
    config_rot = UncrowdingDistributionsConfig(
        dimension="rotation",
        loc=[45.0],
        scale=[0.0],
        base_shapes=["square"],
        row_counts=[1, 3],
        col_counts=[1, 3],
        num_samples_vernier_inside=10,
        num_samples_vernier_outside=10,
        output_folder=str(output_dir / "rot"),
    )
    generate_distributions(config_rot)

    assert len(captured_calls) > 0
    for call in captured_calls:
        rots = call["rotations"]
        r = len(rots)
        c = len(rots[0])
        r_c, c_c = r // 2, c // 2
        assert rots[r_c][c_c] == 0.0, f"Central flanker rotated in {r}x{c} grid: {rots[r_c][c_c]}"
        if (r, c) == (3, 3):
            surround_rots = [rots[i][j] for i in range(r) for j in range(c) if (i, j) != (r_c, c_c)]
            assert any(rot == 45.0 for rot in surround_rots)

    captured_calls.clear()

    # Test scale
    config_scale = UncrowdingDistributionsConfig(
        dimension="scale",
        loc=[0.5, 0.5],
        scale=[0.0, 0.0],
        base_shapes=["square"],
        row_counts=[1, 3],
        col_counts=[1, 3],
        num_samples_vernier_inside=10,
        num_samples_vernier_outside=10,
        output_folder=str(output_dir / "scale"),
    )
    generate_distributions(config_scale)

    assert len(captured_calls) > 0
    for call in captured_calls:
        scales = call["scales"]
        r = len(scales)
        c = len(scales[0])
        r_c, c_c = r // 2, c // 2
        assert scales[r_c][c_c] == 1.0, f"Central flanker scaled in {r}x{c} grid: {scales[r_c][c_c]}"
        if (r, c) == (3, 3):
            surround_scales = [scales[i][j] for i in range(r) for j in range(c) if (i, j) != (r_c, c_c)]
            assert any((sc == 0.5 if sc != 'none' else True) for sc in surround_scales )
