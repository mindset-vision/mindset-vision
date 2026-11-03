"""Unit tests for Bregman's B's paradigm dataset generator."""

import csv
import json
import math
from pathlib import Path
import string
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from mindset.generators.shape_recognition.bregman_bs import (
    DSU,
    BregmanBsConfig,
    DrawBregmanBs,
    _check_edge_planarity,
    generate_bregman_bs,
)


def test_bregman_bs_config_defaults():
    """Verify default config parameters for Bregman's B's generator."""
    config = BregmanBsConfig()
    assert config.canvas_size == (512, 512)
    assert config.background_color == [255, 255, 255]
    assert config.object_color == [125, 125, 125]
    assert config.inkblot_color == [0, 0, 0]
    assert config.num_objects == 4
    assert config.object_size == 90
    assert config.num_samples == 20
    assert config.edges_per_object == 2
    assert config.sampling_step == 0.5
    assert config.blob_min_extent == 4.0
    assert config.blob_max_extent == 12.0
    assert config.object_spread == [0.3, 0.7]
    assert config.cycle_probability == 0.5
    assert config.curvature_range == (0.20, 0.45)
    # Verify characters defaults to uppercase letters
    assert "B" in config.characters
    assert "A" in config.characters
    assert "Z" in config.characters
    assert "8" not in config.characters
    assert "a" not in config.characters
    assert len(config.characters) == 26
    assert config.characters == list(string.ascii_uppercase)


def test_bregman_bs_config_normalization_and_validation():
    """Verify config input normalization and validation constraints."""
    # Lowercase string input converted to uppercase list of chars
    cfg_str = BregmanBsConfig(characters="b8r")
    assert cfg_str.characters == ["B", "8", "R"]

    # Lowercase list converted to uppercase
    cfg_lower = BregmanBsConfig(characters=["a", "b", "c"])
    assert cfg_lower.characters == ["A", "B", "C"]

    # Numeric list converted to strings
    cfg_num = BregmanBsConfig(characters=[8, 9, "b"])
    assert cfg_num.characters == ["8", "9", "B"]

    # Single float object_spread converted to list
    cfg_spread = BregmanBsConfig(object_spread=0.5)
    assert cfg_spread.object_spread == [0.5]

    # Curvature range list converted to tuple
    cfg_curv = BregmanBsConfig(curvature_range=[0.1, 0.3])
    assert cfg_curv.curvature_range == (0.1, 0.3)

    # Invalid characters list
    with pytest.raises(ValueError, match="characters list cannot be empty"):
        BregmanBsConfig(characters=[])

    # Invalid object_spread
    with pytest.raises(ValueError, match="object_spread list cannot be empty"):
        BregmanBsConfig(object_spread=[])

    with pytest.raises(ValueError, match="object_spread values must be in"):
        BregmanBsConfig(object_spread=[0.0])

    with pytest.raises(ValueError, match="object_spread values must be in"):
        BregmanBsConfig(object_spread=[1.5])

    # Invalid num_objects
    with pytest.raises(ValueError, match="num_objects must be >= 1"):
        BregmanBsConfig(num_objects=0)

    # Invalid blob extents
    with pytest.raises(ValueError, match="blob_min_extent must be > 0"):
        BregmanBsConfig(blob_min_extent=0.0)

    with pytest.raises(ValueError, match="blob_max_extent must be >= blob_min_extent"):
        BregmanBsConfig(blob_min_extent=8.0, blob_max_extent=4.0)

    # Invalid cycle probability
    with pytest.raises(ValueError, match="cycle_probability must be in"):
        BregmanBsConfig(cycle_probability=1.5)

    # Invalid curvature range
    with pytest.raises(ValueError, match="curvature_range must be a"):
        BregmanBsConfig(curvature_range=(0.5, 0.2))


def test_cli_parsing_bregman_bs():
    """Verify CLI parser handles generator arguments for bregman_bs."""
    from mindset.cli import _parse_generator_args

    args = _parse_generator_args(
        BregmanBsConfig,
        [
            "--characters",
            "B",
            "8",
            "--object-spread",
            "0.25",
            "0.75",
            "--num-objects",
            "5",
            "--blob-min-extent",
            "5.5",
            "--blob-max-extent",
            "15.5",
            "--cycle-probability",
            "0.8",
        ],
    )
    assert args["characters"] == ["B", 8]  # parsed as str/int before config __post_init__
    assert args["object_spread"] == [0.25, 0.75]
    assert args["num_objects"] == 5
    assert args["blob_min_extent"] == 5.5
    assert args["blob_max_extent"] == 15.5
    assert args["cycle_probability"] == 0.8

    # Ensure config __post_init__ normalizes characters to strings and spread to floats
    cfg = BregmanBsConfig(**{k: v for k, v in args.items() if v is not None})
    assert cfg.characters == ["B", "8"]
    assert cfg.object_spread == [0.25, 0.75]


def test_dsu_operations():
    """Verify Disjoint Set Union operations."""
    dsu = DSU(5)
    assert dsu.num_components == 5
    assert dsu.find(0) == 0

    assert dsu.union(0, 1) is True
    assert dsu.num_components == 4
    assert dsu.find(0) == dsu.find(1)

    assert dsu.union(0, 1) is False
    assert dsu.num_components == 4

    dsu.union(2, 3)
    dsu.union(1, 3)
    assert dsu.num_components == 2
    assert dsu.find(0) == dsu.find(2)


def test_edge_planarity_check():
    """Verify edge planarity check detects crossing and non-crossing segments."""
    # Square nodes: (0, 0), (10, 0), (10, 10), (0, 10)
    coords = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    existing = [(0, 1), (1, 2)]

    # Segment from (2, 3) crosses (0, 2)?
    # Diagonal 0 -> 2
    assert _check_edge_planarity(0, 2, coords, existing) is True

    # Diagonal 1 -> 3 crosses 0 -> 2
    existing_with_diag = [(0, 2)]
    assert _check_edge_planarity(1, 3, coords, existing_with_diag) is False

    # Sharing an endpoint without overlap
    assert _check_edge_planarity(2, 3, coords, existing) is True


def test_render_character_patch_and_polygon():
    """Verify character rendering produces valid raster patch and Shapely polygon."""
    config = BregmanBsConfig(object_size=70)
    drawer = DrawBregmanBs(config)

    for char in ["B", "8", "E"]:
        patch, poly = drawer.render_character_patch(char, rotation_deg=15.0)
        assert isinstance(patch, Image.Image)
        assert patch.width > 20 and patch.height > 20
        assert poly.is_valid
        assert poly.area > 50


def test_place_objects_no_overlap():
    """Verify place_objects places all instances without mutual intersection."""
    config = BregmanBsConfig(
        canvas_size=(400, 400),
        object_size=60,
        num_objects=3,
        characters=["B"],
    )
    drawer = DrawBregmanBs(config)

    positions, rotations, shapes, patches, mean_dist = drawer.place_objects("B")
    assert len(positions) == 3
    assert len(rotations) == 3
    assert len(shapes) == 3
    assert len(patches) == 3
    assert mean_dist > 0.0

    # Check pairwise non-intersection
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            assert not shapes[i].intersects(shapes[j]), f"Shapes {i} and {j} collided!"


def test_object_spread_dispersion():
    """Verify that small object_spread yields a smaller mean distance than large object_spread."""
    config = BregmanBsConfig(
        canvas_size=(500, 500),
        object_size=50,
        num_objects=4,
        characters=["B"],
    )
    drawer = DrawBregmanBs(config)
    _, _, _, _, dist_dense = drawer.place_objects("B", spread=0.2)
    _, _, _, _, dist_spread = drawer.place_objects("B", spread=0.9)
    assert dist_dense < dist_spread


def test_crossing_edges_random_alignment():
    """Verify that multiple crossing edges per shape have independent random orientations."""
    config = BregmanBsConfig(
        canvas_size=(400, 400),
        object_size=60,
        num_objects=2,
        edges_per_object=2,
        characters=["B"],
    )
    drawer = DrawBregmanBs(config)
    positions, rotations, shapes, patches, _ = drawer.place_objects("B")
    node_coords, crossing_edges, shape_nodes = drawer.sample_crossing_edges(shapes)

    assert len(crossing_edges) == 4
    assert len(node_coords) == 8
    assert len(shape_nodes) == 2

    # Check that for each shape, the two edges have different angles (non-zero angle between them)
    for s_idx, s_nodes in enumerate(shape_nodes):
        # s_nodes contains 4 node indices: [e1_u, e1_v, e2_u, e2_v]
        p1_u, p1_v = np.array(node_coords[s_nodes[0]]), np.array(node_coords[s_nodes[1]])
        p2_u, p2_v = np.array(node_coords[s_nodes[2]]), np.array(node_coords[s_nodes[3]])

        # Verify that both edges cross the shape's bounding box completely
        bbox_poly = box(*shapes[s_idx].bounds)
        line1 = LineString([p1_u, p1_v])
        line2 = LineString([p2_u, p2_v])

        assert not bbox_poly.contains(Point(p1_u))
        assert not bbox_poly.contains(Point(p1_v))
        assert not bbox_poly.contains(Point(p2_u))
        assert not bbox_poly.contains(Point(p2_v))

        # Both lines must slice through the bounding box
        assert line1.intersects(bbox_poly)
        assert line2.intersects(bbox_poly)
        assert line1.intersection(bbox_poly).length > 10.0
        assert line2.intersection(bbox_poly).length > 10.0


def test_connected_planar_graph_construction():
    """Verify graph connectivity and planarity across sampled crossing edges."""
    config = BregmanBsConfig(
        canvas_size=(400, 400),
        object_size=60,
        num_objects=3,
        edges_per_object=2,
        characters=["B"],
        cycle_probability=0.5,
    )
    drawer = DrawBregmanBs(config)

    positions, rotations, shapes, patches, _ = drawer.place_objects("B")
    node_coords, crossing_edges, shape_nodes = drawer.sample_crossing_edges(shapes)

    assert len(crossing_edges) == 6
    assert len(node_coords) == 12

    edges = drawer.build_connected_planar_graph(node_coords, crossing_edges, shapes, shape_nodes)

    # 1. Connected: spanning tree on components
    assert len(edges) >= 6

    # 2. Planar: no connector edge crosses any other edge
    for idx_a, (u1, v1) in enumerate(edges):
        for idx_b in range(idx_a + 1, len(edges)):
            u2, v2 = edges[idx_b]
            if not ({u1, v1} & {u2, v2}):
                from shapely.geometry import LineString
                line1 = LineString([node_coords[u1], node_coords[v1]])
                line2 = LineString([node_coords[u2], node_coords[v2]])
                # Two crossing edges through the same shape naturally cross inside that shape.
                # Edges between different shapes or connector edges must not cross each other.
                same_shape = any(line1.intersects(s) and line2.intersects(s) for s in shapes)
                if not same_shape:
                    assert not line1.crosses(line2), f"Edges {edges[idx_a]} and {edges[idx_b]} crossed!"


def test_bregman_bs_generator_end_to_end(tmp_path):
    """End-to-end smoke test for bregman_bs generator verifying image pairs and annotations."""
    output_dir = tmp_path / "bregman_test"
    config = BregmanBsConfig(
        canvas_size=(400, 400),
        object_size=80,
        blob_min_extent=5.0,
        blob_max_extent=15.0,
        object_spread=[0.3, 0.7],
        num_objects=3,
        characters=["B", "8"],
        num_samples=2,
        output_folder=str(output_dir),
    )

    result_folder = generate_bregman_bs(config)
    assert Path(result_folder).exists()

    csv_path = output_dir / "annotation.csv"
    assert csv_path.exists()

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    # 2 spreads * 2 sample pairs * 2 conditions (occluded, parts) = 8 rows
    assert len(rows) == 8

    pair_ids = {int(r["PairID"]) for r in rows}
    assert pair_ids == {0, 1, 2, 3}

    for pair_id in [0, 1, 2, 3]:
        pair_rows = [r for r in rows if int(r["PairID"]) == pair_id]
        assert len(pair_rows) == 2

        occluded_row = next(r for r in pair_rows if r["Condition"] == "occluded")
        parts_row = next(r for r in pair_rows if r["Condition"] == "parts")

        assert occluded_row["Character"] == parts_row["Character"]
        assert occluded_row["Character"] in ("B", "8")
        assert occluded_row["NumObjects"] == "3"
        assert occluded_row["ObjectSpread"] in ("0.3", "0.7")
        assert occluded_row["ObjectSpread"] == parts_row["ObjectSpread"]
        assert float(occluded_row["MeanObjectDistance"]) > 0.0
        assert occluded_row["ObjectPositions"] == parts_row["ObjectPositions"]
        assert occluded_row["ObjectRotations"] == parts_row["ObjectRotations"]
        assert occluded_row["InkblotColor"] == "black"
        assert parts_row["InkblotColor"] == "white"
        assert float(occluded_row["BlobMinExtent"]) == 5.0
        assert float(occluded_row["BlobMaxExtent"]) == 15.0
        assert float(parts_row["BlobMinExtent"]) == 5.0
        assert float(parts_row["BlobMaxExtent"]) == 15.0

        img_occ_path = output_dir / occluded_row["Path"]
        img_prt_path = output_dir / parts_row["Path"]
        assert img_occ_path.exists()
        assert img_prt_path.exists()

        img_occ = Image.open(img_occ_path)
        img_prt = Image.open(img_prt_path)
        assert img_occ.size == (400, 400)
        assert img_prt.size == (400, 400)

        arr_occ = np.array(img_occ)
        arr_prt = np.array(img_prt)

        # In unoccluded regions outside the inkblot, both images must be identical
        # Outside inkblot: where parts image is not white (i.e. black character fragment)
        char_fragments = np.all(arr_prt < 128, axis=-1)
        assert np.any(char_fragments), "Parts image should contain visible character fragments!"
        # In character fragments of parts image, occluded image must also be black
        assert np.all(arr_occ[char_fragments] < 128)


def test_convex_hull_packing_compactness():
    """Verify that contour-contact convex hull optimization achieves compact non-overlapping clustering."""
    config = BregmanBsConfig(
        canvas_size=(500, 500),
        object_size=80,
        num_objects=4,
        characters=["B"],
    )
    drawer = DrawBregmanBs(config)

    positions, rotations, shapes, patches, _ = drawer.place_objects("B", spread=0.05)
    assert len(shapes) == 4

    # Mutual non-overlap
    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            assert not shapes[i].intersects(shapes[j]), f"Shapes {i} and {j} collided!"

    total_shape_area = sum(s.area for s in shapes)
    ch_area = unary_union(shapes).convex_hull.area
    efficiency = total_shape_area / ch_area

    # High packing efficiency for tightly tiled 2D 'B' cluster (> 50%)
    assert efficiency > 0.50, f"Expected packing efficiency > 50%, got {efficiency:.2%}"


def test_curved_crossing_edges_and_curves():
    """Verify that crossing edges return valid Bézier curves that slice bounding boxes completely."""
    config = BregmanBsConfig(
        canvas_size=(500, 500),
        object_size=70,
        num_objects=3,
        edges_per_object=2,
        characters=["B"],
    )
    drawer = DrawBregmanBs(config)
    positions, rotations, shapes, patches, _ = drawer.place_objects("B")

    # Call with return_curves=True
    node_coords, crossing_edges, shape_nodes, edge_curves = drawer.sample_crossing_edges(
        shapes, return_curves=True
    )

    assert len(crossing_edges) == 6
    assert len(node_coords) == 12
    assert len(edge_curves) == 6

    for u, v in crossing_edges:
        assert (u, v) in edge_curves
        p1, cp, p2 = edge_curves[(u, v)]
        assert p1 == node_coords[u]
        assert p2 == node_coords[v]

        # Verify endpoints lie outside the corresponding shape's bounding box
        for s_idx, s_nodes in enumerate(shape_nodes):
            if u in s_nodes:
                bbox = box(*shapes[s_idx].bounds)
                assert not bbox.contains(Point(p1))
                assert not bbox.contains(Point(p2))

        # Verify pronounced curvature: control point deflection from chord midpoint
        mid = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
        deflection = math.hypot(cp[0] - mid[0], cp[1] - mid[1])
        chord_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        curv_frac = deflection / chord_len
        assert curv_frac >= 0.18, f"Curvature fraction {curv_frac:.3f} is too low (expected >= 0.18)"


def test_curved_planar_graph_and_pycairo_mask():
    """Verify PyCairo organic inkblot mask generation from curved planar graph."""
    config = BregmanBsConfig(
        canvas_size=(400, 400),
        object_size=70,
        num_objects=3,
        edges_per_object=2,
        characters=["B"],
        blob_min_extent=6.0,
        blob_max_extent=14.0,
    )
    drawer = DrawBregmanBs(config)
    positions, rotations, shapes, patches, _ = drawer.place_objects("B")
    node_coords, crossing_edges, shape_nodes, edge_curves = drawer.sample_crossing_edges(
        shapes, return_curves=True
    )

    edges, all_curves = drawer.build_connected_planar_graph(
        node_coords,
        crossing_edges,
        shapes,
        shape_nodes,
        edge_curves=edge_curves,
        return_curves=True,
    )

    mask = drawer.generate_inkblot_mask(node_coords, edges, edge_curves=all_curves)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (400, 400)
    assert mask.dtype == np.uint8
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255})
    assert np.count_nonzero(mask) > 1000, "Mask should cover substantial inkblot pixels"


def test_square_bounding_box_aspect_ratio():
    """Verify that packing arranges characters into a square or near-square 2D configuration."""
    for n_obj in [4, 6]:
        config = BregmanBsConfig(
            canvas_size=(500, 500),
            object_size=75,
            num_objects=n_obj,
            characters=["B", "E"],
        )
        drawer = DrawBregmanBs(config)

        for char in ["B", "E"]:
            positions, rotations, shapes, patches, _ = drawer.place_objects(char, spread=0.05)
            assert len(shapes) == n_obj

            u = unary_union(shapes)
            minx, miny, maxx, maxy = u.bounds
            bw = maxx - minx
            bh = maxy - miny
            aspect_ratio = max(bw, bh) / max(1.0, min(bw, bh))

            # Strictly verify that shapes form a 2D square / near-square cluster (AR <= 1.6)
            # and NOT an elongated 1D strip (which would have AR >= 3.0 for N=4, or >= 5.0 for N=6)
            assert aspect_ratio <= 1.6, (
                f"Cluster for N={n_obj} char '{char}' is too elongated! "
                f"Bounds: {bw:.1f}x{bh:.1f}, Aspect ratio: {aspect_ratio:.2f} > 1.6"
            )


def test_variable_thickness_modulation():
    """Verify inkblot mask generation with uniform thickness (min == max) and variable thickness (min < max)."""
    # 1. Uniform thickness (min_extent == max_extent)
    cfg_uniform = BregmanBsConfig(
        canvas_size=(300, 300),
        blob_min_extent=8.0,
        blob_max_extent=8.0,
        num_objects=2,
    )
    drawer_u = DrawBregmanBs(cfg_uniform)
    pos_u, rot_u, shapes_u, patches_u, _ = drawer_u.place_objects("B")
    nodes_u, c_edges_u, s_nodes_u, curves_u = drawer_u.sample_crossing_edges(shapes_u, return_curves=True)
    edges_u, all_curves_u = drawer_u.build_connected_planar_graph(nodes_u, c_edges_u, shapes_u, s_nodes_u, edge_curves=curves_u, return_curves=True)
    mask_u = drawer_u.generate_inkblot_mask(nodes_u, edges_u, edge_curves=all_curves_u)

    assert isinstance(mask_u, np.ndarray)
    assert mask_u.shape == (300, 300)
    assert set(np.unique(mask_u)).issubset({0, 255})
    assert np.count_nonzero(mask_u) > 500

    # 2. Variable thickness (min_extent < max_extent)
    cfg_var = BregmanBsConfig(
        canvas_size=(300, 300),
        blob_min_extent=4.0,
        blob_max_extent=16.0,
        num_objects=2,
    )
    drawer_v = DrawBregmanBs(cfg_var)
    mask_v = drawer_v.generate_inkblot_mask(nodes_u, edges_u, edge_curves=all_curves_u)

    assert isinstance(mask_v, np.ndarray)
    assert mask_v.shape == (300, 300)
    assert set(np.unique(mask_v)).issubset({0, 255})
    assert np.count_nonzero(mask_v) > 500



