"""Bregman's B's paradigm dataset generator.

This generator extends the classic Bregman (1981) visual occlusion paradigm.
Instances of a character (e.g. letters or digits) are placed across the canvas,
and a procedurally generated connected planar inkblot is constructed that crosses
and occludes the shapes. For each trial, two matched stimuli are produced:
1. 'occluded': The inkblot is solid black, visually occluding the shapes and
   enabling amodal completion.
2. 'parts': The inkblot matches the white background, carving away the occluded
   regions and leaving only disconnected fragments without an explicit occluder.
"""

import cairo
import csv
import json
import math
import random
import string
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.ndimage
from shapely.affinity import translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union
from tqdm.auto import tqdm

from mindset.generators._base import GeneratorConfig, generator, register


class DSU:
    """Disjoint Set Union (Union-Find) with path compression for connected components."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.num_components = n

    def find(self, i: int) -> int:
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int) -> bool:
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j
            self.num_components -= 1
            return True
        return False


def sample_quadratic_bezier(
    p1: tuple[float, float],
    p2: tuple[float, float],
    cp: tuple[float, float],
    num_pts: int = 15,
) -> list[tuple[float, float]]:
    """Sample points along a quadratic Bézier curve from p1 to p2 via control point cp."""
    p1_a = np.array(p1, dtype=np.float64)
    p2_a = np.array(p2, dtype=np.float64)
    cp_a = np.array(cp, dtype=np.float64)
    t = np.linspace(0.0, 1.0, num_pts)[:, None]
    curve = (1.0 - t) ** 2 * p1_a + 2.0 * (1.0 - t) * t * cp_a + t ** 2 * p2_a
    return [tuple(pt) for pt in curve]


def _check_edge_planarity(
    u: int,
    v: int,
    node_coords: list[tuple[float, float]],
    existing_edges: list[tuple[int, int]],
    cand_line: LineString | None = None,
    existing_lines: list[LineString] | None = None,
) -> bool:
    """Check if straight or curved segment (u, v) can be added to existing_edges without crossing any edge."""
    p_u = node_coords[u]
    p_v = node_coords[v]
    if cand_line is None:
        cand_line = LineString([p_u, p_v])

    # Check that candidate edge does not pass directly through another node
    for w_idx, p_w in enumerate(node_coords):
        if w_idx not in (u, v):
            if Point(p_w).distance(cand_line) < 1.0:
                return False

    for idx, (e_a, e_b) in enumerate(existing_edges):
        if (e_a, e_b) == (u, v) or (e_a, e_b) == (v, u):
            return False  # Already exists

        if existing_lines is not None and idx < len(existing_lines):
            edge_line = existing_lines[idx]
        else:
            p_a = node_coords[e_a]
            p_b = node_coords[e_b]
            edge_line = LineString([p_a, p_b])

        shared_nodes = {u, v} & {e_a, e_b}
        if not shared_nodes:
            if cand_line.intersects(edge_line):
                return False
        else:
            # Edges share an endpoint: they must not overlap collinearly
            inter = cand_line.intersection(edge_line)
            if inter.geom_type != "Point":
                return False

    return True


@dataclass
class BregmanBsConfig(GeneratorConfig):
    """config for Bregman's B's paradigm dataset generator."""

    canvas_size: tuple[int, int] = field(
        default_factory=lambda: (512, 512),
        metadata={"min": 128, "max": 1024, "step": 32, "label": "canvas size (width, height)"},
    )
    background_color: list[int] = field(
        default_factory=lambda: [255, 255, 255],
        metadata={"label": "background color (RGB)"},
    )
    object_color: list[int] = field(
        default_factory=lambda: [125, 125, 125],
        metadata={"label": "object color (RGB)"},
    )
    inkblot_color: list[int] = field(
        default_factory=lambda: [0, 0, 0],
        metadata={"label": "inkblot color when occluding (RGB)"},
    )
    characters: list[str] = field(
        default_factory=lambda: list(string.ascii_uppercase),
        metadata={"label": "pool of uppercase letter characters to sample from"},
    )
    num_objects: int = field(
        default=4,
        metadata={"min": 1, "max": 12, "label": "number of character instances per image"},
    )
    object_size: int = field(
        default=90,
        metadata={"min": 30, "max": 250, "label": "font size for character objects"},
    )
    num_samples: int = field(
        default=20,
        metadata={"min": 1, "max": 10000, "label": "number of stimulus pairs to generate"},
    )
    edges_per_object: int = field(
        default=2,
        metadata={"min": 1, "max": 4, "label": "number of crossing edges per object"},
    )
    sampling_step: float = field(
        default=0.5,
        metadata={"min": 0.1, "max": 5.0, "label": "sampling step size along edges (px) for Gaussian blob rendering"},
    )
    blob_min_extent: float = field(
        default=4.0,
        metadata={"min": 1.0, "max": 20.0, "label": "minimum radius of inkblot extent (px)"},
    )
    blob_max_extent: float = field(
        default=12.0,
        metadata={"min": 1.0, "max": 30.0, "label": "maximum radius of inkblot extent (px)"},
    )
    object_spread: list[float] = field(
        default_factory=lambda: [0.3, 0.7],
        metadata={"label": "list of spatial spread factors for objects (e.g. 0.3 0.7)"},
    )
    cycle_probability: float = field(
        default=0.5,
        metadata={"min": 0.0, "max": 1.0, "label": "probability of allowing cycles in inkblot graph"},
    )
    max_extra_edges: int = field(
        default=2,
        metadata={"min": 0, "max": 10, "label": "maximum additional cycle edges"},
    )
    min_rotation: float = field(
        default=-30.0,
        metadata={"label": "minimum object rotation in degrees"},
    )
    max_rotation: float = field(
        default=45.0,
        metadata={"label": "maximum object rotation in degrees"},
    )
    curvature_range: tuple[float, float] = field(
        default=(0.20, 0.45),
        metadata={"label": "range of Bézier curve sagitta/deflection as fraction of chord length"},
    )
    font_path: str | None = field(
        default=None,
        metadata={"label": "path to TTF font file (defaults to assets/arial.ttf)"},
    )
    output_folder: str = field(
        default="data/shape_recognition/bregman_bs",
        metadata={"label": "output folder"},
    )

    def __post_init__(self):
        if isinstance(self.characters, str):
            self.characters = list(self.characters.upper())
        else:
            self.characters = [str(c).upper() for c in self.characters]

        if isinstance(self.object_spread, (int, float)):
            self.object_spread = [float(self.object_spread)]
        else:
            self.object_spread = [float(s) for s in self.object_spread]

        if isinstance(self.curvature_range, list):
            self.curvature_range = tuple(self.curvature_range)

        if not self.characters:
            raise ValueError("characters list cannot be empty")
        if not self.object_spread:
            raise ValueError("object_spread list cannot be empty")
        for s in self.object_spread:
            if not (0.0 < s <= 1.0):
                raise ValueError(f"object_spread values must be in (0, 1], got {s}")
        if self.num_objects < 1:
            raise ValueError(f"num_objects must be >= 1, got {self.num_objects}")
        if self.blob_min_extent <= 0:
            raise ValueError(f"blob_min_extent must be > 0, got {self.blob_min_extent}")
        if self.blob_max_extent < self.blob_min_extent:
            raise ValueError(
                f"blob_max_extent must be >= blob_min_extent, got {self.blob_max_extent} < {self.blob_min_extent}"
            )
        if not (0.0 <= self.cycle_probability <= 1.0):
            raise ValueError(f"cycle_probability must be in [0, 1], got {self.cycle_probability}")
        if len(self.curvature_range) != 2 or not (0.0 <= self.curvature_range[0] <= self.curvature_range[1]):
            raise ValueError(f"curvature_range must be a (min, max) tuple with 0 <= min <= max, got {self.curvature_range}")


class DrawBregmanBs:
    """Renders Bregman's B's stimuli with procedurally generated planar inkblots."""

    def __init__(self, config: BregmanBsConfig):
        self.config = config
        if config.font_path is None:
            self.font_path = (
                Path(__file__).resolve().parents[2] / "assets" / "arial.ttf"
            )
        else:
            self.font_path = Path(config.font_path)

        if not self.font_path.exists():
            raise FileNotFoundError(f"font file not found at {self.font_path}")

        self.font = ImageFont.truetype(str(self.font_path), size=config.object_size)

    def render_character_patch(
        self, char: str, rotation_deg: float
    ) -> tuple[Image.Image, Polygon | MultiPolygon]:
        """Render a single character to a rotated PIL Image patch and extract its vector outline polygon."""
        bbox = self.font.getbbox(char)
        w = max(1, bbox[2] - bbox[0])
        h = max(1, bbox[3] - bbox[1])

        pad = int(max(w, h) * 0.6) + 16
        img_patch = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)
        draw = ImageDraw.Draw(img_patch)
        draw.text((pad - bbox[0], pad - bbox[1]), char, font=self.font, fill=255)

        if rotation_deg != 0.0:
            img_rot = img_patch.rotate(
                rotation_deg, resample=Image.Resampling.BILINEAR, expand=True
            )
        else:
            img_rot = img_patch

        arr = np.array(img_rot) > 100
        contours, _ = cv2.findContours(
            arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        polys = []
        for cnt in contours:
            if len(cnt) >= 3:
                pts = cnt.squeeze(axis=1)
                if len(pts) >= 3:
                    poly = Polygon(pts)
                    if poly.is_valid and poly.area > 15:
                        polys.append(poly)

        if not polys:
            # Fallback to non-empty bbox polygon if contour extraction fails
            polys = [Polygon([(0, 0), (w, 0), (w, h), (0, h)])]

        shape_poly = unary_union(polys)
        return img_rot, shape_poly

    def place_objects(
        self, char: str, spread: float = 0.6, max_attempts: int = 150
    ) -> tuple[list[tuple[int, int]], list[float], list[Polygon | MultiPolygon], list[Image.Image], float]:
        """Sample collision-free locations and orientations for num_objects instances of char.

        Lays out shapes by contour-contact guided convex hull minimization to pack the central
        space as tightly as possible, and scales the relative configuration by the spread factor.
        """
        w_canvas, h_canvas = self.config.canvas_size
        margin = max(15, int(self.config.object_size * 0.25))
        s = self.config.object_size
        n = self.config.num_objects

        # 1. Place initial shape at center with random rotation
        rot0 = random.uniform(self.config.min_rotation, self.config.max_rotation)
        patch0, poly0 = self.render_character_patch(char, rot0)
        c0 = poly0.centroid
        poly0_centered = translate(poly0, -c0.x, -c0.y)

        placed_cluster_polys = [poly0_centered]
        placed_patches = [patch0]
        placed_orig_polys = [poly0]
        placed_rotations = [rot0]
        local_centroids = [np.array([c0.x, c0.y])]
        placed_centroids = [np.array([0.0, 0.0])]

        clearance = 4.0
        probe_angles = np.linspace(0, 2 * math.pi, 24, endpoint=False)

        # 2. Sequentially place shapes 1 ... n-1 into a compact square / near-square 2D cluster
        for _ in range(1, n):
            u = unary_union(placed_cluster_polys)
            u_buf = u.buffer(clearance)
            cx, cy = u.centroid.x, u.centroid.y
            min_x, min_y, max_x, max_y = u.bounds
            max_r = math.hypot(max_x - min_x, max_y - min_y) + s * 1.5

            best_score = float("inf")
            best_poly_centered = None
            best_orig_poly = None
            best_patch = None
            best_rot = 0.0
            best_lc = None
            best_pos = np.array([0.0, 0.0])

            # Sample diverse candidate rotations for each shape
            rot_candidates = [
                random.uniform(self.config.min_rotation, self.config.max_rotation)
                for _ in range(5)
            ] + [0.0]

            for rot in rot_candidates:
                patch, poly = self.render_character_patch(char, rot)
                pc = poly.centroid
                p_c = translate(poly, -pc.x, -pc.y)

                for ang in probe_angles:
                    dx, dy = math.cos(ang), math.sin(ang)
                    # March inward from max_r toward 0 until contact with clearance buffer
                    r = max_r
                    last_valid_r = None
                    step = 3.0
                    while r >= 0:
                        cand = translate(p_c, cx + r * dx, cy + r * dy)
                        if cand.intersects(u_buf):
                            break
                        last_valid_r = r
                        r -= step

                    if last_valid_r is not None:
                        cand = translate(p_c, cx + last_valid_r * dx, cy + last_valid_r * dy)
                        u_cand = unary_union([u, cand])
                        c_minx, c_miny, c_maxx, c_maxy = u_cand.bounds
                        bw = max(1.0, c_maxx - c_minx)
                        bh = max(1.0, c_maxy - c_miny)
                        ar = max(bw, bh) / min(bw, bh)
                        sq_area = max(bw, bh) ** 2
                        ch_area = u_cand.convex_hull.area
                        # Heavily penalize aspect ratio elongation to force square / near-square layout
                        score = sq_area * (ar ** 1.5) + 0.5 * ch_area
                        if score < best_score:
                            best_score = score
                            best_poly_centered = cand
                            best_orig_poly = poly
                            best_patch = patch
                            best_rot = rot
                            best_lc = np.array([pc.x, pc.y])
                            best_pos = np.array([cx + last_valid_r * dx, cy + last_valid_r * dy])

            if best_poly_centered is None:
                best_poly_centered = translate(p_c, cx + (max_r * 0.5), cy)
                best_orig_poly = poly
                best_patch = patch
                best_rot = rot_candidates[0]
                best_lc = np.array([pc.x, pc.y])
                best_pos = np.array([cx + (max_r * 0.5), cy])

            placed_cluster_polys.append(best_poly_centered)
            placed_patches.append(best_patch)
            placed_orig_polys.append(best_orig_poly)
            placed_rotations.append(best_rot)
            local_centroids.append(best_lc)
            placed_centroids.append(best_pos)

        # 3. Center cluster and scale relative positions according to spread factor
        cluster_mean = np.mean(placed_centroids, axis=0)
        rel_c = [c - cluster_mean for c in placed_centroids]
        cluster_r = float(np.max([np.linalg.norm(c) for c in rel_c])) if n > 1 else 1.0

        max_canvas_r = max(cluster_r + 10.0, min(w_canvas, h_canvas) / 2.0 - margin - s * 0.6)
        max_scale = max(1.0, max_canvas_r / max(1.0, cluster_r))
        spread_c = max(0.0, min(1.0, float(spread)))
        scale = 1.0 + (max_scale - 1.0) * spread_c
        scaled_c = [c * scale for c in rel_c]

        canvas_cx = w_canvas / 2.0
        canvas_cy = h_canvas / 2.0

        positions: list[tuple[int, int]] = []
        placed_polygons: list[Polygon | MultiPolygon] = []

        for i in range(n):
            pw, ph = placed_patches[i].size
            target_cx = canvas_cx + scaled_c[i][0]
            target_cy = canvas_cy + scaled_c[i][1]
            x = int(round(target_cx - local_centroids[i][0]))
            y = int(round(target_cy - local_centroids[i][1]))
            x = max(margin, min(w_canvas - margin - pw, x))
            y = max(margin, min(h_canvas - margin - ph, y))
            positions.append((x, y))
            placed_polygons.append(translate(placed_orig_polys[i], xoff=x, yoff=y))

        poly_centroids = [(p.centroid.x, p.centroid.y) for p in placed_polygons]
        pair_dists = [
            math.hypot(c1[0] - c2[0], c1[1] - c2[1])
            for i, c1 in enumerate(poly_centroids)
            for c2 in poly_centroids[i + 1:]
        ]
        mean_dist = float(np.mean(pair_dists)) if len(pair_dists) > 0 else 0.0

        return positions, placed_rotations, placed_polygons, placed_patches, mean_dist

    def sample_crossing_edges(
        self,
        shapes: list[Polygon | MultiPolygon],
        max_attempts_per_edge: int = 150,
        return_curves: bool = False,
    ) -> tuple:
        """For each shape, sample curved crossing edge(s) slicing completely through its bounding box.

        Endpoints strictly reside outside the shape's bounding box and outside all character strokes.
        Returns node coordinates, crossing edges, shape node groupings, and optionally Bézier curve parameters.
        """
        w_canvas, h_canvas = self.config.canvas_size
        diag = math.hypot(w_canvas, h_canvas)
        node_coords: list[tuple[float, float]] = []
        crossing_edges: list[tuple[int, int]] = []
        shape_node_indices: list[list[int]] = []
        edge_curves: dict[tuple[int, int], tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = {}

        for shape_idx, shape in enumerate(shapes):
            minx, miny, maxx, maxy = shape.bounds
            bbox_poly = box(minx, miny, maxx, maxy)
            bw = maxx - minx
            bh = maxy - miny
            bbox_cx = (minx + maxx) / 2.0
            bbox_cy = (miny + maxy) / 2.0
            r_bound = max(15.0, math.hypot(bw, bh) * 0.5)

            edges_sampled = 0
            angles_sampled: list[float] = []
            curr_shape_nodes: list[int] = []

            for _ in range(max_attempts_per_edge):
                angle = random.uniform(0, math.pi)
                # Ensure minimum angular separation between crossing edges on the same shape
                if any(min(abs(angle - a), math.pi - abs(angle - a)) < math.radians(20) for a in angles_sampled):
                    continue

                dx = math.cos(angle)
                dy = math.sin(angle)
                nx = -dy
                ny = dx

                offset = random.uniform(-0.25 * min(bw, bh), 0.25 * min(bw, bh))
                mid_x = bbox_cx + offset * nx
                mid_y = bbox_cy + offset * ny

                # Long line through mid_x, mid_y across the canvas
                long_line = LineString([
                    (mid_x - diag * dx, mid_y - diag * dy),
                    (mid_x + diag * dx, mid_y + diag * dy),
                ])

                # Intersect with the shape's bounding box to guarantee crossing it completely
                bbox_inter = long_line.intersection(bbox_poly)
                if bbox_inter.is_empty or bbox_inter.geom_type != "LineString":
                    continue

                # Collect boundary intersection points of the bounding box
                b_pts = list(bbox_inter.coords)
                t_vals = [(pt[0] - mid_x) * dx + (pt[1] - mid_y) * dy for pt in b_pts]
                t_min = min(t_vals)
                t_max = max(t_vals)

                # Endpoints extend outside the bounding box
                pad1 = random.uniform(4.0, 8.0)
                pad2 = random.uniform(4.0, 8.0)
                p1 = (mid_x + (t_min - pad1) * dx, mid_y + (t_min - pad1) * dy)
                p2 = (mid_x + (t_max + pad2) * dx, mid_y + (t_max + pad2) * dy)

                # Check canvas bounds
                if not (10 <= p1[0] <= w_canvas - 10 and 10 <= p1[1] <= h_canvas - 10):
                    continue
                if not (10 <= p2[0] <= w_canvas - 10 and 10 <= p2[1] <= h_canvas - 10):
                    continue

                # Endpoints must lie outside all shapes
                pt1 = Point(p1)
                pt2 = Point(p2)
                if any(s.contains(pt1) or s.contains(pt2) for s in shapes):
                    continue

                # Sample pronounced curvature for quadratic Bézier curve
                chord_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
                min_c, max_c = self.config.curvature_range
                curv_mag = random.uniform(min_c, max_c) * chord_len
                sign = random.choice([-1.0, 1.0])
                curv = sign * curv_mag
                cp = ((p1[0] + p2[0]) / 2.0 + curv * nx, (p1[1] + p2[1]) / 2.0 + curv * ny)

                pts_bezier = sample_quadratic_bezier(p1, p2, cp, num_pts=20)
                curve_line = LineString(pts_bezier)

                # Verify that the curved line cuts through actual character strokes
                stroke_inter = curve_line.intersection(shape)
                if stroke_inter.is_empty or stroke_inter.length < 0.12 * r_bound:
                    continue

                # Line must NOT intersect any other shape
                intersects_other = False
                for other_idx, other_shape in enumerate(shapes):
                    if other_idx != shape_idx and curve_line.intersects(other_shape):
                        intersects_other = True
                        break
                if intersects_other:
                    continue

                # Valid crossing edge found
                idx1 = len(node_coords)
                idx2 = idx1 + 1
                node_coords.append(p1)
                node_coords.append(p2)
                crossing_edges.append((idx1, idx2))
                edge_curves[(idx1, idx2)] = (p1, cp, p2)
                curr_shape_nodes.extend([idx1, idx2])
                angles_sampled.append(angle)
                edges_sampled += 1

                if edges_sampled >= self.config.edges_per_object:
                    break

            if edges_sampled == 0:
                raise RuntimeError("Failed to sample crossing edge for a shape.")

            shape_node_indices.append(curr_shape_nodes)

        if return_curves:
            return node_coords, crossing_edges, shape_node_indices, edge_curves
        return node_coords, crossing_edges, shape_node_indices

    def build_connected_planar_graph(
        self,
        node_coords: list[tuple[float, float]],
        crossing_edges: list[tuple[int, int]],
        shapes: list[Polygon | MultiPolygon],
        shape_node_indices: list[list[int]] | None = None,
        edge_curves: dict[tuple[int, int], tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] | None = None,
        return_curves: bool = False,
    ) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], dict[tuple[int, int], tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]]:
        """Connect crossing edges into a single connected planar graph with optional cycles and smooth curved arcs."""
        num_nodes = len(node_coords)
        dsu = DSU(num_nodes)
        for u, v in crossing_edges:
            dsu.union(u, v)

        if shape_node_indices:
            for s_nodes in shape_node_indices:
                for idx in s_nodes[1:]:
                    dsu.union(s_nodes[0], idx)

        shapes_union = unary_union(shapes).buffer(1.5)
        edges = list(crossing_edges)
        all_curves = dict(edge_curves) if edge_curves is not None else {}

        # Precompute initial edge LineStrings for planarity checking
        all_lines: list[LineString] = []
        for u, v in crossing_edges:
            if (u, v) in all_curves:
                p1, cp, p2 = all_curves[(u, v)]
                all_lines.append(LineString(sample_quadratic_bezier(p1, p2, cp)))
            else:
                all_lines.append(LineString([node_coords[u], node_coords[v]]))

        # 1. Gather all potential connector pairs (u, v) with candidate Bézier curves
        candidate_edges: list[tuple[float, int, int, tuple[float, float], LineString]] = []
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if (u, v) in edges or (v, u) in edges:
                    continue
                p_u = node_coords[u]
                p_v = node_coords[v]
                dx = p_v[0] - p_u[0]
                dy = p_v[1] - p_u[1]
                chord_len = math.hypot(dx, dy)
                if chord_len < 1e-4:
                    continue

                nx = -dy / chord_len
                ny = dx / chord_len

                # Test curvatures to navigate around shape boundaries, prioritizing pronounced curved arcs
                best_cand = None
                for c_frac in [0.35, -0.35, 0.25, -0.25, 0.45, -0.45, 0.15, -0.15, 0.0]:
                    cp = (
                        (p_u[0] + p_v[0]) / 2.0 + c_frac * chord_len * nx,
                        (p_u[1] + p_v[1]) / 2.0 + c_frac * chord_len * ny,
                    )
                    pts = sample_quadratic_bezier(p_u, p_v, cp, num_pts=20)
                    line = LineString(pts)

                    # Connector cannot intersect any shape
                    if not line.intersects(shapes_union):
                        best_cand = (chord_len, u, v, cp, line)
                        break

                if best_cand is not None:
                    candidate_edges.append(best_cand)

        # Sort candidate edges by chord length (shortest first)
        candidate_edges.sort(key=lambda item: item[0])

        # 2. Add edges to connect all components into a single connected graph
        for chord_len, u, v, cp, line in candidate_edges:
            if dsu.find(u) != dsu.find(v):
                if _check_edge_planarity(u, v, node_coords, edges, cand_line=line, existing_lines=all_lines):
                    edges.append((u, v))
                    all_lines.append(line)
                    all_curves[(u, v)] = (node_coords[u], cp, node_coords[v])
                    dsu.union(u, v)
                    if dsu.num_components == 1:
                        break

        if dsu.num_components > 1:
            raise RuntimeError("Could not connect all inkblot components into a single planar graph.")

        # 3. Sample additional edges to allow cycles with cycle_probability
        remaining_candidates = [
            (chord_len, u, v, cp, line)
            for chord_len, u, v, cp, line in candidate_edges
            if (u, v) not in edges and (v, u) not in edges
        ]
        random.shuffle(remaining_candidates)

        cycles_added = 0
        for chord_len, u, v, cp, line in remaining_candidates:
            if cycles_added >= self.config.max_extra_edges:
                break
            if random.random() < self.config.cycle_probability:
                if _check_edge_planarity(u, v, node_coords, edges, cand_line=line, existing_lines=all_lines):
                    edges.append((u, v))
                    all_lines.append(line)
                    all_curves[(u, v)] = (node_coords[u], cp, node_coords[v])
                    cycles_added += 1

        if return_curves:
            return edges, all_curves
        return edges

    def generate_inkblot_mask(
        self,
        node_coords: list[tuple[float, float]],
        edges: list[tuple[int, int]],
        edge_curves: dict[tuple[int, int], tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] | None = None,
    ) -> np.ndarray:
        """Render an organic fluid inkblot along graph edges using variable-thickness Bézier curves and droplet nodes."""
        w, h = self.config.canvas_size
        r_min = float(self.config.blob_min_extent)
        r_max = float(self.config.blob_max_extent)
        step = max(0.5, float(self.config.sampling_step))

        surface = cairo.ImageSurface(cairo.FORMAT_A8, w, h)
        ctx = cairo.Context(surface)
        ctx.set_source_rgba(1.0, 1.0, 1.0, 1.0)

        # 1. Render variable-thickness strokes along each Bézier curve
        for u, v in edges:
            if edge_curves is not None and (u, v) in edge_curves:
                p1, cp, p2 = edge_curves[(u, v)]
            elif edge_curves is not None and (v, u) in edge_curves:
                p2, cp, p1 = edge_curves[(v, u)]
            else:
                p1 = node_coords[u]
                p2 = node_coords[v]
                cp = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)

            chord_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            num_steps = max(8, int(chord_len * 1.5 / step))
            t_vals = np.linspace(0.0, 1.0, num_steps)
            p1_a = np.array(p1, dtype=np.float64)
            cp_a = np.array(cp, dtype=np.float64)
            p2_a = np.array(p2, dtype=np.float64)
            pts = [(1.0 - t) ** 2 * p1_a + 2.0 * (1.0 - t) * t * cp_a + t ** 2 * p2_a for t in t_vals]

            # Compute continuous thickness profile sampled uniformly between min and max extent
            if r_max > r_min:
                raw_uniform = np.random.uniform(r_min, r_max, size=num_steps)
                sigma = max(2.0, num_steps * 0.12)
                smoothed = scipy.ndimage.gaussian_filter1d(raw_uniform, sigma=sigma)
                rng = smoothed.max() - smoothed.min()
                if rng > 1e-6:
                    extents = r_min + (r_max - r_min) * (smoothed - smoothed.min()) / rng
                else:
                    extents = [0.5 * (r_min + r_max)] * num_steps
            else:
                extents = [r_min] * num_steps

            # Render discs and connecting quads along the curve
            for i in range(num_steps):
                pt = pts[i]
                r = extents[i]
                ctx.arc(pt[0], pt[1], r, 0.0, 2.0 * math.pi)
                ctx.fill()

                if i < num_steps - 1:
                    pt_next = pts[i + 1]
                    r_next = extents[i + 1]
                    diff = pt_next - pt
                    seg_len = float(np.linalg.norm(diff))
                    if seg_len > 1e-4:
                        normal = np.array([-diff[1], diff[0]]) / seg_len
                        q1 = pt + r * normal
                        q2 = pt - r * normal
                        q3 = pt_next - r_next * normal
                        q4 = pt_next + r_next * normal
                        ctx.move_to(q1[0], q1[1])
                        ctx.line_to(q4[0], q4[1])
                        ctx.line_to(q3[0], q3[1])
                        ctx.line_to(q2[0], q2[1])
                        ctx.close_path()
                        ctx.fill()

        # 2. Draw fluid droplet disks at all graph nodes
        for node in node_coords:
            node_r = random.uniform(r_min, r_max) if r_max > r_min else r_min
            ctx.arc(node[0], node[1], node_r, 0.0, 2.0 * math.pi)
            ctx.fill()

        # 3. Extract alpha buffer and apply Gaussian smoothing for natural liquid inkblot silhouette
        buf = surface.get_data()
        mask_arr = np.ndarray(shape=(h, w), dtype=np.uint8, buffer=buf).copy()
        blurred = cv2.GaussianBlur(mask_arr, (7, 7), 2.0)
        final_mask = ((blurred > 100) * 255).astype(np.uint8)

        return final_mask

    def generate_trial(
        self, char: str, spread: float = 0.6, max_retries: int = 50
    ) -> tuple[Image.Image, Image.Image, dict]:
        """Generate a matched stimulus pair (occluded and parts) for one character configuration."""
        w, h = self.config.canvas_size

        for _ in range(max_retries):
            try:
                positions, rotations, shapes, patches, mean_dist = self.place_objects(char, spread=spread)
                node_coords, crossing_edges, shape_nodes, edge_curves = self.sample_crossing_edges(
                    shapes, return_curves=True
                )
                edges, all_curves = self.build_connected_planar_graph(
                    node_coords,
                    crossing_edges,
                    shapes,
                    shape_nodes,
                    edge_curves=edge_curves,
                    return_curves=True,
                )
                mask = self.generate_inkblot_mask(node_coords, edges, edge_curves=all_curves)
                break
            except RuntimeError:
                continue
        else:
            raise RuntimeError(
                f"Failed to generate valid Bregman stimulus after {max_retries} attempts."
            )

        # 1. Construct base canvas with characters
        bg_col = tuple(self.config.background_color)
        obj_col = tuple(self.config.object_color)
        base_canvas = Image.new("RGB", (w, h), bg_col)

        for (x, y), patch in zip(positions, patches):
            char_rgb = Image.new("RGB", patch.size, obj_col)
            base_canvas.paste(char_rgb, (x, y), mask=patch)

        # 2. Image 1: 'occluded' (inkblot drawn in inkblot_color, e.g. black)
        img_occluded = base_canvas.copy()
        inkblot_col = tuple(self.config.inkblot_color)
        inkblot_rgb = Image.new("RGB", (w, h), inkblot_col)
        mask_pil = Image.fromarray(mask)
        img_occluded.paste(inkblot_rgb, (0, 0), mask=mask_pil)

        # 3. Image 2: 'parts' (inkblot drawn in background_color, e.g. white, carving parts)
        img_parts = base_canvas.copy()
        bg_rgb = Image.new("RGB", (w, h), bg_col)
        img_parts.paste(bg_rgb, (0, 0), mask=mask_pil)

        meta = {
            "Character": char,
            "ObjectSpread": spread,
            "MeanObjectDistance": round(mean_dist, 2),
            "NumObjects": self.config.num_objects,
            "ObjectPositions": json.dumps(positions),
            "ObjectRotations": json.dumps([round(r, 2) for r in rotations]),
            "NumNodes": len(node_coords),
            "NumEdges": len(edges),
            "HasCycle": len(edges) > (len(node_coords) - 1),
            "BlobMinExtent": self.config.blob_min_extent,
            "BlobMaxExtent": self.config.blob_max_extent,
        }

        return img_occluded, img_parts, meta


@register("bregman_bs", "shape_recognition")
@generator(BregmanBsConfig)
def generate_bregman_bs(config: BregmanBsConfig):
    """Generate Bregman's B's dataset with paired occluded and parts conditions."""
    output_folder = Path(config.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    drawer = DrawBregmanBs(config)
    csv_path = output_folder / "annotation.csv"

    fieldnames = [
        "Path",
        "PairID",
        "Condition",
        "InkblotColor",
        "Character",
        "ObjectSpread",
        "MeanObjectDistance",
        "NumObjects",
        "ObjectPositions",
        "ObjectRotations",
        "NumNodes",
        "NumEdges",
        "HasCycle",
        "BlobMinExtent",
        "BlobMaxExtent",
        "BackgroundColor",
        "ObjectColor",
    ]

    with open(csv_path, "w", newline="") as annfile:
        writer = csv.DictWriter(annfile, fieldnames=fieldnames)
        writer.writeheader()

        pair_id = 0
        for spread in config.object_spread:
            for _ in tqdm(range(config.num_samples), desc=f"Generating pairs (spread={spread})"):
                char = random.choice(config.characters)
                img_occluded, img_parts, meta = drawer.generate_trial(char, spread=spread)

                stem_occluded = f"pair_{pair_id:04d}_char_{char}_spread{spread}_occluded.png"
                stem_parts = f"pair_{pair_id:04d}_char_{char}_spread{spread}_parts.png"

                path_occluded = output_folder / stem_occluded
                path_parts = output_folder / stem_parts

                img_occluded.save(path_occluded)
                img_parts.save(path_parts)

                row_occluded = {
                    "Path": stem_occluded,
                    "PairID": pair_id,
                    "Condition": "occluded",
                    "InkblotColor": "black",
                    "BackgroundColor": str(config.background_color),
                    "ObjectColor": str(config.object_color),
                    **meta,
                }
                writer.writerow(row_occluded)

                row_parts = {
                    "Path": stem_parts,
                    "PairID": pair_id,
                    "Condition": "parts",
                    "InkblotColor": "white",
                    "BackgroundColor": str(config.background_color),
                    "ObjectColor": str(config.object_color),
                    **meta,
                }
                writer.writerow(row_parts)

                pair_id += 1

    return str(output_folder)
