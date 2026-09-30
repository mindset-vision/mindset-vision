"""uncrowding dataset generator with shape and distribution variants."""

import csv
import math
import random
import uuid
from itertools import combinations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm.auto import tqdm

from mindset.drawing.base import DrawStimuli
from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils import apply_antialiasing

# Shape identifier constants
SQUARE = 1
CIRCLE = 2
HEXAGON = 3
STAR = 4  # 7-pointed star

SHAPE_NAMES = {
    SQUARE: "square",
    CIRCLE: "circle",
    HEXAGON: "hexagon",
    STAR: "star",
}

NAME_TO_SHAPE = {v: k for k, v in SHAPE_NAMES.items()}


class DrawUncrowding(DrawStimuli):
    """draws uncrowding stimuli with verniers and shape flanker grids."""

    def __init__(
        self,
        bar_width=2,
        bar_height=12,
        flanker_bar_width=3,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.vernier_bar_width = bar_width
        self.vernier_bar_height = bar_height
        self.flanker_bar_width = flanker_bar_width

    def _draw_polygon_edges(
        self, draw: ImageDraw.ImageDraw, vertices: list[tuple[float, float]]
    ) -> None:
        """draw closed polygon boundary lines along sequential vertices."""
        n = len(vertices)
        for i in range(n):
            draw.line([vertices[i], vertices[(i + 1) % n]], fill=self.line_col, width=self.flanker_bar_width)

    def draw_square(
        self, shape_size: int, rotation: float = 0.0, scale: float = 1.0
    ) -> Image.Image:
        """draw a square outline patch with full half-side and optional rotation and scale."""
        patch = Image.new("RGB", (shape_size, shape_size), self.background)
        draw = ImageDraw.Draw(patch)
        margin = max(1, self.flanker_bar_width) + 1.0
        center_x = shape_size / 2.0
        center_y = shape_size / 2.0
        r_max = max(1.0, (shape_size / 2.0) - margin)
        half_side = (r_max / math.sqrt(2.0)) * scale

        rot_rad = math.radians(rotation)
        base_corners = [
            (-half_side, -half_side),
            (half_side, -half_side),
            (half_side, half_side),
            (-half_side, half_side),
        ]
        vertices = []
        for dx, dy in base_corners:
            x = center_x + dx * math.cos(rot_rad) - dy * math.sin(rot_rad)
            y = center_y + dx * math.sin(rot_rad) + dy * math.cos(rot_rad)
            vertices.append((x, y))

        self._draw_polygon_edges(draw, vertices)
        return patch

    def draw_circle(self, shape_size: int, scale: float = 1.0) -> Image.Image:
        """draw a circle outline patch with optional scale."""
        patch = Image.new("RGB", (shape_size, shape_size), self.background)
        draw = ImageDraw.Draw(patch)
        margin = max(1, self.flanker_bar_width) + 1.0
        center_x = shape_size / 2.0
        center_y = shape_size / 2.0
        radius = max(1.0, ((shape_size / 2.0) - margin) * scale)

        draw.ellipse(
            [center_x - radius, center_y - radius, center_x + radius, center_y + radius],
            outline=self.line_col,
            width=self.flanker_bar_width,
        )
        return patch

    def draw_hexagon(
        self, shape_size: int, rotation: float = 0.0, scale: float = 1.0
    ) -> Image.Image:
        """draw a regular hexagon outline patch with optional rotation (deg) and scale."""
        patch = Image.new("RGB", (shape_size, shape_size), self.background)
        draw = ImageDraw.Draw(patch)
        center_x = shape_size / 2.0
        center_y = shape_size / 2.0
        margin = max(1, self.flanker_bar_width) + 1.0
        radius = max(1.0, ((shape_size / 2.0) - margin) * scale)

        rot_rad = math.radians(rotation)
        vertices = []
        for k in range(6):
            angle = k * math.pi / 3.0 + rot_rad
            x = center_x + radius * math.cos(angle)
            y = center_y - radius * math.sin(angle)
            vertices.append((x, y))

        self._draw_polygon_edges(draw, vertices)
        return patch

    def draw_star_7(
        self, shape_size: int, rotation: float = 0.0, scale: float = 1.0
    ) -> Image.Image:
        """draw a regular 7-pointed star (heptagram) outline patch with enlarged body."""
        patch = Image.new("RGB", (shape_size, shape_size), self.background)
        draw = ImageDraw.Draw(patch)
        center_x = shape_size / 2.0
        center_y = shape_size / 2.0
        margin = max(1, self.flanker_bar_width // 2) + 1.0
        radius_outer = max(1.0, ((shape_size / 2.0) - margin) * scale)
        radius_inner = radius_outer * 0.65

        rot_rad = math.radians(rotation)
        vertices = []
        for k in range(14):
            m = k // 2
            if k % 2 == 0:
                r = radius_outer
                angle = m * (2.0 * math.pi / 7.0) - math.pi / 2.0 + rot_rad
            else:
                r = radius_inner
                angle = (m + 0.5) * (2.0 * math.pi / 7.0) - math.pi / 2.0 + rot_rad
            x = center_x + r * math.cos(angle)
            y = center_y + r * math.sin(angle)
            vertices.append((x, y))

        self._draw_polygon_edges(draw, vertices)
        return patch

    def draw_shape(
        self,
        shape_id: int | str,
        shape_size: int,
        rotation: float = 0.0,
        scale: float = 1.0,
    ) -> Image.Image:
        """draw a flanker shape patch by ID or name with transformation."""
        if isinstance(shape_id, str):
            shape_id = NAME_TO_SHAPE.get(shape_id.lower(), shape_id)

        if shape_id == SQUARE:
            return self.draw_square(shape_size, rotation=rotation, scale=scale)
        elif shape_id == CIRCLE:
            return self.draw_circle(shape_size, scale=scale)
        elif shape_id == HEXAGON:
            return self.draw_hexagon(shape_size, rotation=rotation, scale=scale)
        elif shape_id == STAR:
            return self.draw_star_7(shape_size, rotation=rotation, scale=scale)
        else:
            return Image.new("RGB", (shape_size, shape_size), self.background)

    def draw_vernier(
        self, vernier_type: int, offset_pixels: int
    ) -> Image.Image:
        """
        draw a Vernier stimulus patch.

        vernier_type = 0: top line shifted to the right.
        vernier_type = 1: top line shifted to the left.
        """
        bar_w = self.vernier_bar_width
        bar_h = self.vernier_bar_height
        patch_w = bar_w * 2 + offset_pixels + 6
        patch_h = bar_h * 2 + 4
        patch = Image.new("RGB", (patch_w, patch_h), self.background)
        draw = ImageDraw.Draw(patch)

        center_x = patch_w // 2

        if vernier_type == 0:
            # Top line to right, bottom line to left
            x_top = center_x + math.ceil(offset_pixels / 2.0)
            x_bottom = center_x - math.floor(offset_pixels / 2.0)
        else:
            # Top line to left, bottom line to right
            x_top = center_x - math.floor(offset_pixels / 2.0)
            x_bottom = center_x + math.ceil(offset_pixels / 2.0)

        y_top_start = 2
        y_top_end = y_top_start + bar_h
        y_bottom_start = y_top_end
        y_bottom_end = y_bottom_start + bar_h

        draw.line(
            [(x_top, y_top_start), (x_top, y_top_end - 1)],
            fill=self.line_col,
            width=bar_w,
        )
        draw.line(
            [(x_bottom, y_bottom_start), (x_bottom, y_bottom_end - 1)],
            fill=self.line_col,
            width=bar_w,
        )

        return patch


def generate_grid_layout(num_rows: int, num_cols: int, arrangement: str) -> np.ndarray:
    """
    generate a binary (0/1) grid assignment matrix of shape (num_rows, num_cols).

    Group 0 is the central flanker group surrounding the vernier.
    Group 1 is the alternate flanker group.
    """
    if num_rows % 2 == 0 or num_cols % 2 == 0:
        raise ValueError(f"num_rows and num_cols must be odd, got ({num_rows}, {num_cols})")

    r_c = num_rows // 2
    c_c = num_cols // 2
    grid = np.zeros((num_rows, num_cols), dtype=int)

    if arrangement == "uniform":
        return grid
    elif arrangement == "interleaved_cols":
        for c in range(num_cols):
            grid[:, c] = abs(c - c_c) % 2
    elif arrangement == "interleaved_rows":
        for r in range(num_rows):
            grid[r, :] = abs(r - r_c) % 2
    elif arrangement == "interleaved_both":
        for r in range(num_rows):
            for c in range(num_cols):
                grid[r, c] = (abs(r - r_c) + abs(c - c_c)) % 2
    else:
        raise ValueError(f"Unknown arrangement: {arrangement}")

    return grid


def get_valid_arrangements(num_rows: int, num_cols: int) -> list[str]:
    """return non-degenerate spatial arrangements for given grid dimensions."""
    if num_rows == 1 and num_cols == 1:
        return ["uniform"]
    elif num_rows == 1:
        return ["uniform", "interleaved_cols"]
    elif num_cols == 1:
        return ["uniform", "interleaved_rows"]
    else:
        return ["uniform", "interleaved_cols", "interleaved_rows", "interleaved_both"]


def serialize_grid(grid: np.ndarray) -> str:
    """serialize binary grid into a compact string format: RxC:row0/row1/..."""
    r, c = grid.shape
    rows_str = "/".join("".join(str(val) for val in grid[i, :]) for i in range(r))
    return f"{r}x{c}:{rows_str}"


def generate_uncrowding_stimulus(
    drawer: DrawUncrowding,
    cell_shapes: list[list[int | str]],
    cell_rotations: list[list[float]],
    cell_scales: list[list[float]],
    vernier_type: int,
    vernier_offset: int,
    vernier_in_out: str,
    shape_size: int,
    canvas_size: tuple[int, int],
) -> Image.Image:
    """generate a complete uncrowding stimulus image with flanker grid and Vernier."""
    canvas = drawer.create_canvas()

    num_rows = len(cell_shapes)
    num_cols = len(cell_shapes[0])

    grid_width = num_cols * shape_size
    grid_height = num_rows * shape_size

    canvas_w, canvas_h = canvas_size
    grid_x0 = (canvas_w - grid_width) // 2
    grid_y0 = (canvas_h - grid_height) // 2

    for r in range(num_rows):
        for c in range(num_cols):
            s_id = cell_shapes[r][c]
            rot = cell_rotations[r][c]
            sc = cell_scales[r][c]
            shape_patch = drawer.draw_shape(
                shape_id=s_id,
                shape_size=shape_size,
                rotation=rot,
                scale=sc,
            )
            px = grid_x0 + c * shape_size
            py = grid_y0 + r * shape_size
            canvas.paste(shape_patch, (px, py))

    vernier_patch = drawer.draw_vernier(vernier_type, vernier_offset)
    vw, vh = vernier_patch.size

    r_c = num_rows // 2
    c_c = num_cols // 2

    if vernier_in_out == "inside":
        center_x = grid_x0 + c_c * shape_size + shape_size // 2
        center_y = grid_y0 + r_c * shape_size + shape_size // 2

        vx = center_x - vw // 2
        vy = center_y - vh // 2

        canvas.paste(vernier_patch, (vx, vy), mask=vernier_patch.convert("L"))
    else:
        # Outside condition: sample collision-free position outside grid bounding box
        safety_margin = 10
        min_x, min_y = safety_margin, safety_margin
        max_x = canvas_w - vw - safety_margin
        max_y = canvas_h - vh - safety_margin

        gx1 = grid_x0 - safety_margin
        gy1 = grid_y0 - safety_margin
        gx2 = grid_x0 + grid_width + safety_margin
        gy2 = grid_y0 + grid_height + safety_margin

        valid_position = False
        attempts = 0

        while not valid_position and attempts < 300:
            attempts += 1
            cand_x = random.randint(min_x, max(min_x, max_x))
            cand_y = random.randint(min_y, max(min_y, max_y))

            vx1, vy1 = cand_x, cand_y
            vx2, vy2 = cand_x + vw, cand_y + vh

            overlap = not (vx2 < gx1 or vx1 > gx2 or vy2 < gy1 or vy1 > gy2)
            if not overlap:
                valid_position = True
                vx, vy = cand_x, cand_y

        if not valid_position:
            # Fallback to vertical margin (above or below grid)
            if gy1 - vh >= min_y:
                vx = max(min_x, (canvas_w - vw) // 2)
                vy = min_y
            elif gy2 <= max_y:
                vx = max(min_x, (canvas_w - vw) // 2)
                vy = max_y
            else:
                vx = min_x
                vy = min_y

        canvas.paste(vernier_patch, (vx, vy), mask=vernier_patch.convert("L"))

    img = canvas.convert("RGBA")
    if drawer.antialiasing:
        img = apply_antialiasing(img)
    return img


# ---------------------------------------------------------------------------
# Shared Configuration & Generator Utilities
# ---------------------------------------------------------------------------


@dataclass
class _BaseUncrowdingConfig(GeneratorConfig):
    """shared base configuration for uncrowding generators."""

    row_counts: list[int] = field(
        default_factory=lambda: [1, 3, 5],
        metadata={"label": "row counts (odd numbers <= 5)"},
    )
    col_counts: list[int] = field(
        default_factory=lambda: [1, 3, 5, 7],
        metadata={"label": "col counts (odd numbers <= 7)"},
    )
    num_samples_vernier_inside: int | None = field(
        default=None,
        metadata={"label": "vernier inside samples (None for all arrangements)"},
    )
    num_samples_vernier_outside: int | None = field(
        default=3_000,
        metadata={"label": "vernier outside samples (None for all arrangements)"},
    )
    vernier_offset: list[int] | int = field(
        default=4,
        metadata={"min": 2, "max": 8, "label": "vernier horizontal offset/separation (px)"},
    )
    canvas_size: tuple[int, int] = field(
        default_factory=lambda: (512, 512),
        metadata={"min": 32, "max": 1024, "step": 16, "label": "canvas size"},
    )
    flanker_bar_width: int = field(
        default=3,
        metadata={"min": 1, "max": 5, "label": "flanker stroke width (px)"},
    )
    bar_width: int = field(
        default=2,
        metadata={"min": 2, "max": 5, "label": "vernier stroke width (px)"},
    )
    bar_height: int = field(
        default=12,
        metadata={"min": 10, "max": 15, "label": "vernier bar height (px)"},
    )
    shape_size: int = field(
        default=64,
        metadata={"min": 20, "max": 256, "label": "flanker shape size (px)"},
    )
    antialiasing: bool = field(default=False, metadata={"label": "antialiasing"})
    output_folder: str = field(
        default="data/low_mid_vision/uncrowding",
        metadata={"label": "output folder"},
    )

    def __post_init__(self):
        if isinstance(self.vernier_offset, (int, float, str)):
            self.vernier_offset = [int(self.vernier_offset)]
        elif isinstance(self.vernier_offset, (list, tuple)):
            self.vernier_offset = [int(x) for x in self.vernier_offset]
        else:
            self.vernier_offset = [int(self.vernier_offset)]


def _init_generator_environment(
    config: _BaseUncrowdingConfig,
) -> tuple[Path, DrawUncrowding, list[tuple[int, int]]]:
    """initialize generator output directories, drawer, and valid grid sizes."""
    output_folder = Path(config.output_folder)
    for v_mode in ("outside", "inside"):
        for v_type in (0, 1):
            (output_folder / v_mode / str(v_type)).mkdir(exist_ok=True, parents=True)

    drawer = DrawUncrowding(
        canvas_size=config.canvas_size,
        background=tuple(config.background_color),
        antialiasing=config.antialiasing,
        bar_width=config.bar_width,
        bar_height=config.bar_height,
        flanker_bar_width=config.flanker_bar_width,
    )

    grid_sizes = [
        (r, c)
        for r in config.row_counts
        for c in config.col_counts
        if r <= c and r % 2 != 0 and c % 2 != 0
    ]
    return output_folder, drawer, grid_sizes


def _sample_conditions(
    conditions: list[dict], num_requested: int | None
) -> list[dict]:
    """subsample conditions if requested, otherwise return all conditions."""
    if num_requested is not None:
        return random.choices(conditions, k=int(num_requested))
    return list(conditions)


def _save_stimulus(
    img: Image.Image,
    output_folder: Path,
    v_mode: str,
    v_type: int,
    stem: str,
) -> Path:
    """save a stimulus image to its category folder with a unique hex ID."""
    unique_hex = uuid.uuid4().hex[:8]
    rel_path = Path(v_mode) / str(v_type) / f"{stem}_{unique_hex}.png"
    img.save(output_folder / rel_path)
    return rel_path


# ---------------------------------------------------------------------------
# Generator 1: uncrowding_shapes
# ---------------------------------------------------------------------------


@dataclass
class UncrowdingShapesConfig(_BaseUncrowdingConfig):
    """config for uncrowding dataset with shape combinations."""

    shapes: list[str] = field(
        default_factory=lambda: ["square", "circle", "hexagon", "star"],
        metadata={"label": "shapes to combine"},
    )
    bar_width: int = field(
        default=2,
        metadata={"min": 2, "max": 5, "label": "vernier stroke width (px)"},
    )
    output_folder: str = field(
        default="data/low_mid_vision/uncrowding_shapes",
        metadata={"label": "output folder"},
    )


@register("uncrowding_shapes", "low_mid_vision")
@generator(UncrowdingShapesConfig)
def generate_shapes(config: UncrowdingShapesConfig):
    """generate uncrowding dataset over shape combinations."""
    output_folder, drawer, grid_sizes = _init_generator_environment(config)

    conditions = []
    for r, c in grid_sizes:
        for arr in get_valid_arrangements(r, c):
            grid = generate_grid_layout(r, c, arr)
            pat_str = serialize_grid(grid)

            if arr == "uniform":
                for s in config.shapes:
                    conditions.append(
                        {
                            "num_rows": r,
                            "num_cols": c,
                            "arrangement": arr,
                            "grid": grid,
                            "pat_str": pat_str,
                            "center_shape": s,
                            "alternate_shape": "none",
                        }
                    )
            else:
                for s_a in config.shapes:
                    for s_b in config.shapes:
                        if s_a != s_b:
                            conditions.append(
                                {
                                    "num_rows": r,
                                    "num_cols": c,
                                    "arrangement": arr,
                                    "grid": grid,
                                    "pat_str": pat_str,
                                    "center_shape": s_a,
                                    "alternate_shape": s_b,
                                }
                            )

    csv_path = output_folder / "annotation.csv"
    with open(csv_path, "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "ConditionType",
                "VernierInOut",
                "VernierType",
                "VernierOffset",
                "NumRows",
                "NumCols",
                "GridArrangement",
                "CenterShape",
                "AlternateShape",
                "GridPattern",
                "BackgroundColor",
                "ShapeSize",
                "IterNum",
            ]
        )

        for v_mode in tqdm(["outside", "inside"], desc="Mode"):
            num_req = (
                config.num_samples_vernier_outside
                if v_mode == "outside"
                else config.num_samples_vernier_inside
            )
            mode_conditions = _sample_conditions(conditions, num_req)

            for v_offset in config.vernier_offset:
                for v_type in (0, 1):
                    for n, cond in enumerate(tqdm(mode_conditions, desc='Conditions')):
                        grid = cond["grid"]
                        r, c = cond["num_rows"], cond["num_cols"]
                        s_a = cond["center_shape"]
                        s_b = cond["alternate_shape"]

                        cell_shapes = [
                            [s_a if grid[i, j] == 0 else s_b for j in range(c)]
                            for i in range(r)
                        ]
                        cell_rotations = [[0.0] * c for _ in range(r)]
                        cell_scales = [[1.0] * c for _ in range(r)]

                        img = generate_uncrowding_stimulus(
                            drawer=drawer,
                            cell_shapes=cell_shapes,
                            cell_rotations=cell_rotations,
                            cell_scales=cell_scales,
                            vernier_type=v_type,
                            vernier_offset=v_offset,
                            vernier_in_out=v_mode,
                            shape_size=config.shape_size,
                            canvas_size=config.canvas_size,
                        )

                        stem = (
                            f"{v_mode}_v{v_type}_offset{v_offset}_"
                            f"{r}x{c}_{cond['arrangement']}_{s_a}_{s_b}_{n}"
                        )
                        rel_path = _save_stimulus(img, output_folder, v_mode, v_type, stem)
                        writer.writerow(
                            [
                                rel_path.as_posix(),
                                "two_shapes",
                                v_mode,
                                v_type,
                                v_offset,
                                r,
                                c,
                                cond["arrangement"],
                                s_a,
                                s_b,
                                cond["pat_str"],
                                drawer.background,
                                config.shape_size,
                                n,
                            ]
                        )

    return str(output_folder)


# ---------------------------------------------------------------------------
# Generator 2: uncrowding_distributions
# ---------------------------------------------------------------------------


@dataclass
class UncrowdingDistributionsConfig(_BaseUncrowdingConfig):
    """config for uncrowding dataset with user-specified feature distribution configurations."""

    dimension: str = field(
        default="rotation",
        metadata={"choices": ["rotation", "scale"], "label": "feature dimension (rotation or scale)"},
    )
    loc: list[float] = field(
        default_factory=lambda: [0.0, 0.0],
        metadata={"label": "distribution locations (e.g. angle in deg or scale factor)"},
    )
    scale: list[float] = field(
        default_factory=lambda: [10.0, 10.0],
        metadata={"label": "distribution scales (e.g. spread / std)"},
    )
    base_shapes: list[str] = field(
        default_factory=lambda: ["square", "hexagon", "star"],
        metadata={"label": "base shapes to use (circles excluded due to rotational symmetry)"},
    )
    bar_width: int = field(
        default=3,
        metadata={"min": 2, "max": 5, "label": "vernier stroke width (px)"},
    )
    output_folder: str = field(
        default="data/low_mid_vision/uncrowding_distributions",
        metadata={"label": "output folder"},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.dimension not in ("rotation", "scale"):
            raise ValueError(f"dimension must be 'rotation' or 'scale', got '{self.dimension}'")

        self.loc = [float(x) for x in self.loc]
        self.scale = [float(x) for x in self.scale]

        if len(self.loc) != len(self.scale):
            raise ValueError(
                f"loc and scale lists must have equal length (got {len(self.loc)} locs and {len(self.scale)} scales)"
            )


@register("uncrowding_distributions", "low_mid_vision")
@generator(UncrowdingDistributionsConfig)
def generate_distributions(config: UncrowdingDistributionsConfig):
    """generate uncrowding dataset over continuous feature distributions."""
    output_folder, drawer, grid_sizes = _init_generator_environment(config)

    # Ensure circles are never used in distribution generator
    base_shapes = [s for s in config.base_shapes if s.lower() != "circle"]

    conditions = []
    for r, c in grid_sizes:
        for arr in get_valid_arrangements(r, c):
            grid = generate_grid_layout(r, c, arr)
            pat_str = serialize_grid(grid)

            for shape in base_shapes:
                if arr == "uniform":
                    for (loc_a, scale_a) in zip(config.loc, config.scale):
                        conditions.append(
                            {
                                "num_rows": r,
                                "num_cols": c,
                                "arrangement": arr,
                                "grid": grid,
                                "pat_str": pat_str,
                                "base_shape": shape,
                                "loc_a": loc_a,
                                "scale_a": scale_a,
                                "loc_b": 'none',
                                "scale_b": 'none',
                            }
                        )
                else:
                    dist_comb = combinations(zip(config.loc, config.scale), 2)
                    for (loc_a, scale_a), (loc_b, scale_b) in dist_comb:
                        conditions.append(
                            {
                                "num_rows": r,
                                "num_cols": c,
                                "arrangement": arr,
                                "grid": grid,
                                "pat_str": pat_str,
                                "base_shape": shape,
                                "loc_a": loc_a,
                                "scale_a": scale_a,
                                "loc_b": loc_b,
                                "scale_b": scale_b,
                            }
                        )

    csv_path = output_folder / "annotation.csv"
    with open(csv_path, "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(
            [
                "Path",
                "ConditionType",
                "VernierInOut",
                "VernierType",
                "VernierOffset",
                "NumRows",
                "NumCols",
                "GridArrangement",
                "BaseShape",
                "Dimension",
                "GridPattern",
                "LocA",
                "ScaleA",
                "LocB",
                "ScaleB",
                "BackgroundColor",
                "ShapeSize",
                "IterNum",
            ]
        )

        for v_mode in tqdm(["outside", "inside"], desc="Mode"):
            num_req = (
                config.num_samples_vernier_outside
                if v_mode == "outside"
                else config.num_samples_vernier_inside
            )
            mode_conditions = _sample_conditions(conditions, num_req)

            for v_offset in config.vernier_offset:
                for v_type in (0, 1):
                    for n, cond in enumerate(tqdm(mode_conditions, desc='Conditions')):
                        grid = cond["grid"]
                        r, c = cond["num_rows"], cond["num_cols"]
                        shape = cond["base_shape"]
                        arr = cond["arrangement"]

                        loc_a = cond["loc_a"]
                        scale_a = cond["scale_a"]
                        loc_b = cond["loc_b"]
                        scale_b = cond["scale_b"]

                        cell_shapes = [[shape] * c for _ in range(r)]
                        cell_rotations = [[0.0] * c for _ in range(r)]
                        cell_scales = [[1.0] * c for _ in range(r)]

                        r_c, c_c = r // 2, c // 2
                        for i in range(r):
                            for j in range(c):
                                if i == r_c and j == c_c:
                                    # Central flanker enclosing Vernier: strictly fixed
                                    cell_rotations[i][j] = 0.0
                                    cell_scales[i][j] = 1.0
                                elif (arr == "uniform") or (grid[i, j] == 1):
                                    val = loc_a if scale_a <= 1e-7 else random.gauss(loc_a, scale_a)
                                    if config.dimension == "rotation":
                                        cell_rotations[i][j] = val
                                    elif config.dimension == "scale":
                                        cell_scales[i][j] = max(0.1, min(1.0, val))
                                else:
                                    val = loc_b if scale_b <= 1e-7 else random.gauss(loc_b, scale_b)
                                    if config.dimension == "rotation":
                                        cell_rotations[i][j] = val
                                    elif config.dimension == "scale":
                                        cell_scales[i][j] = max(0.1, min(1.0, val))

                        img = generate_uncrowding_stimulus(
                            drawer=drawer,
                            cell_shapes=cell_shapes,
                            cell_rotations=cell_rotations,
                            cell_scales=cell_scales,
                            vernier_type=v_type,
                            vernier_offset=v_offset,
                            vernier_in_out=v_mode,
                            shape_size=config.shape_size,
                            canvas_size=config.canvas_size,
                        )

                        stem = (
                            f"{v_mode}_v{v_type}_offset{v_offset}_"
                            f"{r}x{c}_{arr}_{shape}_{config.dimension}_"
                            f"cfg_loc_a{loc_a}_scale_a{scale_a}_loc_b{loc_b}_scale_b{scale_b}_{n}"
                        )
                        rel_path = _save_stimulus(img, output_folder, v_mode, v_type, stem)
                        writer.writerow(
                            [
                                rel_path.as_posix(),
                                "two_distributions",
                                v_mode,
                                v_type,
                                v_offset,
                                r,
                                c,
                                arr,
                                shape,
                                config.dimension,
                                cond["pat_str"],
                                loc_a,
                                scale_a,
                                loc_b,
                                scale_b,
                                drawer.background,
                                config.shape_size,
                                n,
                            ]
                        )

    return str(output_folder)
