"""tilt illusion dataset generator."""
import csv
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from mindset.generators._base import GeneratorConfig, generator, register
from mindset.utils.drawing_utils import DrawStimuli
from mindset.utils.misc import apply_antialiasing


def generate_grating(canvas_size, frequency, orientation, phase=0):
    """generate a sinusoidal grating pattern."""
    width, height = canvas_size
    x = np.linspace(-np.pi, np.pi, width)
    y = np.linspace(-np.pi, np.pi, height)
    x, y = np.meshgrid(x, y)

    x_prime = x * np.cos(orientation) - y * np.sin(orientation)

    grating = 0.5 * (1 + np.sin(frequency * x_prime + phase))
    return grating


class DrawTiltIllusion(DrawStimuli):
    """draws tilt illusion stimuli."""

    def generate_illusion(
        self, theta_center, radius, center_test, freq, theta_context=None
    ):
        """generate a tilt illusion image with center and optional context grating."""
        if theta_context is not None:
            context = generate_grating(self.canvas_size, freq, theta_context)
            context = Image.fromarray(np.uint8(context * 255))
        else:
            context = self.create_canvas()
        if theta_center is not None:
            center = generate_grating(self.canvas_size, freq, theta_center)
            center = Image.fromarray(np.uint8(center * 255))
        else:
            center = self.create_canvas()
        mask = Image.new("L", center.size, 0)

        draw = ImageDraw.Draw(mask)
        center_test = np.array(center_test) * self.canvas_size
        draw.ellipse(
            (
                center_test[0] - radius,
                center_test[1] - radius,
                center_test[0] + radius,
                center_test[1] + radius,
            ),
            fill=255,
        )

        context.paste(center, mask=mask)
        return apply_antialiasing(context) if self.antialiasing else context


@dataclass
class TiltConfig(GeneratorConfig):
    """config for tilt illusion dataset."""
    num_samples_only_center: int = field(default=1000, metadata={"min": 1, "max": 50000, "step": 10, "label": "only center samples"})
    num_samples_only_context: int = field(default=1000, metadata={"min": 1, "max": 50000, "step": 10, "label": "only context samples"})
    num_samples_center_context: int = field(default=1000, metadata={"min": 1, "max": 50000, "step": 10, "label": "center+context samples"})
    output_folder: str = field(default="data/visual_illusions/tilt", metadata={"label": "output folder"})


def _get_random_values(canvas_size):
    """return random theta, radius, center, freq for tilt stimuli."""
    size_scale = np.random.uniform(0.1, 0.6)
    radius = canvas_size[0] // 2 * size_scale
    center = (
        np.random.uniform(radius, canvas_size[0] - radius) // canvas_size[0],
        np.random.uniform(radius, canvas_size[1] - radius) // canvas_size[1],
    )
    freq = random.randint(5, 20)
    theta = np.random.uniform(-np.pi / 2, np.pi / 2)
    return theta, radius, center, freq


@register("tilt", "visual_illusions")
@generator(TiltConfig)
def generate_all(config: TiltConfig):
    """generate tilt illusion dataset."""
    output_folder = Path(config.output_folder)
    for d in ["only_center", "only_context", "center_context"]:
        (output_folder / d).mkdir(parents=True, exist_ok=True)

    ds = DrawTiltIllusion(
        background=config.background_color,
        canvas_size=config.canvas_size,
        antialiasing=config.antialiasing,
    )

    with open(output_folder / "annotation.csv", "w", newline="") as annfile:
        writer = csv.writer(annfile)
        writer.writerow(["Path", "Type", "BackgroundColor", "ThetaCenter", "Radius", "Frequency", "ThetaContext", "IterNum"])

        for i in tqdm(range(config.num_samples_only_center)):
            unique_hex = uuid.uuid4().hex[:8]
            theta_center, radius, _, freq = _get_random_values(config.canvas_size)
            path = Path("only_center") / f"{-theta_center:.3f}_0_{unique_hex}.png"
            img = ds.generate_illusion(theta_center, radius, (0.5, 0.5), freq)
            img.save(str(output_folder / path))
            writer.writerow([path, "only_center", ds.background, theta_center, radius, freq, "", i])

        for i in tqdm(range(config.num_samples_only_context)):
            unique_hex = uuid.uuid4().hex[:8]
            theta_context, radius, _, freq = _get_random_values(config.canvas_size)
            path = Path("only_context") / f"{-theta_context:.3f}_0_{unique_hex}.png"
            img = ds.generate_illusion(None, radius, (0.5, 0.5), freq, theta_context)
            img.save(str(output_folder / path))
            writer.writerow([path, "only_context", ds.background, 0, radius, freq, theta_context, i])

        all_thetas = np.linspace(-np.pi / 2, np.pi / 2, config.num_samples_center_context)
        for i, theta_context in enumerate(tqdm(all_thetas)):
            _, radius, _, freq = _get_random_values(config.canvas_size)
            img = ds.generate_illusion(0, radius, (0.5, 0.5), freq, theta_context)
            unique_hex = uuid.uuid4().hex[:8]
            path = Path("center_context") / f"0_{theta_context:.3f}_{unique_hex}.png"
            img.save(output_folder / path)
            writer.writerow([path, "center_context", ds.background, 0, radius, freq, theta_context, i])

    return str(output_folder)
