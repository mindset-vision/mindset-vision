import numpy as np
from PIL import Image, ImageDraw

from mindset.utils.drawing_utils import DrawStimuli
from mindset.utils.misc import apply_antialiasing


def generate_grating(canvas_size, frequency, orientation, phase=0):
    """generate a sinusoidal grating pattern."""
    width, height = canvas_size
    x = np.linspace(-np.pi, np.pi, width)
    y = np.linspace(-np.pi, np.pi, height)
    x, y = np.meshgrid(x, y)

    # Rotate the grid by the specified orientation
    x_prime = x * np.cos(orientation) - y * np.sin(orientation)

    # Create the sinusoidal grating
    grating = 0.5 * (1 + np.sin(frequency * x_prime + phase))
    return grating


class DrawTiltIllusion(DrawStimuli):
    def generate_illusion(
        self, theta_center, radius, center_test, freq, theta_context=None
    ):
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
