"""drawing class for weber law line length stimuli."""

from PIL import ImageDraw

from mindset.utils.drawing_utils import DrawStimuli


class DrawWeberLength(DrawStimuli):
    """draws a horizontal line with configurable length, width, and luminance."""

    def gen_stim(self, length, width, lum):
        """generate a single line stimulus image."""
        img = self.create_canvas()
        x0, y0 = self.canvas_size[0] / 2 - (length / 2), self.canvas_size[1] / 2
        x1, y1 = self.canvas_size[0] / 2 + (length / 2), self.canvas_size[1] / 2
        bbox = [(x0, y0), (x1, y1)]
        drawing = ImageDraw.Draw(img)
        drawing.line(bbox, width=width, fill=(lum, lum, lum))
        return img
