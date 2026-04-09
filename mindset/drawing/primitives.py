"""shared drawing primitives for visual illusion generators."""

from PIL import ImageDraw


def add_arrow(canvas, coord: tuple[float, float], fill=255, arrow_size=1):
    """add a downward-pointing arrow indicator at the given coordinate."""
    image_width = canvas.size[0]
    resize_ratio = arrow_size * image_width / (4 * 224)

    arrow_line_length = 45 * resize_ratio
    triangle_height = 30 * resize_ratio
    triangle_width = 20 * resize_ratio
    line_width = int(2 * resize_ratio)

    coord = (coord[0], coord[1] - 1)

    coord_1 = coord
    coord_2 = (coord[0], coord[1] - arrow_line_length)

    coord_triangle_left = (coord[0] - triangle_width, coord[1] - triangle_height)
    coord_triangle_right = (coord[0] + triangle_width, coord[1] - triangle_height)

    draw = ImageDraw.Draw(canvas)
    draw.line((coord_1, coord_2), fill=fill, width=line_width)
    draw.polygon((coord_1, coord_triangle_left, coord_triangle_right), fill=fill)
    return canvas
