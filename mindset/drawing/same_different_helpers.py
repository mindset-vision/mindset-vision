import cv2
import numpy as np

from mindset.drawing.same_different import DrawSameDifferentStimuli


def is_overlapping(img: np.array, background_color: tuple, threshold: int = 2):
    """detect whether two shapes overlap in an image."""
    img_c = img.copy()
    img_c[img == background_color] = 0

    gray = cv2.cvtColor(img_c, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    b_rects = []
    for c in cnts:
        b_rects.append(cv2.boundingRect(c))

    return len(b_rects) != 2


get_irregular_polygon = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=False, sides=None, thickness=1, **kwargs
)

get_regular = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=True, sides=None, thickness=1, **kwargs
)

get_open = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=False, sides=None, thickness=1, closed=False, **kwargs,
)

get_wider_line = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=False, sides=None, thickness=2, **kwargs
)


def get_rnd_color(ds, label, **kwargs):
    """generate a shape with random color."""
    color = tuple(np.random.randint(1, high=256, size=3))
    color = (int(color[0]), int(color[1]), int(color[2]))
    return ds.svrt_1_img(
        category=label, regular=False, color_a=color, sides=None, thickness=1, **kwargs
    )


get_filled = lambda ds, label, **kwargs: ds.svrt_1_img(
    category=label, regular=False, sides=None, thickness=1, filled=True, **kwargs,
)

get_straight_lines = lambda ds, label, **kwargs: ds.make_straight_lines_sd_diffrot(
    category=label, line_thickness=1, **kwargs
)

get_rectangles = lambda ds, label, **kwargs: ds.make_rectangles_sd(
    category=label, **kwargs
)

get_open_squares = lambda ds, label, **kwargs: ds.make_connected_open_squares(
    category=label, line_width=1, **kwargs
)

get_closed_squares = lambda ds, label, **kwargs: ds.make_connected_open_squares(
    category=label, line_width=1, is_closed=True, **kwargs
)


def is_integer(n):
    """check whether a value can be parsed as an integer."""
    try:
        int(n)
        return True
    except ValueError:
        return False
