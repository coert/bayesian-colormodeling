from PIL import Image as PILImage
from io import BytesIO

from IPython.core.display import Image
from IPython.display import display

from unicodedata import normalize as normalize_slug
from matplotlib import colors as mcolors
from matplotlib._color_data import BASE_COLORS
from matplotlib._color_data import CSS4_COLORS
from matplotlib._color_data import XKCD_COLORS


def getImage(ima, width=None):
    """
    description: use HTML5 to display from memory an image array in a notebook
    parameters:
        - ima: a numpy image array of color depth 1 or 3
    returns:
        n/a
    """
    im = PILImage.fromarray(ima)
    bio = BytesIO()
    im.save(bio, format="png")
    if width is not None:
        return Image(bio.getvalue(), format="png", width=width)
    else:
        return Image(bio.getvalue(), format="png")


def displayImArr(ima, width=500):
    """
    description: use HTML5 to display from memory an image array in a notebook
    parameters:
        - ima: a numpy image array of color depth 1 or 3
    returns:
        n/a
    """
    display(getImage(ima, width=width))

def displayImURI(imu, width=500):
    display(Image(imu, format="png", width=width))


def get_valid_cv2_color(color: str) -> tuple[int, ...]:
    if color in mcolors.get_named_colors_mapping():
        cv2_color = tuple(
            round(c * 255)
            for c in mcolors.to_rgb(mcolors.get_named_colors_mapping()[color])
        )

    else:
        cv2_color = tuple(
            round(c * 255)
            for c in (mcolors.to_rgb(mcolors.get_named_colors_mapping()["red"]))
        )
    return cv2_color

