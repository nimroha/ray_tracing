import numpy as np
from PIL import Image

from functools import partial


norm2 = partial(np.linalg.norm, ord=2)

atol = 1e-08


def write_img(img, img_path):
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(img_path)


def is_close(a, b):
    return np.sum(np.abs(a-b)) < atol

def get_reflected_vector(d, n):
    # reflect_direction calculation: shading13.pdf - slide 41 "The Highlight Vector"
    r = d - 2 * np.dot(d, n) * n
    return r
