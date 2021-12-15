import random
import io
from PIL import Image

import numpy as np
import torch


def print_args(args):
    print(' ' * 26 + 'Options')
    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))


def set_global_seeds(seed, use_torch=False):
    if use_torch:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)


def array2png(arr):
    rgb_image = Image.fromarray(arr)
    output_io = io.BytesIO()
    rgb_image.save(output_io, format="PNG")
    png_image = output_io.getvalue()
    output_io.close()
    rgb_image.close()
    return png_image


def png2array(png):
    png_io = io.BytesIO(png)
    png_image = Image.open(png_io)
    rgb_array = np.array(png_image)
    png_image.close()
    png_io.close()
    return rgb_array

