import random

import numpy as np
from PIL import Image


def _makeGrayNoise(width: int, height: int, power: float) -> Image.Image:
    buffer = np.zeros([height, width], dtype=int)

    for y in range(0, height):
        for x in range(0, width):
            buffer[y, x] = random.gauss(128, power)
    buffer = buffer.clip(0, 255)
    return Image.fromarray(buffer.astype(dtype=np.uint8))


def _makeRgbNoise(width: int, height: int, power: float, saturation: float) -> Image.Image:
    buffer = np.zeros([height, width, 3], dtype=int)
    intens_power = power * (1.0 - saturation)
    for y in range(0, height):
        for x in range(0, width):
            intens = random.gauss(128, intens_power)
            buffer[y, x, 0] = random.gauss(0, power) * saturation + intens
            buffer[y, x, 1] = random.gauss(0, power) * saturation + intens
            buffer[y, x, 2] = random.gauss(0, power) * saturation + intens

    buffer = buffer.clip(0, 255)
    return Image.fromarray(buffer.astype(dtype=np.uint8))


def grainGen(
    width: int,
    height: int,
    grain_size: float,
    power: float,
    saturation: float,
    seed: int = 1,
) -> Image.Image:
    # A grain_size of 1 means the noise buffer will be made 1:1
    # A grain_size of 2 means the noise buffer will be resampled 1:2
    noise_width = int(width / grain_size)
    noise_height = int(height / grain_size)
    random.seed(seed)

    if saturation < 0.0:
        img = _makeGrayNoise(noise_width, noise_height, power)
    else:
        img = _makeRgbNoise(noise_width, noise_height, power, saturation)

    # Resample
    if grain_size != 1.0:
        img = img.resize((width, height), resample=Image.LANCZOS)

    return img


if __name__ == "__main__":
    import sys

    if len(sys.argv) == 8:
        width = int(sys.argv[2])
        height = int(sys.argv[3])
        grain_size = float(sys.argv[4])
        power = float(sys.argv[5])
        sat = float(sys.argv[6])
        seed = int(sys.argv[7])
        out = grainGen(width, height, grain_size, power, sat, seed)
        out.save(sys.argv[1])
