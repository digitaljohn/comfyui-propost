from PIL import Image
import random
import numpy as np

def _makeGrayNoise(width_value, height_value, power_value):
    buffer = np.zeros([height_value, width_value], dtype=int)

    for y in range(0, height_value):
        for x in range(0, width_value):
            buffer[y, x] = random.gauss(128, power_value)
    buffer = buffer.clip(0, 255)
    return Image.fromarray(buffer.astype(dtype=np.uint8))

def _makeRgbNoise(width_param, height_param, noise_power, saturation):
    buffer = np.zeros([height_param, width_param, 3], dtype=int)
    intens_power = noise_power * (1.0 - saturation)
    for y in range(0, height_param):
        for x in range(0, width_param):
            intens = random.gauss(128, intens_power)
            buffer[y, x, 0] = random.gauss(0, noise_power) * saturation + intens
            buffer[y, x, 1] = random.gauss(0, noise_power) * saturation + intens
            buffer[y, x, 2] = random.gauss(0, noise_power) * saturation + intens

    buffer = buffer.clip(0, 255)
    return Image.fromarray(buffer.astype(dtype=np.uint8))


def grainGen(width_param, grain_height, grain_size_param, power_value, saturation, seed_value=1):
    # A grain_size of 1 means the noise buffer will be made 1:1
    # A grain_size of 2 means the noise buffer will be resampled 1:2
    noise_width = int(width_param / grain_size_param)
    noise_height = int(grain_height / grain_size_param)
    random.seed(seed_value)

    if saturation < 0.0:
        print("Making B/W grain, width: %d, height: %d, grain-size: %s, power: %s, seed: %d" % (
            noise_width, noise_height, str(grain_size_param), str(power_value), seed_value))
        img = _makeGrayNoise(noise_width, noise_height, power_value)
    else:
        print("Making RGB grain, width: %d, height: %d, saturation: %s, grain-size: %s, power: %s, seed: %d" % (
            noise_width, noise_height, str(saturation), str(grain_size_param), str(power_value), seed_value))
        img = _makeRgbNoise(noise_width, noise_height, power_value, saturation)

    # Resample
    if grain_size_param != 1.0:
        img = img.resize((width_param, grain_height), resample=Image.LANCZOS)

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