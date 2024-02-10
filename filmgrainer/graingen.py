from PIL import Image
import random
import numpy as np
import pyfastnoisesimd as fns
import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message=".*`product` is deprecated.*")

def _makeGrayNoise(width, height, power, seed=0):
    # Setup noise type and parameters
    noise = fns.Noise(seed=seed)
    noise.noise_type = fns.NoiseType.Perlin
    noise.frequency = 1.0  # Adjust frequency for different scales of noise

    # Generate noise
    buffer = noise.genAsGrid(shape=(height, width))

    # Scale noise by power before normalization
    buffer *= power

    # Normalize and scale to 0-255 range
    buffer = (buffer - buffer.min()) / (buffer.max() - buffer.min()) * 255
    return Image.fromarray(buffer.astype(np.uint8))


def _makeRgbNoise(width, height, power, saturation=0.5, seed=0):
    # Setup noise type and parameters
    noise = fns.Noise(seed=seed)
    noise.noise_type = fns.NoiseType.Perlin
    noise.frequency = 1.0  # Adjust frequency for different scales of noise
    
    # Initialize RGB buffer
    rgb_buffer = np.zeros((height, width, 3), dtype=np.float32)

    # Generate base intensity noise and scale by power
    intens_buffer = noise.genAsGrid(shape=(height, width)) * power

    # Normalize intensity noise
    intens_buffer = (intens_buffer - intens_buffer.min()) / (intens_buffer.max() - intens_buffer.min()) * 255
    
    for i in range(3):
        noise.seed += 1  # Change seed for each channel to get different noise
        color_buffer = noise.genAsGrid(shape=(height, width)) * power
        color_buffer = (color_buffer - color_buffer.min()) / (color_buffer.max() - color_buffer.min()) * 255
        # Blend intensity and color noise based on saturation
        rgb_buffer[..., i] = intens_buffer * (1 - saturation) + color_buffer * saturation

    rgb_buffer = np.clip(rgb_buffer, 0, 255)  # Ensure values are in byte range
    return Image.fromarray(rgb_buffer.astype(np.uint8))


def grainGen(width, height, grain_size, power, saturation, seed = 1):
    # A grain_size of 1 means the noise buffer will be made 1:1
    # A grain_size of 2 means the noise buffer will be resampled 1:2
    noise_width = int(width / grain_size)
    noise_height = int(height / grain_size)
    random.seed(seed)

    if saturation < 0.0:
        print("Making B/W grain, width: %d, height: %d, grain-size: %s, power: %s, seed: %d" % (
            noise_width, noise_height, str(grain_size), str(power), seed))
        img = _makeGrayNoise(noise_width, noise_height, power, seed)
    else:
        print("Making RGB grain, width: %d, height: %d, saturation: %s, grain-size: %s, power: %s, seed: %d" % (
            noise_width, noise_height, str(saturation), str(grain_size), str(power), seed))
        img = _makeRgbNoise(noise_width, noise_height, power, saturation, seed)

    # Resample
    if grain_size != 1.0:
        img = img.resize((width, height), resample = Image.LANCZOS)

    return img


if __name__ == "__main__":
    import sys
    import time  # Import the time module

    if len(sys.argv) == 8:
        width = int(sys.argv[2])
        height = int(sys.argv[3])
        grain_size = float(sys.argv[4])
        power = float(sys.argv[5])
        sat = float(sys.argv[6])
        seed = int(sys.argv[7])

        start_time = time.time()  # Start timing

        # Generate the noise image
        out = grainGen(width, height, grain_size, power, sat, seed)

        # Save the generated image
        out.save(sys.argv[1])

        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time  # Calculate elapsed time

        print(f"Image generated and saved in {elapsed_time} seconds.")  # Print out the elapsed time