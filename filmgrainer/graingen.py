from PIL import Image
import random
import numpy as np
import pyfastnoisesimd as fns
import warnings

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message=".*`product` is deprecated.*")

def pad_dimensions_to_simd(width, height, simd_width=16):
    """
    Pad the dimensions to be divisible by a given SIMD width.

    Parameters:
    - width: The original width of the image.
    - height: The original height of the image.
    - simd_width: The SIMD width to pad dimensions to (default 16).

    Returns:
    - padded_width: The padded width.
    - padded_height: The padded height.
    """
    padded_width = ((width + simd_width - 1) // simd_width) * simd_width
    padded_height = ((height + simd_width - 1) // simd_width) * simd_width
    return padded_width, padded_height

def _makeGrayNoise(width, height, power, seed=0):
    # Pad dimensions for SIMD compatibility
    padded_width, padded_height = pad_dimensions_to_simd(width, height)

    # Setup noise type and parameters
    noise = fns.Noise(seed=seed)
    noise.noise_type = fns.NoiseType.Perlin
    noise.frequency = 1.0  # Adjust frequency for different scales of noise

    # Generate noise with padded dimensions
    padded_buffer = noise.genAsGrid(shape=(padded_height, padded_width))

    # Scale noise by power before normalization
    padded_buffer *= power

    # Crop the padded buffer to the original dimensions if necessary
    buffer = padded_buffer[:height, :width]

    # Normalize and scale to 0-255 range
    buffer = (buffer - buffer.min()) / (buffer.max() - buffer.min()) * 255
    return Image.fromarray(buffer.astype(np.uint8))


def _makeRgbNoise(width, height, power, saturation=0.5, seed=0):
    # Pad dimensions for SIMD compatibility
    padded_width, padded_height = pad_dimensions_to_simd(width, height)

    # Setup noise type and parameters
    noise = fns.Noise(seed=seed)
    noise.noise_type = fns.NoiseType.Perlin
    noise.frequency = 1.0  # Adjust frequency for different scales of noise
    
    # Initialize padded RGB buffer
    padded_rgb_buffer = np.zeros((padded_height, padded_width, 3), dtype=np.float32)

    # Generate base intensity noise with padded dimensions and scale by power
    padded_intens_buffer = noise.genAsGrid(shape=(padded_height, padded_width)) * power

    # Normalize padded intensity noise
    padded_intens_buffer = (padded_intens_buffer - padded_intens_buffer.min()) / (padded_intens_buffer.max() - padded_intens_buffer.min()) * 255
    
    for i in range(3):
        noise.seed += 1  # Change seed for each channel to get different noise
        padded_color_buffer = noise.genAsGrid(shape=(padded_height, padded_width)) * power
        padded_color_buffer = (padded_color_buffer - padded_color_buffer.min()) / (padded_color_buffer.max() - padded_color_buffer.min()) * 255
        # Blend intensity and color noise based on saturation
        padded_rgb_buffer[..., i] = padded_intens_buffer * (1 - saturation) + padded_color_buffer * saturation

    # Crop the padded RGB buffer to the original dimensions
    rgb_buffer = padded_rgb_buffer[:height, :width, :]

    rgb_buffer = np.clip(rgb_buffer, 0, 255)  # Ensure values are in byte range
    return Image.fromarray(rgb_buffer.astype(np.uint8))


def grainGen(width, height, grain_size, power, saturation, seed = 1):
    # A grain_size of 1 means the noise buffer will be made 1:1
    # A grain_size of 2 means the noise buffer will be resampled 1:2
    noise_width = int(width / grain_size)
    noise_height = int(height / grain_size)
    random.seed(seed)

    if saturation < 0.0:
        # print("Making B/W grain, width: %d, height: %d, grain-size: %s, power: %s, seed: %d" % (
        #     noise_width, noise_height, str(grain_size), str(power), seed))
        img = _makeGrayNoise(noise_width, noise_height, power, seed)
    else:
        # print("Making RGB grain, width: %d, height: %d, saturation: %s, grain-size: %s, power: %s, seed: %d" % (
        #     noise_width, noise_height, str(saturation), str(grain_size), str(power), seed))
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