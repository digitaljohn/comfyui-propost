from PIL import Image
import random
import torch
from torchvision.transforms import ToPILImage
import time

def _makeGrayNoise(width, height, power, generator=None):
    # Generate Gaussian noise with mean 128 and standard deviation = power

    # Ensure a generator is used if provided
    if generator is None:
        generator = torch.Generator()

    # Note: torch.randn_like is not used here because we're creating the tensor from scratch
    noise = torch.normal(mean=128, std=power, size=(height, width)).clamp(0, 255)

    # Convert to uint8, since we're simulating an 8-bit grayscale image
    noise = noise.to(torch.uint8)

    # Convert the tensor to a PIL image for compatibility with many image processing libraries
    to_pil_image = ToPILImage()
    noise_image = to_pil_image(noise.unsqueeze(0))  # Add a channel dimension

    return noise_image

def _makeRgbNoise(width, height, power, saturation, generator=None):
    # Ensure a generator is used if provided
    if generator is None:
        generator = torch.Generator()

    # Initialize the noise tensor for RGB channels
    noise_base = torch.normal(mean=128, std=power * (1.0 - saturation), size=(height, width, 1))
    noise_color = torch.normal(mean=0, std=power * saturation, size=(height, width, 3))
    
    # Combine base intensity and color noise, ensuring values are within byte range
    noise = (noise_base + noise_color).clamp(0, 255)
    
    # Convert the noise tensor to uint8 since we're simulating an 8-bit color image
    noise = noise.to(torch.uint8)

    # Convert the tensor to a PIL image
    to_pil_image = ToPILImage()
    noise_image = to_pil_image(noise.permute(2, 0, 1))  # Reorder dimensions to CxHxW for PIL

    return noise_image


def grainGen(width, height, grain_size, power, saturation, seed=1):
    start_time = time.time()  # Start time of the function execution

    # Create a PyTorch generator with the specified seed
    generator = torch.Generator().manual_seed(seed)
    
    # A grain_size of 1 means the noise buffer will be made 1:1
    # A grain_size of 2 means the noise buffer will be resampled 1:2
    noise_width = int(width / grain_size)
    noise_height = int(height / grain_size)
    random.seed(seed)

    if saturation < 0.0:
        print("Making B/W grain, width: %d, height: %d, grain-size: %s, power: %s, seed: %d" % (
            noise_width, noise_height, str(grain_size), str(power), seed))
        img = _makeGrayNoise(noise_width, noise_height, power, generator)
    else:
        print("Making RGB grain, width: %d, height: %d, saturation: %s, grain-size: %s, power: %s, seed: %d" % (
            noise_width, noise_height, str(saturation), str(grain_size), str(power), seed))
        img = _makeRgbNoise(noise_width, noise_height, power, saturation, generator)

    # Resample
    if grain_size != 1:
        img = img.resize((width, height), resample=Image.LANCZOS)
    
    end_time = time.time()  # End time of the function execution
    print("Execution Time: {:.2f} seconds".format(end_time - start_time))

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