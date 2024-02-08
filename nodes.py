import sys
import os
import torch
import numpy as np

# Get the directory of the current file
current_file_directory = os.path.dirname(os.path.abspath(__file__))

# Append this directory to sys.path
sys.path.append(current_file_directory)

import filmgrainer.filmgrainer as filmgrainer

class ProPostVignette:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()
 
    FUNCTION = "vignette_image"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "propost/Effects"
 
    def vignette_image(self, image: torch.Tensor, intensity: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            # Apply vignette
            vignette_image = self.apply_vignette(tensor_image, intensity)

            tensor = torch.from_numpy(vignette_image).unsqueeze(0)
            result[b] = tensor

        return (result,)

    def apply_vignette(self, image, vignette_strength):
        if vignette_strength == 0:
            return image

        height, width, _ = image.shape
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        radius = np.sqrt(X ** 2 + Y ** 2)

        radius = radius / np.max(radius)
        opacity = np.clip(vignette_strength, 0, 1)
        vignette = 1 - radius * opacity

        return np.clip(image * vignette[..., np.newaxis], 0, 1)

class ProPostFilmGrain:
    grain_types = ["Fine", "Fine Simple", "Coarse", "Coarser"]

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "gray_scale": ("BOOLEAN", {
                    "default": False
                }),
                "grain_type": (s.grain_types,),
                "grain_sat": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "grain_power": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "shadows": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "highs": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "sharpen": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10
                }),
                "src_gamma": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "seed": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1000
                }),
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()
 
    FUNCTION = "filmgrain_image"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "propost/Film Grain"
 
    def filmgrain_image(self, image: torch.Tensor, gray_scale: bool, grain_type: str, grain_sat: float, grain_power: float, shadows: float, highs: float, scale: float, sharpen: int, src_gamma: float, seed: int):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        # find index of grain_type
        grain_type_index = self.grain_types.index(grain_type) + 1;


        for b in range(batch_size):
            tensor_image = image[b].numpy()

            # Apply vignette
            vignette_image = self.apply_filmgrain(tensor_image, gray_scale, grain_type_index, grain_sat, grain_power, shadows, highs, scale, sharpen, src_gamma, seed)

            tensor = torch.from_numpy(vignette_image).unsqueeze(0)
            result[b] = tensor

        return (result,)

    def apply_filmgrain(self, image, gray_scale, grain_type, grain_sat, grain_power, shadows, highs, scale, sharpen, src_gamma, seed):
        out_image = filmgrainer.process(image, scale, src_gamma, 
            grain_power, shadows, highs, grain_type, 
            grain_sat, gray_scale, sharpen, seed)
        return out_image


class ProPostRadialBlur:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "blur_strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
                "max_blur": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01
                }),
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()
 
    FUNCTION = "radialblur_image"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "propost/Radial Blur"
 
    def radialblur_image(self, image: torch.Tensor, blur_strength: float, max_blur: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            # Apply blur
            blur_image = self.apply_radialblur(tensor_image, blur_strength, max_blur)

            tensor = torch.from_numpy(blur_image).unsqueeze(0)
            result[b] = tensor

        return (result,)

    def apply_radialblur(self, image, blur_strength, max_blur):
        if blur_strength == 0:
            return image

        height, width = image.shape[:2]
        
        # If no center is provided, use the center of the image
        if center is None:
            center_x, center_y = width // 2, height // 2
        else:
            center_x, center_y = center

        # Create a radius map from the center of the image
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x, y)
        distance_from_center = np.sqrt((X - center_x/width*2) ** 2 + (Y - center_y/height*2) ** 2)
        
        # Normalize the distance to have a maximum value of 1
        normalized_distance = np.clip(distance_from_center / np.max(distance_from_center), 0, 1)
        
        # Adjust max_blur based on blur_strength
        adjusted_max_blur = max_blur * (blur_strength / 10)
        
        # Calculate the blur amount at each point (linearly scales with distance)
        blur_amount = normalized_distance * adjusted_max_blur
        
        # Create an empty array to store the result
        blurred_image = np.zeros_like(image)
        
        # Apply varying blur across the image based on the calculated blur_amount
        for y in range(height):
            for x in range(width):
                kernel_size = int(blur_amount[y, x])
                if kernel_size % 2 == 0: kernel_size += 1  # Ensure kernel size is odd
                if kernel_size > 1:
                    region = cv2.getRectSubPix(image, (kernel_size, kernel_size), (x, y))
                    blurred_image[y, x] = cv2.blur(region, (kernel_size, kernel_size))[kernel_size // 2, kernel_size // 2]
                else:
                    blurred_image[y, x] = image[y, x]
        
        return blurred_image
 
 
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ProPostVignette": ProPostVignette,
    "ProPostFilmGrain": ProPostFilmGrain,
    "ProPostRadialBlur": ProPostRadialBlur
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProPostVignette": "ProPost Vignette",
    "ProPostFilmGrain": "ProPost Film Grain",
    "ProPostRadialBlur": "ProPost Radial Blur"
}