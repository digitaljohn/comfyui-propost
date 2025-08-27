import sys
import os
import cv2
import torch
import numpy as np
import folder_paths
from .utils import processing as processing_utils
from .utils import loading as loading_utils

# Get the directory of the current file and add it to the system path
current_file_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_directory)

from .filmgrainer import filmgrainer

# Create the directory for the LUTs
dir_luts = os.path.join(folder_paths.models_dir, "luts")
os.makedirs(dir_luts, exist_ok=True)
folder_paths.folder_names_and_paths["luts"] = ([dir_luts], set(['.cube']))


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
                "center_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()
 
    FUNCTION = "vignette_image"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "Pro Post/Camera Effects"
 
    def vignette_image(self, image: torch.Tensor, intensity: float, center_x: float, center_y: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        if intensity == 0:
            return image

        # Generate the vignette for each image in the batch
        # Create linear space but centered around the provided center point ratios
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1, 1, height)
        X, Y = np.meshgrid(x - (2 * center_x - 1), y - (2 * center_y - 1))
        
        # Calculate distances to the furthest corner
        distances_to_corners = [
            np.sqrt((0 - center_x) ** 2 + (0 - center_y) ** 2),
            np.sqrt((1 - center_x) ** 2 + (0 - center_y) ** 2),
            np.sqrt((0 - center_x) ** 2 + (1 - center_y) ** 2),
            np.sqrt((1 - center_x) ** 2 + (1 - center_y) ** 2)
        ]
        max_distance_to_corner = np.max(distances_to_corners)

        radius = np.sqrt(X ** 2 + Y ** 2)
        radius = radius / (max_distance_to_corner * np.sqrt(2))  # Normalize radius
        opacity = np.clip(intensity, 0, 1)
        vignette = 1 - radius * opacity

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            # Apply vignette
            vignette_image = self.apply_vignette(tensor_image, vignette)

            tensor = torch.from_numpy(vignette_image).unsqueeze(0)
            result[b] = tensor

        return (result,)

    def apply_vignette(self, image, vignette):
        # If image needs to be normalized (0-1 range)
        needs_normalization = image.max() > 1
        if needs_normalization:
            image = image.astype(np.float32) / 255
        
        final_image = np.clip(image * vignette[..., np.newaxis], 0, 1)
        
        if needs_normalization:
            final_image = (final_image * 255).astype(np.uint8)

        return final_image

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
                    "max": 0xffffffffffffffff
                }),
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()
 
    FUNCTION = "filmgrain_image"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "Pro Post/Camera Effects"
 
    def filmgrain_image(self, image: torch.Tensor, gray_scale: bool, grain_type: str, grain_sat: float, grain_power: float, shadows: float, highs: float, scale: float, sharpen: int, src_gamma: float, seed: int):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        # find index of grain_type
        grain_type_index = self.grain_types.index(grain_type) + 1;


        for b in range(batch_size):
            tensor_image = image[b].numpy()

            # Apply vignette
            vignette_image = self.apply_filmgrain(tensor_image, gray_scale, grain_type_index, grain_sat, grain_power, shadows, highs, scale, sharpen, src_gamma, seed+b)

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
                    "default": 64.0,
                    "min": 0.0,
                    "max": 256.0,
                    "step": 1.0
                }),
                "center_x": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "focus_spread": ("FLOAT", {
                    "default": 1,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.1
                }),
                "steps": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 32,
                }),
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()
 
    FUNCTION = "radialblur_image"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "Pro Post/Blur Effects"
 
    def radialblur_image(self, image: torch.Tensor, blur_strength: float, center_x: float, center_y:float, focus_spread:float, steps: int):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        # Generate the vignette for each image in the batch
        c_x, c_y = int(width * center_x), int(height * center_y)

        # Calculate distances to all corners from the center
        distances_to_corners = [
            np.sqrt((c_x - 0)**2 + (c_y - 0)**2),
            np.sqrt((c_x - width)**2 + (c_y - 0)**2),
            np.sqrt((c_x - 0)**2 + (c_y - height)**2),
            np.sqrt((c_x - width)**2 + (c_y - height)**2)
        ]
        max_distance_to_corner = max(distances_to_corners)

        # Create and adjust radial mask
        X, Y = np.meshgrid(np.arange(width) - c_x, np.arange(height) - c_y)
        radial_mask = np.sqrt(X**2 + Y**2) / max_distance_to_corner

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            # Apply blur
            blur_image = self.apply_radialblur(tensor_image, blur_strength, radial_mask, focus_spread, steps)

            tensor = torch.from_numpy(blur_image).unsqueeze(0)
            result[b] = tensor

        return (result,)

    def apply_radialblur(self, image, blur_strength, radial_mask, focus_spread, steps):
        needs_normalization = image.max() > 1
        if needs_normalization:
            image = image.astype(np.float32) / 255

        blurred_images = processing_utils.generate_blurred_images(image, blur_strength, steps, focus_spread)
        final_image = processing_utils.apply_blurred_images(image, blurred_images, radial_mask)

        if needs_normalization:
            final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)

        return np.clip(final_image, 0, 1)


class ProPostDepthMapBlur:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "blur_strength": ("FLOAT", {
                    "default": 64.0,
                    "min": 0.0,
                    "max": 256.0,
                    "step": 1.0
                }),
                "focal_depth": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "focus_spread": ("FLOAT", {
                    "default": 1,
                    "min": 1.0,
                    "max": 8.0,
                    "step": 0.1
                }),
                "steps": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 32,
                }),
                "focal_range": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "mask_blur": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 127,
                    "step": 2
                }),
            },
        }
 
    RETURN_TYPES = ("IMAGE","MASK")
    RETURN_NAMES = ()
 
    FUNCTION = "depthblur_image"
    DESCRIPTION = """
    blur_strength: Represents the blur strength. This parameter controls the overall intensity of the blur effect; the higher the value, the more blurred the image becomes.

    focal_depth: Represents the focal depth. This parameter is used to determine which depth level in the image should remain sharp, while other levels are blurred based on depth differences.

    focus_spread: Represents the focus spread range. This parameter controls the size of the blur transition area near the focal depth; the larger the value, the wider the transition area, and the smoother the blur effect spreads around the focus.

    steps: Represents the number of steps in the blur process. This parameter determines the calculation precision of the blur effect; the more steps, the finer the blur effect, but this also increases the computational load.

    focal_range: Represents the focal range. This parameter is used to adjust the depth range within the focal depth that remains sharp; the larger the value, the wider the area around the focal depth that remains sharp.
    
    mask_blur: Represents the mask blur strength for blurring the depth map. This parameter controls the intensity of the depth map's blur treatment, used for preprocessing the depth map before calculating the final blur effect, to achieve a more natural blur transition.
    """
    #OUTPUT_NODE = False
 
    CATEGORY = "Pro Post/Blur Effects"
 
    def depthblur_image(self, image: torch.Tensor, depth_map: torch.Tensor, blur_strength: float, focal_depth: float, focus_spread:float, steps: int, focal_range: float, mask_blur: int):
        batch_size, height, width, _ = image.shape
        image_result = torch.zeros_like(image)
        mask_result = torch.zeros((batch_size, height, width), dtype=torch.float32)

        for b in range(batch_size):
            tensor_image = image[b].numpy()
            tensor_image_depth = depth_map[b].numpy()

            # Apply blur
            blur_image,depth_mask = self.apply_depthblur(tensor_image, tensor_image_depth, blur_strength, focal_depth, focus_spread, steps, focal_range, mask_blur)

            tensor_image = torch.from_numpy(blur_image).unsqueeze(0)
            tensor_mask = torch.from_numpy(depth_mask).unsqueeze(0)

            image_result[b] = tensor_image
            mask_result[b] = tensor_mask

        return (image_result,mask_result)

    def apply_depthblur(self, image, depth_map, blur_strength, focal_depth, focus_spread, steps, focal_range, mask_blur):
        # Normalize the input image if needed
        needs_normalization = image.max() > 1
        if needs_normalization:
            image = image.astype(np.float32) / 255

        # Normalize the depth map if needed
        depth_map = depth_map.astype(np.float32) / 255 if depth_map.max() > 1 else depth_map

        # Resize depth map to match the image dimensions
        depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
        if len(depth_map_resized.shape) > 2:
            depth_map_resized = cv2.cvtColor(depth_map_resized, cv2.COLOR_BGR2GRAY)

        # Adjust the depth map based on the focal plane
        depth_mask = np.abs(depth_map_resized - focal_depth)
        depth_mask = np.clip(depth_mask / np.max(depth_mask), 0, 1)

        # Process the depth_mask
        depth_mask[depth_mask < focal_range] = 0
        depth_mask[depth_mask >= focal_range] = (depth_mask[depth_mask >= focal_range] - focal_range) / (1 - focal_range)

        # Apply mask blur
        depth_mask = cv2.GaussianBlur(depth_mask, (mask_blur, mask_blur), 0)

        # Generate blurred versions of the image
        blurred_images = processing_utils.generate_blurred_images(image, blur_strength, steps, focus_spread)

        # Use the adjusted depth map as a mask for applying blurred images
        final_image = processing_utils.apply_blurred_images(image, blurred_images, depth_mask)

        # Convert back to original range if the image was normalized
        if needs_normalization:
            final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)

        return final_image, depth_mask


class ProPostApplyLUT:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_name": (folder_paths.get_filename_list("luts"), ),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "log": ("BOOLEAN", {
                    "default": False
                }),
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()
 
    FUNCTION = "lut_image"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "Pro Post/Color Grading"
 
    def lut_image(self, image: torch.Tensor, lut_name, strength: float, log: bool):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        # Read the LUT
        lut_path = os.path.join(dir_luts, lut_name)
        lut = loading_utils.read_lut(lut_path, clip=True)

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            # Apply LUT
            lut_image = self.apply_lut(tensor_image, lut, strength, log)

            tensor = torch.from_numpy(lut_image).unsqueeze(0)
            result[b] = tensor

        return (result,)

    def apply_lut(self, image, lut, strength, log):
        if strength == 0:
            return image

        # Apply the LUT
        is_non_default_domain = not np.array_equal(lut.domain, np.array([[0., 0., 0.], [1., 1., 1.]]))
        dom_scale = None

        im_array = image.copy()

        if is_non_default_domain:
            dom_scale = lut.domain[1] - lut.domain[0]
            im_array = im_array * dom_scale + lut.domain[0]
        if log:
            im_array = im_array ** (1/2.2)

        im_array = lut.apply(im_array)

        if log:
            im_array = im_array ** (2.2)
        if is_non_default_domain:
            im_array = (im_array - lut.domain[0]) / dom_scale

        # Blend the original image and the LUT-applied image based on the strength
        blended_image = (1 - strength) * image + strength * im_array

        return blended_image
 
 
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ProPostVignette": ProPostVignette,
    "ProPostFilmGrain": ProPostFilmGrain,
    "ProPostRadialBlur": ProPostRadialBlur,
    "ProPostDepthMapBlur": ProPostDepthMapBlur,
    "ProPostApplyLUT": ProPostApplyLUT
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProPostVignette": "ProPost Vignette",
    "ProPostFilmGrain": "ProPost Film Grain",
    "ProPostRadialBlur": "ProPost Radial Blur",
    "ProPostDepthMapBlur": "ProPost Depth Map Blur",
    "ProPostApplyLUT": "ProPost Apply LUT"
}