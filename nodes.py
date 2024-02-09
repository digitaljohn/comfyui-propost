import sys
import os
import cv2
import torch
import numpy as np
import folder_paths
from colour.io.luts.iridas_cube import read_LUT_IridasCube, LUT3D, LUT3x1D
from .utils import processing as processing_utils

# Get the directory of the current file and add it to the system path
current_file_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_file_directory)

import filmgrainer.filmgrainer as filmgrainer

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
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()
 
    FUNCTION = "vignette_image"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "Pro Post/Camera Effects"
 
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
 
    CATEGORY = "Pro Post/Camera Effects"
 
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
                    "default": 64.0,
                    "min": 0.0,
                    "max": 256.0,
                    "step": 1.0
                }),
                "center_focus_weight": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
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
 
    def radialblur_image(self, image: torch.Tensor, blur_strength: float, center_focus_weight: float, steps: int):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            # Apply blur
            blur_image = self.apply_radialblur(tensor_image, blur_strength, center_focus_weight, steps)

            tensor = torch.from_numpy(blur_image).unsqueeze(0)
            result[b] = tensor

        return (result,)

    def apply_radialblur(self, image, blur_strength, center_focus_weight, steps):
        needs_normalization = image.max() > 1
        if needs_normalization:
            image = image.astype(np.float32) / 255
        
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2

        # Create and adjust radial mask
        X, Y = np.meshgrid(np.arange(width) - center_x, np.arange(height) - center_y)
        radial_mask = np.sqrt(X**2 + Y**2) / np.sqrt(center_x**2 + center_y**2)
        radial_mask = np.power(radial_mask, center_focus_weight)

        blurred_images = processing_utils.generate_blurred_images(image, blur_strength, steps)
        final_image = processing_utils.apply_blurred_images(image, blurred_images, radial_mask)

        if needs_normalization:
            final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)

        return final_image


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
                "steps": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 32,
                }),
                "focal_depth": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
            },
        }
 
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()
 
    FUNCTION = "depthblur_image"
 
    #OUTPUT_NODE = False
 
    CATEGORY = "Pro Post/Blur Effects"
 
    def depthblur_image(self, image: torch.Tensor, depth_map: torch.Tensor, blur_strength: float, steps: int, focal_depth: float):
        batch_size, height, width, _ = image.shape
        result = torch.zeros_like(image)

        for b in range(batch_size):
            tensor_image = image[b].numpy()
            tensor_image_depth = depth_map[b].numpy()

            # Apply blur
            blur_image = self.apply_depthblur(tensor_image, tensor_image_depth, blur_strength, steps, focal_depth)

            tensor = torch.from_numpy(blur_image).unsqueeze(0)
            result[b] = tensor

        return (result,)

    def apply_depthblur(self, image, depth_map, blur_strength, steps, focal_depth):
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

        # Generate blurred versions of the image
        blurred_images = processing_utils.generate_blurred_images(image, blur_strength, steps)

        # Use the adjusted depth map as a mask for applying blurred images
        final_image = processing_utils.apply_blurred_images(image, blurred_images, depth_mask)

        # Convert back to original range if the image was normalized
        if needs_normalization:
            final_image = np.clip(final_image * 255, 0, 255).astype(np.uint8)

        return final_image


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

        for b in range(batch_size):
            tensor_image = image[b].numpy()

            # Apply LUT
            lut_image = self.apply_lut(tensor_image, lut_name, strength, log)

            tensor = torch.from_numpy(lut_image).unsqueeze(0)
            result[b] = tensor

        return (result,)

    def apply_lut(self, image, lut_name, strength, log):
        if strength == 0:
            return image

        # Read the LUT
        lut_path = os.path.join(dir_luts, lut_name)
        lut = self.read_lut(lut_path, clip=True)

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

    def read_lut(self, lut_path, clip=False):
        """
        Reads a LUT from the specified path, returning instance of LUT3D or LUT3x1D

        <lut_path>: the path to the file from which to read the LUT (
        <clip>: flag indicating whether to apply clipping of LUT values, limiting all values to the domain's lower and
            upper bounds
        """
        lut: Union[LUT3x1D, LUT3D] = read_LUT_IridasCube(lut_path)
        lut.name = os.path.splitext(os.path.basename(lut_path))[0]  # use base filename instead of internal LUT name

        if clip:
            if lut.domain[0].max() == lut.domain[0].min() and lut.domain[1].max() == lut.domain[1].min():
                lut.table = np.clip(lut.table, lut.domain[0, 0], lut.domain[1, 0])
            else:
                if len(lut.table.shape) == 2:  # 3x1D
                    for dim in range(3):
                        lut.table[:, dim] = np.clip(lut.table[:, dim], lut.domain[0, dim], lut.domain[1, dim])
                else:  # 3D
                    for dim in range(3):
                        lut.table[:, :, :, dim] = np.clip(lut.table[:, :, :, dim], lut.domain[0, dim], lut.domain[1, dim])

        return lut
 
 
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