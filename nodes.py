import torch
import numpy as np

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
 
 
# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "ProPostVignette": ProPostVignette
}
 
# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "ProPostVignette": "ProPost Vignette"
}