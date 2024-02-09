import cv2
import numpy as np

def generate_blurred_images(image, blur_strength, steps):
    blurred_images = []
    for step in range(1, steps + 1):
        blur_size = max(1, int(blur_strength * step / steps))
        blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1  # Ensure blur_size is odd
        
        blurred_image = cv2.GaussianBlur(image, (blur_size, blur_size), 0)
        lendblurred_image = lens_blur(blurred_image, blades_shape = 5, blades_radius = int(blur_size * 0.5), blades_rotation = 0, method = "dilate")
        blurred_images.append(lendblurred_image)
    return blurred_images

def lens_blur(image, blades_shape = 5, blades_radius = 10, blades_rotation = 0, method = "dilate"):
    angles = np.linspace(0, 2 * np.pi, blades_shape + 1)[:-1] + blades_rotation * np.pi / 180
    x = blades_radius * np.cos(angles) + blades_radius
    y = blades_radius * np.sin(angles) + blades_radius
    pts = np.stack([x, y], axis=1).astype(np.int32)

    mask = np.zeros((blades_radius * 2 + 1, blades_radius * 2 + 1), np.uint8)
    cv2.fillPoly(mask, [pts], 255)

    gaussian_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    if method == "dilate":
        kernel = cv2.filter2D(mask, -1, gaussian_kernel)
        result = cv2.dilate(image, kernel)
    elif method == "filter":
        height, width = image.shape[:2]
        dilate_size = min(height, width) // 512

        if dilate_size > 0:
            image = cv2.dilate(image, np.ones((dilate_size, dilate_size), np.uint8))

        kernel = mask.astype(np.float32) / np.sum(mask)
        kernel = cv2.filter2D(kernel, -1, gaussian_kernel)
        result = cv2.filter2D(image, -1, kernel)
    else:
        raise ValueError("Unsupported method.")

    return result

def apply_blurred_images(image, blurred_images, mask):
    steps = len(blurred_images)  # Calculate the number of steps based on the blurred images provided
    final_image = np.zeros_like(image)
    step_size = 1.0 / steps
    for i, blurred_image in enumerate(blurred_images):
        # Calculate the mask for the current step
        current_mask = np.clip((mask - i * step_size) * steps, 0, 1)
        next_mask = np.clip((mask - (i + 1) * step_size) * steps, 0, 1)
        blend_mask = current_mask - next_mask

        # Apply the blend mask
        final_image += blend_mask[:, :, np.newaxis] * blurred_image

    # Ensure no division by zero; add the original image for areas without blurring
    final_image += (1 - np.clip(mask * steps, 0, 1))[:, :, np.newaxis] * image
    return final_image