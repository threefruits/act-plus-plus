import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def augment_image(image):
    """
    Apply random data augmentations to the input image including:
    - Adding light/dark regions
    - Adjusting the light and dark of the whole image
    - Adding noise
    - Adjusting contrast

    Parameters:
    image (numpy.ndarray): Input image

    Returns:
    numpy.ndarray: Augmented image
    """
    # Add light/dark regions
    def add_light_dark_regions(img):
        h, w = img.shape[:2]
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        radius = np.random.randint(min(h, w) // 4, min(h, w) // 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (x, y), radius, (255), -1)
        mask = cv2.GaussianBlur(mask, (31, 31), 0)  # Apply Gaussian blur for smooth transition
        intensity = np.random.randint(-15, 15)
        mask = mask[:, :, np.newaxis]
        img = np.clip(img + intensity * (mask / 255.0), 0, 255).astype(np.uint8)
        return img

    # Adjust the light and dark of the whole image
    def adjust_light_dark(img):
        factor = 1 + (np.random.rand() - 0.5) * 2 * 0.2  # Adjusting by ±30%
        return np.clip(img * factor, 0, 255).astype(np.uint8)

    # Add noise
    def add_noise(img):
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        return np.clip(img + noise, 0, 255).astype(np.uint8)

    # Adjust contrast
    def adjust_contrast(img):
        factor = 1 + (np.random.rand() - 0.5) * 2 * 0.2  # Adjusting by ±20%
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        return np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)

    # Apply the augmentations
    image = add_light_dark_regions(image)
    image = adjust_light_dark(image)
    image = add_noise(image)
    image = adjust_contrast(image)

    return image

if __name__ == '__main__':
    # Read the sample image
    image_path = '/data/home/share/fetch_data/pre_open_fridge/episode_0/rgb_145.jpg'
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} not found.")
    # Convert to RGB for display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Directory to save augmented images
    output_dir = './augmented_images'
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save 10 different augmented images
    for i in range(10):
        augmented_image = augment_image(image_rgb)
        augmented_image_bgr = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        output_path = os.path.join(output_dir, f'augmented_image_{i+1}.jpg')
        cv2.imwrite(output_path, augmented_image_bgr)

    print("Augmented images have been saved in the 'augmented_images' directory.")
