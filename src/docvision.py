import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def distort_gs(image_path, output_path):
    """Applies a subtle scanned document effect using OpenCV with mild distortion."""
    
    # Load image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply a very light Gaussian Blur (softens without destroying text clarity)
    blurred = cv2.GaussianBlur(image, (3, 3), 1)

    # Slight Perspective Warping to simulate scanning misalignment
    rows, cols = blurred.shape
    src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    dst_points = np.float32([[3, 2], [cols-5, 2], [2, rows-3], [cols-2, rows-4]])  # Less extreme warp

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(blurred, matrix, (cols, rows))

    # Add very mild noise (to simulate light scan grain)
    noise = np.random.normal(0, 8, warped.shape).astype(np.uint8)  # Lower noise level
    noisy_image = cv2.add(warped, noise)

    # Light contrast adjustment to maintain readability
    alpha = 1.2  # Mild contrast control
    beta = 5     # Slight brightness boost
    scanned_effect = cv2.convertScaleAbs(noisy_image, alpha=alpha, beta=beta)

    # Save the final processed image
    cv2.imwrite(output_path, scanned_effect)
    
    return output_path

def distort_cv(image_path):
    """Applies a moderate scanned document effect with salt & pepper noise while keeping table lines visible."""
    
    # Load image in original color format
    image = cv2.imread(image_path)

    # Apply a milder Gaussian Blur (reducing intensity to keep table lines visible)
    blurred = cv2.GaussianBlur(image, (3, 3), 0.5)

    # Randomized Light Perspective Warping (reducing intensity)
    rows, cols, _ = blurred.shape
    distortion_range = random.uniform(1, 3)  # Lower distortion

    src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    dst_points = np.float32([
        [distortion_range, distortion_range], 
        [cols-distortion_range-1, distortion_range], 
        [distortion_range, rows-distortion_range-1], 
        [cols-distortion_range-1, rows-distortion_range-1]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(blurred, matrix, (cols, rows))

    # Add random noise with reduced intensity
    noise_intensity = random.randint(3, 8)  # Lower noise strength
    noise = np.random.normal(0, noise_intensity, warped.shape).astype(np.uint8)
    noisy_image = cv2.addWeighted(warped, 0.98, noise, 0.02, 0)  # 2% noise blending

     # Introduce random local warping to distort edges slightly
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows), indexing="xy")
    displacement = random.randint(1, 2)
    map_x = (map_x + np.sin(map_y / 30) * displacement).astype(np.float32)
    map_y = (map_y + np.sin(map_x / 30) * displacement).astype(np.float32)
    warped = cv2.remap(warped, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Add moderate salt-and-pepper noise
    salt_pepper_ratio = 0.002  # Less aggressive
    num_salt = int(salt_pepper_ratio * image.size * 0.5)
    num_pepper = int(salt_pepper_ratio * image.size * 0.5)

    # Add Salt (White Pixels)
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    noisy_image[salt_coords[0], salt_coords[1]] = [228, 241, 248]  # Softer white pixels

    # # Add Pepper (Black Pixels)
    # pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    # noisy_image[pepper_coords[0], pepper_coords[1]] = [20, 20, 20]  # Softer black pixels

    # Reduce contrast & brightness adjustments (to avoid excessive fading)
    alpha = 1.05  # Lower contrast boost
    beta = 3  # Minimal brightness shift
    scanned_effect = cv2.convertScaleAbs(noisy_image, alpha=alpha, beta=beta)

    cv2.imwrite(image_path, scanned_effect)
    return image_path


def visualize_grounding(image_path, grounding):
    # Load the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for proper display
    h, w, _ = image.shape
    
    # Create a figure
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    for item in grounding:
        box = item['box']
        type = item['type']

        color = {"text": "green", "table": "blue", "chart": "red"}
        
        # Convert normalized coordinates to pixel values
        l, t, r, b = box['l'] * w, box['t'] * h, box['r'] * w, box['b'] * h
        
        # Draw the rectangle
        plt.gca().add_patch(plt.Rectangle((l, t), r - l, b - t, 
                                          edgecolor=color[type], linewidth=1, fill=False))
        
        # Put text label
        plt.text(l, t - 5, type[:30] + ('...' if len(type) > 30 else ''), 
                 color=color[type], fontsize=8)
    
    plt.axis("off")
    plt.show()


