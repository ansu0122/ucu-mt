import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def distort_cv2(image_path):
    """Applies a moderate scanned document effect—balancing distortion, noise, and readability."""

    image = cv2.imread(image_path)

    blurred = cv2.GaussianBlur(image, (3, 3), 0.5)

    rows, cols, _ = blurred.shape
    distortion_range = random.uniform(2, 5)

    src_points = np.float32([[0, 0], [cols-1, 0], [0, rows-1], [cols-1, rows-1]])
    dst_points = np.float32([
        [distortion_range, distortion_range], 
        [cols-distortion_range-1, distortion_range], 
        [distortion_range, rows-distortion_range-1], 
        [cols-distortion_range-1, rows-distortion_range-1]
    ])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(blurred, matrix, (cols, rows))

    # add moderate noise (grain effect)
    noise_intensity = random.randint(3, 8)  # Lower noise strength
    noise = np.random.normal(0, noise_intensity, warped.shape).astype(np.uint8)
    noisy_image = cv2.addWeighted(warped, 0.98, noise, 0.02, 0)

    # local warping for an authentic scan effect
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows), indexing="xy")
    displacement = random.randint(1, 2)  # Moderate local distortion
    map_x = (map_x + np.sin(map_y / 60) * displacement).astype(np.float32)
    map_y = (map_y + np.sin(map_x / 60) * displacement).astype(np.float32)
    warped = cv2.remap(noisy_image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Add slight salt-and-pepper noise
    # salt_pepper_ratio = 0.004 
    # num_salt = int(salt_pepper_ratio * image.size * 0.5)
    # num_pepper = int(salt_pepper_ratio * image.size * 0.5)

    # # Add Salt (White Pixels)
    # salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    # warped[salt_coords[0], salt_coords[1]] = [235, 235, 235]  # Softer white noise

    # # Add Pepper (Black Pixels)
    # pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    # warped[pepper_coords[0], pepper_coords[1]] = [25, 25, 25]  # Softer black noise

    # Adjust contrast & brightness moderately
    alpha = 1.08  # Slight contrast boost
    beta = 4  # Subtle brightness shift
    scanned_effect = cv2.convertScaleAbs(warped, alpha=alpha, beta=beta)

    cv2.imwrite(image_path, scanned_effect)
    
    return image_path



def visualize_grounding(image: Image, grounding: list):
    """Visualizes the grounding information on the image."""
    image = np.array(image)
    h, w, _ = image.shape
    
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    
    for item in grounding:
        box = item['box']
        type = item['type']

        color = {"text": "green", "table": "blue", "chart": "red"}
        l, t, r, b = box['l'] * w, box['t'] * h, box['r'] * w, box['b'] * h
        plt.gca().add_patch(plt.Rectangle((l, t), r - l, b - t, 
                                          edgecolor=color[type], linewidth=1, fill=False))
        
        plt.text(l, t - 5, type[:30] + ('...' if len(type) > 30 else ''), 
                 color=color[type], fontsize=8)
    
    plt.axis("off")
    plt.show()


