import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Function to load an example grayscale image
def load_image():
    print("Upload an image file:")
    uploaded = files.upload()
    image_path = list(uploaded.keys())[0]
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at '{image_path}'. Please provide a valid image.")
    return image

# 1. Log Transformation
def log_transform(image):
    c = 255 / np.log(1 + np.max(image))  # Scale constant
    log_image = c * np.log(1 + image)
    return np.uint8(log_image)

# 2. Power-Law (Gamma) Transformation
def power_law_transform(image, gamma=1.0):
    c = 255 / (np.max(image) ** gamma)
    power_image = c * (image ** gamma)
    return np.uint8(power_image)

# 3. Contrast Adjustment
def contrast_adjustment(image, alpha=1.5, beta=0):
    contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return contrast_image

# 4. Histogram Equalization
def histogram_equalization(image):
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

# 5. Thresholding
def thresholding(image, threshold=128):
    _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

# 6. Halftoning
def halftoning(image):
    halftone_image = np.where(image > 127, 255, 0).astype(np.uint8)
    return halftone_image

# Display images with their titles
def display_images(images, titles):
    plt.figure(figsize=(15, 10))
    for i, (image, title) in enumerate(zip(images, titles)):
        rows, cols = 2, 4  # Adjust grid size to fit all images
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Main Functionality
if __name__ == "__main__":
    # Load the image
    image = load_image()

    # Apply transformations
    log_image = log_transform(image)
    power_image = power_law_transform(image, gamma=0.5)
    contrast_image = contrast_adjustment(image, alpha=2.0, beta=50)
    equalized_image = histogram_equalization(image)
    binary_image = thresholding(image, threshold=128)
    halftone_image = halftoning(image)

    # Display results
    images = [image, log_image, power_image, contrast_image, equalized_image, binary_image, halftone_image]
    titles = [
        "Original", "Log Transform", "Power-law Transform",
        "Contrast Adjustment", "Histogram Equalization",
        "Thresholding", "Halftoning"
    ]
    display_images(images, titles)
