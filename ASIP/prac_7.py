
#PRACTICAL 7
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def apply_image_enhancements(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to load image!")
        return

    # 1. Smoothing using Gaussian Blur
    smoothed = cv2.GaussianBlur(img, (5, 5), 1)

    # 2. Sharpening using Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    sharpened = cv2.convertScaleAbs(img - laplacian)

    # 3. Unsharp Masking
    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 1)
    unsharp_mask = cv2.addWeighted(img, 1.5, gaussian_blur, -0.5, 0)

    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    
    plt.subplot(2, 2, 2)
    plt.title("Smoothing (Gaussian Blur)")
    plt.axis('off')
    plt.imshow(smoothed, cmap='gray')
    
    plt.subplot(2, 2, 3)
    plt.title("Sharpening (Laplacian)")
    plt.axis('off')
    plt.imshow(sharpened, cmap='gray')
    
    plt.subplot(2, 2, 4)
    plt.title("Unsharp Masking")
    plt.axis('off')
    plt.imshow(unsharp_mask, cmap='gray')
    
    plt.tight_layout()
    plt.show()

# File upload
print("Please upload an image file:")
uploaded = files.upload()

# Apply image enhancements
if uploaded:
    # Get the filename of the uploaded file
    image_path = next(iter(uploaded))
    apply_image_enhancements(image_path)
