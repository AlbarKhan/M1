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

    # Compute the Gradient using Sobel operators
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)  # Magnitude of the gradient

    # Compute the Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=3)

    # Normalize the results for display
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    gradient_magnitude = cv2.convertScaleAbs(gradient_magnitude)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Plot the original and enhanced images
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.title("Original Image")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    
    plt.subplot(2, 3, 2)
    plt.title("Gradient (Sobel X)")
    plt.axis('off')
    plt.imshow(sobel_x, cmap='gray')
    
    plt.subplot(2, 3, 3)
    plt.title("Gradient (Sobel Y)")
    plt.axis('off')
    plt.imshow(sobel_y, cmap='gray')
    
    plt.subplot(2, 3, 4)
    plt.title("Gradient Magnitude")
    plt.axis('off')
    plt.imshow(gradient_magnitude, cmap='gray')
    
    plt.subplot(2, 3, 5)
    plt.title("Laplacian")
    plt.axis('off')
    plt.imshow(laplacian, cmap='gray')
    
    plt.tight_layout()
    plt.show()

# File upload for Google Colab
print("Please upload an image file.")
uploaded = files.upload()

# Assuming a single file is uploaded, get its filename
if uploaded:
    image_path = list(uploaded.keys())[0]  # Get the uploaded file name
    apply_image_enhancements(image_path)
else:
    print("No file uploaded.")
