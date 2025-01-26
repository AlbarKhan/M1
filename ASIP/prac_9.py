import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def morphological_processing(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to load image!")
        return

    # Define a kernel
    kernel = np.ones((5, 5), np.uint8)

    # Morphological operations
    erosion = cv2.erode(img, kernel, iterations=1)
    dilation = cv2.dilate(img, kernel, iterations=1)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

    # Plot the results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 3, 1)
    plt.title("Original Image")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    
    plt.subplot(3, 3, 2)
    plt.title("Erosion")
    plt.axis('off')
    plt.imshow(erosion, cmap='gray')
    
    plt.subplot(3, 3, 3)
    plt.title("Dilation")
    plt.axis('off')
    plt.imshow(dilation, cmap='gray')
    
    plt.subplot(3, 3, 4)
    plt.title("Opening")
    plt.axis('off')
    plt.imshow(opening, cmap='gray')
    
    plt.subplot(3, 3, 5)
    plt.title("Closing")
    plt.axis('off')
    plt.imshow(closing, cmap='gray')
    
    plt.subplot(3, 3, 6)
    plt.title("Gradient")
    plt.axis('off')
    plt.imshow(gradient, cmap='gray')
    
    plt.subplot(3, 3, 7)
    plt.title("Top Hat")
    plt.axis('off')
    plt.imshow(tophat, cmap='gray')
    
    plt.subplot(3, 3, 8)
    plt.title("Black Hat")
    plt.axis('off')
    plt.imshow(blackhat, cmap='gray')
    
    plt.tight_layout()
    plt.show()

# File upload
print("Please upload an image file:")
uploaded = files.upload()

# Apply morphological processing
if uploaded:
    # Get the filename of the uploaded file
    image_path = next(iter(uploaded))
    morphological_processing(image_path)
