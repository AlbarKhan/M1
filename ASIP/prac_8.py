#Practical 8
import cv2
import matplotlib.pyplot as plt
from google.colab import files

def apply_edge_detection(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to load image!")
        return

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x-direction
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y-direction
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Magnitude of gradient

    # Convert Sobel results to 8-bit for visualization
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_combined = cv2.convertScaleAbs(sobel_combined)

    # Apply Canny edge detection
    canny_edges = cv2.Canny(img, threshold1=100, threshold2=200)

    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    
    plt.subplot(2, 2, 2)
    plt.title("Sobel - Gradient X")
    plt.axis('off')
    plt.imshow(sobel_x, cmap='gray')
    
    plt.subplot(2, 2, 3)
    plt.title("Sobel - Gradient Y")
    plt.axis('off')
    plt.imshow(sobel_y, cmap='gray')
    
    plt.subplot(2, 2, 4)
    plt.title("Sobel - Combined Magnitude")
    plt.axis('off')
    plt.imshow(sobel_combined, cmap='gray')
    
    plt.figure()
    plt.title("Canny Edge Detection")
    plt.axis('off')
    plt.imshow(canny_edges, cmap='gray')
    
    plt.tight_layout()
    plt.show()

# File upload
print("Please upload an image file:")
uploaded = files.upload()

# Apply edge detection
if uploaded:
    # Get the filename of the uploaded file
    image_path = next(iter(uploaded))
    apply_edge_detection(image_path)
