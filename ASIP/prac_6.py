import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def apply_noise_smoothing(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Unable to load image!")
        return

    # Add artificial noise to the image
    noisy_img = img + np.random.normal(0, 25, img.shape).astype(np.uint8)

    # Linear Smoothing: Gaussian Blur
    linear_smooth = cv2.GaussianBlur(noisy_img, (5, 5), 1)

    # Nonlinear Smoothing: Median Blur
    nonlinear_smooth = cv2.medianBlur(noisy_img, 5)

    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.axis('off')
    plt.imshow(img, cmap='gray')
    
    plt.subplot(2, 2, 2)
    plt.title("Noisy Image")
    plt.axis('off')
    plt.imshow(noisy_img, cmap='gray')
    
    plt.subplot(2, 2, 3)
    plt.title("Linear Smoothing (Gaussian)")
    plt.axis('off')
    plt.imshow(linear_smooth, cmap='gray')
    
    plt.subplot(2, 2, 4)
    plt.title("Nonlinear Smoothing (Median)")
    plt.axis('off')
    plt.imshow(nonlinear_smooth, cmap='gray')
    
    plt.tight_layout()
    plt.show()

# File upload for Google Colab
print("Please upload an image file.")
uploaded = files.upload()

# Assuming a single file is uploaded, get its filename
if uploaded:
    image_path = list(uploaded.keys())[0]  # Get the uploaded file name
    apply_noise_smoothing(image_path)
else:
    print("No file uploaded.")


# SECOND CODE PRACTICAL SIX

import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

def apply_noise_smoothing_to_signal():
    # Generate a synthetic noisy signal
    fs = 1000  # Sampling frequency
    t = np.linspace(0, 1, fs, endpoint=False)  # Time axis
    clean_signal = np.sin(2 * np.pi * 5 * t)  # Clean sine wave signal
    noise = np.random.normal(0, 0.5, clean_signal.shape)  # Gaussian noise
    noisy_signal = clean_signal + noise

    # Linear Smoothing: Low-pass filter
    b, a = signal.butter(4, 0.1, btype='low')  # 4th order Butterworth low-pass filter
    linear_smooth = signal.filtfilt(b, a, noisy_signal)

    # Nonlinear Smoothing: Median filter
    nonlinear_smooth = signal.medfilt(noisy_signal, kernel_size=5)

    # Plot the results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(4, 1, 1)
    plt.title("Clean Signal")
    plt.plot(t, clean_signal, label="Clean Signal")
    plt.grid()
    plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.title("Noisy Signal")
    plt.plot(t, noisy_signal, label="Noisy Signal", color="orange")
    plt.grid()
    plt.legend()
    
    plt.subplot(4, 1, 3)
    plt.title("Linear Smoothing (Low-pass Filter)")
    plt.plot(t, linear_smooth, label="Linear Smooth", color="green")
    plt.grid()
    plt.legend()
    
    plt.subplot(4, 1, 4)
    plt.title("Nonlinear Smoothing (Median Filter)")
    plt.plot(t, nonlinear_smooth, label="Nonlinear Smooth", color="red")
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# Call the function
apply_noise_smoothing_to_signal()
