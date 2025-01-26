import numpy as np
import cv2 
import matplotlib.pyplot as plt 

image = cv2.imread('11.jpg', cv2.IMREAD_GRAYSCALE) 
 
downsampled_image = cv2.resize(image, None, fx=0.5, fy=0.5, 
interpolation=cv2.INTER_AREA) 
 
upsampled_image = cv2.resize(downsampled_image, (image.shape[1], 
image.shape[0]), interpolation=cv2.INTER_LINEAR) 
 
plt.figure(figsize=(12, 4)) 
 
plt.subplot(1, 3, 1) 
plt.title("Original Image") 
plt.imshow(image, cmap='gray') 
plt.axis('off') 
 
plt.subplot(1, 3, 2) 
plt.title("Downsampled Image") 
plt.imshow(downsampled_image, cmap='gray') 
plt.axis('off') 
 
plt.subplot(1, 3, 3) 
plt.title("Upsampled Image") 
plt.imshow(upsampled_image, cmap='gray') 
plt.axis('off') 
 
plt.tight_layout() 
plt.show() 


ifft_image_shifted = np.fft.ifftshift(fft_image)
reconstructed_image = np.fft.ifft2(ifft_image_shifted)
reconstructed_image = np.abs(reconstructed_image)

plt.figure()
plt.title("Reconstructed Image from FFT")
plt.imshow(reconstructed_image, cmap='gray')
plt.axis('off')
plt.show()