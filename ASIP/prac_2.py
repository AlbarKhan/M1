import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import convolve
from google.colab import files

# Function to perform template matching on image data
def template_matching():
    try:
        print("Upload the main image:")
        uploaded_image = files.upload()
        image_path = list(uploaded_image.keys())[0]

        print("Upload the template image:")
        uploaded_template = files.upload()
        template_path = list(uploaded_template.keys())[0]

        # Load the image and the template
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        # Check if the files were loaded correctly
        if image is None or template is None:
            raise FileNotFoundError("Image or template file not found.")

        # Perform template matching
        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Extract the top-left corner of the matching area
        top_left = max_loc
        h, w = template.shape

        # Draw a rectangle around the matched region
        matched_image = image.copy()
        cv2.rectangle(matched_image, top_left, (top_left[0] + w, top_left[1] + h), 255, 2)

        # Plot the results
        plt.figure(figsize=(12, 6))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Image")
        plt.axis('off')

        # Template image
        plt.subplot(1, 3, 2)
        plt.imshow(template, cmap='gray')
        plt.title("Template Image")
        plt.axis('off')

        # Matched image with rectangle
        plt.subplot(1, 3, 3)
        plt.imshow(matched_image, cmap='gray')
        plt.title("Matched Image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        # Return results for further processing if needed
        return result, top_left, (w, h)

    except Exception as e:
        print(f"An error occurred during template matching: {e}")
        return None


# Function to perform convolution on sound data
def sound_convolution():
    try:
        print("Upload the WAV file:")
        uploaded_wav = files.upload()
        input_wav = list(uploaded_wav.keys())[0]

        # Read the sound file
        sample_rate, sound_data = wavfile.read(input_wav)

        # Handle multi-channel audio
        if sound_data.ndim > 1:  # Stereo or multi-channel
            sound_data = sound_data[:, 0]  # Use the first channel

        # Check for file truncation
        if len(sound_data) == 0:
            raise ValueError("Sound file is empty or corrupted.")

        # Normalize sound data
        sound_data = sound_data / np.max(np.abs(sound_data))

        # Define a simple kernel for smoothing
        kernel = np.ones(5) / 5

        # Perform convolution
        convolved_data = convolve(sound_data, kernel, mode='same')

        # Plot original and convolved sound data
        plt.figure(figsize=(12, 6))
        
        # Plot original sound
        plt.subplot(2, 1, 1)
        plt.plot(sound_data, label="Original Sound")
        plt.title("Original Sound Signal")
        plt.grid()
        plt.legend()

        # Plot convolved sound
        plt.subplot(2, 1, 2)
        plt.plot(convolved_data, label="Convolved Sound", color='orange')
        plt.title("Convolved Sound Signal")
        plt.grid()
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Return the convolved data
        return convolved_data

    except Exception as e:
        print(f"An error occurred during sound convolution: {e}")
        return None


# Example Usage
# 1. Template Matching
print("Performing Template Matching...")
template_matching()

# 2. Sound Convolution
print("\nPerforming Sound Convolution...")
sound_convolution()
