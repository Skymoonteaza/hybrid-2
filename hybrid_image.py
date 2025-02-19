import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

def cross_correlation(img, kernel):
    '''Computes cross-correlation between image and kernel.'''
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape
    pad_h, pad_w = kernel_height // 2, kernel_width // 2

    # Pad the image with zeros
    img_padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')

    # Output image
    output = np.zeros(img.shape)

    # Apply kernel to each pixel
    for i in range(img_height):
        for j in range(img_width):
            region = img_padded[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(region * kernel)

    return output

def convolution(img, kernel):
    '''Flips kernel and performs cross-correlation (which is equivalent to convolution).'''
    flipped_kernel = np.flipud(np.fliplr(kernel))
    return cross_correlation(img, flipped_kernel)

def gaussian_blur(sigma, height, width):
    '''Generates a Gaussian kernel.'''
    center_h, center_w = height // 2, width // 2
    kernel = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            diff = ((i - center_h) ** 2 + (j - center_w) ** 2)
            kernel[i, j] = np.exp(-diff / (2 * sigma ** 2))

    kernel /= np.sum(kernel)  # Normalize
    return kernel

def low_pass(img, sigma, size):
    '''Applies a low-pass filter using a Gaussian kernel.'''
    kernel = gaussian_blur(sigma, size, size)
    return np.stack([convolution(img[:, :, i], kernel) for i in range(3)], axis=-1)

def high_pass(img, sigma, size):
    '''Applies a high-pass filter by subtracting the low-pass image from the original.'''
    return img - low_pass(img, sigma, size)

def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2, high_low2, mixin_ratio):
    '''Creates a hybrid image by blending low-pass and high-pass images.'''
    img1 = img1.astype(np.float32) / 255.0  # Normalize
    img2 = img2.astype(np.float32) / 255.0

    img1 = low_pass(img1, sigma1, size1) if high_low1 == 'low' else high_pass(img1, sigma1, size1)
    img2 = low_pass(img2, sigma2, size2) if high_low2 == 'low' else high_pass(img2, sigma2, size2)

    # Merge images
    hybrid_img = (img1 * (1 - mixin_ratio)) + (img2 * mixin_ratio)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

# Load images with swapped roles
img1 = cv2.imread('image1_LowFreq.png')  # Now Low-pass
img2 = cv2.imread('image2_HighFreq.png')  # Now High-pass

# Check if images are loaded
if img1 is None:
    print("Error: 'image1_LowFreq.png' not found.")
    sys.exit()
if img2 is None:
    print("Error: 'image2_HighFreq.png' not found.")
    sys.exit()

# Convert from BGR to RGB for correct display
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# Resize images
img1 = cv2.resize(img1, (500, 500))
img2 = cv2.resize(img2, (500, 500))

# Define parameters
sigma1, size1 = 5, 15  # Low-pass for img1
sigma2, size2 = 5, 15  # High-pass for img2
mixin_ratio = 0.5  # Balance blending

# Generate hybrid image
hybrid_img = create_hybrid_image(img1, img2, sigma1, size1, 'low', sigma2, size2, 'high', mixin_ratio)

# Display images
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Low-Pass Image")
plt.imshow(low_pass(img1, sigma1, size1).astype(np.uint8))

plt.subplot(1, 3, 2)
plt.title("High-Pass Image")
plt.imshow(high_pass(img2, sigma2, size2).astype(np.uint8))

plt.subplot(1, 3, 3)
plt.title("Hybrid Image")
plt.imshow(hybrid_img)

plt.show()

# Save the result
cv2.imwrite("hybrid_image.png", cv2.cvtColor(hybrid_img, cv2.COLOR_RGB2BGR))
print("Hybrid image saved as 'hybrid_image.png'")
