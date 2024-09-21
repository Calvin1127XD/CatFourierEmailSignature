import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image (your sticker with background removed)
image = cv2.imread('cat_iOS.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to separate the cat from the background
_, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (which should be the cat)
largest_contour = max(contours, key=cv2.contourArea)

# Draw the contour on the original image
contour_img = image.copy()
cv2.drawContours(contour_img, [largest_contour], -1, (255, 0, 0), 2)  # Blue contour

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Sticker')
ax1.axis('off')

ax2.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
ax2.set_title('Cat Boundary')
ax2.axis('off')

plt.tight_layout()
plt.show()

# Save the contour points for further use
np.save('cat_iOS_contour.npy', largest_contour)
print(f"Contour saved with {len(largest_contour)} points.")
