import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="path to image",
                    type=str)
args = parser.parse_args()

# Importing image in BGR format
img = cv2.imread(args.image_path, cv2.IMREAD_COLOR)

# Normalize to values in [0, 1]
norm_img = img / 255

# Converting from BGR to RBG
img_rbg = img.copy()
img_rbg[:, :, 0] = img[:, :, 2]
img_rbg[:, :, 2] = img[:, :, 0]
norm_img_rbg = img_rbg / 255

# Plotting original image using matplotlib
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax.set_title("Original")
plt.imshow(img_rbg)

# Plotting normalized image using matplotlib
ax = fig.add_subplot(1, 2, 2)
ax.set_title("Normalized")
plt.imshow(norm_img_rbg)
plt.show()

# Plotting original and normalized image using opencv
cv2.imshow('Original', img)
cv2.imshow('Normalized', norm_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

