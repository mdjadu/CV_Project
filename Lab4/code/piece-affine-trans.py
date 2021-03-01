import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import PiecewiseAffineTransform, warp
from skimage import data
import cv2 as cv


# image = data.lena()
# image = data.astronaut()
image = plt.imread('../data/piece/brick.png')

rows, cols = image.shape[0], image.shape[1]

# Creating grid 
src_cols = np.linspace(0, cols, 20) # Divided the image in 20 parts in X axis (horizontal direction) 
src_rows = np.linspace(0, rows, 20)

# Grid created
src_rows, src_cols = np.meshgrid(src_rows,src_cols)
src = np.dstack([src_cols.flat, src_rows.flat])[0]


# The imp part
# Adding the sin to make the image back to normal
# Amplitude was found to be 50
# Cycle was found to 3 pis (by observation)
dst_rows = src[:, 1] + np.sin(np.linspace(0,3*np.pi, src.shape[0]))*50    
dst_cols = src[:, 0] 

# adding this to remove the black part from the top
dst_rows += 1.5 * 50

dst = np.vstack([dst_cols, dst_rows]).T

# THE TRANSFORMATION
tform = PiecewiseAffineTransform()
tform.estimate(src, dst)

# adding this to remove the black part from the bottom
out_rows = rows - 2.5*50
out_cols = cols 
out = warp(image, tform, output_shape=(out_rows, out_cols))

fig, ax = plt.subplots()
ax.imshow(out)
# This to show the points (grid) 
# Commenting it as the original image doesnt contain such points
# ax.plot(tform.inverse(src)[:, 0], tform.inverse(src)[:, 1], '.b')
ax.axis((0, out_cols, out_rows, 0))
plt.show()
plt.imsave('../results/piece-affine-results/brick.png',out)