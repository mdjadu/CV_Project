import cv2
import matplotlib.pyplot as plt
import numpy as np


img1 = cv2.imread("../data/calib_images/img1.jpg")

img1 = cv2.resize(img1, (640, 480))

pixel_coords = [[[264, 127], [338, 126], [412, 126], [265, 167], [338, 167], [414, 166], [268, 209], [341, 209], [416, 207]],
                [[387, 178], [388, 219], [384, 256], [316, 179], [318, 218], [315, 258], [246, 178], [244, 217], [244, 257]]]
real_coords = [[[0, 2, 0], [1, 2, 0], [2, 2, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]],
               [[0, 2, 0], [1, 2, 0], [2, 2, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]]]


real_coords = np.array(real_coords, dtype=np.float32)
pixel_coords = np.array(pixel_coords, dtype=np.float32)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(real_coords, pixel_coords, img1.shape[:2][::-1], None, None)
focal_length = mtx[0][0]*3.92/img1.shape[1]
print("focal_length: " + str(focal_length))