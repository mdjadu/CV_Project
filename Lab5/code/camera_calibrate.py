import cv2
import matplotlib.pyplot as plt
import numpy as np


img1 = cv2.imread("../data/calib_images/img1.jpg")

img1 = cv2.resize(img1, (640, 480))

real_coords = [[[0, 2, 0], [1, 2, 0], [2, 2, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]],
               [[0, 2, 0], [1, 2, 0], [2, 2, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]],
               [[0, 2, 0], [1, 2, 0], [2, 2, 0], [0, 1, 0], [1, 1, 0], [2, 1, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]]]

pixel_coords = [[[244, 177], [315, 177], [387, 178], [244, 218], [315, 218], [387, 219], [245, 255], [316, 257], [386, 256]],
                [[267, 209], [267, 166], [267, 128], [341, 208], [341, 167], [340, 127], [414, 208], [413, 166], [413, 127]],
                [[308, 187], [327, 150], [345, 113], [376, 198], [395, 160], [414, 123], [444, 208], [462, 171], [480, 132]]]

real_coords = np.array(real_coords, dtype=np.float32)
pixel_coords = np.array(pixel_coords, dtype=np.float32)

for point in pixel_coords[0]:
    x = point[0]
    y = point[1]
    cv2.circle(img1, (x, y), 1, (255, 0, 0), 2)

cv2.imwrite("../data/real.jpg", img1)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(real_coords, pixel_coords, img1.shape[:2][::-1], None, None)

tot_error = 0
n_points = 0
for i in range(len(real_coords)):
    prod, _ = cv2.projectPoints(real_coords[i], rvecs[i], tvecs[i], mtx, dist)
    n_points += len(pixel_coords[i])
    error = cv2.norm(pixel_coords[i], prod.reshape(9, 2), cv2.NORM_L2)
    tot_error += error

avg_err = tot_error / n_points
print("Reconstruction Error: " + str(avg_err))

projected_coords, _ = cv2.projectPoints(real_coords[0], rvecs[0], tvecs[0], mtx, dist)
font = cv2.FONT_HERSHEY_SIMPLEX
dx = np.array([-4, 4], dtype=np.float32)
for point in projected_coords:
    point[0] += dx
    x = point[0][0]
    y = point[0][1]
    cv2.putText(img1, 'x', (x, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imshow("projected", img1)
cv2.imwrite("../data/projected.jpg", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()