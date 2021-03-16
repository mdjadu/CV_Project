import numpy as np
import argparse
import cv2

def file_read(file):
    with open(file) as textFile:
        lines = np.array([np.array([int(x) for x in line.split(" ")]) for line in textFile])
    return lines


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("FilePath", type=str ,help="Path to the data file")
	args = parser.parse_args()
	data = file_read(args.FilePath)

	data_3 = np.insert(data,2,0,axis=1)   # 3D co-ordinates with Z=0 for real co-ordinates

	real_coords = [data_3[0:5], data_3[0:5]]
	pixel_coords = [data[0:5], data[6:11]]

	real_coords = np.array(real_coords, dtype=np.float32)
	pixel_coords = np.array(pixel_coords, dtype=np.float32)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(real_coords, pixel_coords, (640, 480), None, None)

	print(mtx)
	print(rvecs)

	tot_error = 0
	n_points = 0
	for i in range(len(real_coords)):
	    prod, _ = cv2.projectPoints(real_coords[i], rvecs[i], tvecs[i], mtx, dist)
	    n_points += len(pixel_coords[i])
	    error = cv2.norm(pixel_coords[i], prod.reshape(pixel_coords.shape[1:]), cv2.NORM_L2)
	    tot_error += error

	avg_err = tot_error / n_points
	print("Reconstruction Error: " + str(avg_err))