import numpy as np
import argparse
import cv2

def file_read(file):
    with open(file) as textFile:
    	lines = textFile.readlines()
    return lines

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("FilePath", type=str ,help="Path to the data file")
	args = parser.parse_args()
	data = file_read(args.FilePath)

	data_x = []
	data_y = []

	for d in data:
	    x = int(d.split(" ")[0])
	    y = int(d.split(" ")[1])
	    data_x.append(x)
	    data_y.append(y)

	sort1_idx = np.lexsort((data_y[0:6], data_x[0:6]))
	sort2_idx = np.lexsort((data_y[6:12], data_x[6:12]))

	for i in range(0,len(sort1_idx)-1,2):
		j = sort1_idx[i]
		k = sort1_idx[i+1]
		if data_y[k] < data_y[j]:
			temp = sort1_idx[i]
			sort1_idx[i] = sort1_idx[i+1] 
			sort1_idx[i+1] = temp

	for i in range(0,len(sort2_idx)-1,2):
		j = sort2_idx[i]
		k = sort2_idx[i+1]
		if data_y[k+6] < data_y[j+6]:
			temp = sort2_idx[i]
			sort2_idx[i] = sort2_idx[i+1] 
			sort2_idx[i+1] = temp

	p1 = []
	for i in sort1_idx:
		p1.append([data_x[i],data_y[i]]) 

	p2 = []
	for i in sort2_idx:
		p2.append([data_x[i+6],data_y[i+6]]) 


	p1_3 = np.insert(p1,2,0,axis=1)     # 3D co-ordinates with Z=0 for real co-ordinates
	p2_3 = np.insert(p2,2,0,axis=1)

	world_coords = [p2_3, p2_3]         # Method2 (refer Reflection Essay). For image 1 put p1_3 instead of p2_3, & vice-versa.
	pixel_coords = [p1, p2]

	# world_coords = [[[0, 0, 0], [0, 3, 0], [3, 0, 0], [3, 3, 0], [6, 0, 0], [6, 3, 0]],          # Method1 (refer Reflection Essay)
 #                	[[0, 0, 0], [0, 3, 0], [3, 0, 0], [3, 3, 0], [6, 0, 0], [6, 3, 0]]]

	world_coords = np.array(world_coords, dtype=np.float32)
	pixel_coords = np.array(pixel_coords, dtype=np.float32)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_coords, pixel_coords, (1000,1000), None, None)

	print(mtx)
	# print(rvecs)

	tot_error = 0
	n_points = 0
	for i in range(len(world_coords)):
	    prod, _ = cv2.projectPoints(world_coords[i], rvecs[i], tvecs[i], mtx, dist)
	    n_points += len(pixel_coords[i])
	    error = cv2.norm(pixel_coords[i], prod.reshape(pixel_coords.shape[1:]), cv2.NORM_L2)
	    tot_error += error

	avg_err = tot_error / n_points
	print("Reconstruction Error: " + str(avg_err))