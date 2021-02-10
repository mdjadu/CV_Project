# Coordinates of chessboard corners in the distorted image in 
# clockwise direction starting from top left corner as origin are [(0,0), (600,-60),(660,-660),(60,-600)].

# Coordinates of chessboard corners in the original image in 
# clockwise direction starting from top left corner as origin are [(0,0), (600,0),(600,-600),(0,-600)].

# [x_org;y_org] = H * [x_dis;y_dis;1], H is an affine transformation.


import numpy as np
import argparse
import cv2

def manual(src_img, src_coord, des_coord):
	H = [[],[]]
	homo = np.array([1, 1, 1])
	src_coord_homo = np.concatenate((src_coord, homo[:,None]), axis=1)

	H[0] = np.linalg.solve(src_coord_homo, des_coord[:,0])
	H[1] = np.linalg.solve(src_coord_homo, des_coord[:,1])
	H = np.array(H)
	org_img = cv2.warpAffine(src_img, H, (600,600), flags=cv2.WARP_INVERSE_MAP)
	return org_img

def api(src_img, src_coord, des_coord):
	H = cv2.getAffineTransform(src_coord, des_coord)
	org_img = cv2.warpAffine(src_img, H, (600,600), flags=cv2.WARP_INVERSE_MAP)
	return org_img


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-mat", default='api', help="Computation method")
	args = parser.parse_args()

	dis_img = cv2.imread("../data/distorted.jpg")  # mention the image path if it is in a different directory
	C_org = np.float32(np.array([[600,0],[600,-600],[0,-600]]))
	C_dis = np.float32(np.array([[600,-60],[660,-660],[60,-600]]))

	if args.mat == 'manual':
		org_img = manual(dis_img, C_dis, C_org)
	
	if args.mat == 'api':
		org_img = api(dis_img, C_dis, C_org)

	cv2.imshow("Original Image", org_img)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
