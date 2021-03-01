import sys
import os
import cv2 as cv
import numpy as np

"""
This function stitches the warImg and refImg with refImg being intact.
"""
def stitch(warpImg, refImg):
	# to decrease the number of rotations
	wy_nz, wx_nz, _ = np.nonzero(warpImg)
	wymin, wymax, wxmin, wxmax = np.min(wy_nz),np.max(wy_nz), np.min(wx_nz),np.max(wx_nz)
	ry_nz, rx_nz, _ = np.nonzero(refImg)
	rymin, rymax, rxmin, rxmax = np.min(ry_nz),np.max(ry_nz), np.min(rx_nz),np.max(rx_nz)

	ymin, ymax, xmin, xmax = min(wymin,rymin),max(wymax,rymax), min(wxmin,rxmin),max(wxmax,rxmax)
	
	for i in range(xmin,xmax,1):
		for j in range(ymin,ymax,1):
	# for i in range(refImg.shape[1]):
	# 	for j in range(refImg.shape[0]):
			if(np.array_equal(refImg[j,i],np.array([0,0,0])) and  np.array_equal(warpImg[j,i],np.array([0,0,0]))):
				warpImg[j,i] = [0, 0, 0]
			else:
				if (np.array_equal(refImg[j,i],[0,0,0])):
					refImg[j,i] = warpImg[j,i]
				else:
					if not np.array_equal(refImg[j,i], [0,0,0]):
						bw, gw, rw = warpImg[j,i]
						br,gr,rr = refImg[j,i]
						warpImg[j, i] = [br,gr,rr]
	return refImg


"""
This function gives the homography of stitchImg wrt to refImg.
"""
def homography(refImg, stitchImg): 
	refImg_bw = cv.cvtColor(refImg,cv.COLOR_BGR2GRAY) 
	stitchImg_bw = cv.cvtColor(stitchImg, cv.COLOR_BGR2GRAY) 

	n = 500

	orb = cv.ORB_create(n) 

	kp1, des1 = orb.detectAndCompute(refImg_bw,None) 
	kp2, des2 = orb.detectAndCompute(stitchImg_bw,None) 

	matcher = cv.BFMatcher(cv.NORM_HAMMING,crossCheck=True) 
	matches = matcher.match(des1,des2)
	matches = sorted(matches, key = lambda x:x.distance)
	matchesFilt = matches[:int(len(matches)/5)]

	matches_img = cv.drawMatches(refImg,kp1,stitchImg,kp2,matchesFilt,None,matchColor = (0,255,0))

	if len(matchesFilt) > 10:
		dst_pts = np.float32([kp1[m.queryIdx].pt for m in matchesFilt]).reshape(-1,1,2)
		src_pts = np.float32([kp2[m.trainIdx].pt for m in matchesFilt]).reshape(-1,1,2)    # points from the image that needs to be transformed

		M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC)  
		return M
	else:
		print ("Not enough matches are found")
		return np.zeros((3,3))


def mosaic(imagesPad, refImage):
	for i, j in enumerate(imagesPad):
		difference = cv.subtract(refImage, j)
		b, g, r = cv.split(difference)
		if cv.countNonZero(b) == 0 and cv.countNonZero(g) == 0 and cv.countNonZero(r) == 0:
			refIndex = i

	print(refIndex)

	H = []
	for i in imagesPad:
		H.append(np.eye(3))

	H[refIndex] = np.eye(3)
	i = refIndex-1
	while i>=0:
		H_temp = homography(imagesPad[i+1],imagesPad[i])
		H[i] = np.dot(H_temp,H[i+1])
		i -= 1

	j = refIndex+1
	while j<len(imagesPad):
		H_temp = homography(imagesPad[j-1],imagesPad[j])
		H[j] = np.dot(H_temp,H[j-1])
		j += 1

	print(H)

	imagesWarp = []
	for i,j in enumerate(imagesPad):
		warp_temp = cv.warpPerspective(j,H[i],(j.shape[1], j.shape[0]))
		imagesWarp.append(warp_temp)

	imgStitched = imagesWarp[refIndex]
	i = refIndex-1
	while i>=0:
		imgStitched = stitch(imagesWarp[i],imgStitched)
		i -= 1

	j = refIndex+1
	while j<len(imagesPad):
		imgStitched = stitch(imagesWarp[j],imgStitched)
		j += 1

	return imgStitched


if __name__ == '__main__':
	files = os.listdir(sys.argv[1])
	imgNames = []
	for f in files:
		if f[-4:] == ".png" or f[-4:] == ".jpg" or f[-5:] == ".jpeg":  
			imgNames.append(f)
	imgNames.sort()
	print(imgNames)

	images = []
	for i in imgNames:
		img = cv.imread(f'{sys.argv[1]}/{i}')
		img = cv.resize(img,(0,0),fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
		images.append(img)

	print(images[0].dtype)
	
	refIndex = int(sys.argv[2])-1
	ref_y = images[refIndex].shape[0]
	ref_x = images[refIndex].shape[1]
	res_shape = (len(images)*ref_y, len(images)*ref_x, 3)   

	# res_shape = (2*len(images)*ref_y, 2*len(images)*ref_x, 3)  

	imagesPad = []
	for i in images:
		i = cv.resize(i, (ref_x,ref_y))
		res = np.zeros(res_shape, dtype=np.uint8)
		res[ref_y*int(len(images)/2):ref_y*int(len(images)/2)+ref_y, refIndex*ref_x:ref_x*(refIndex+1)] = i

		# res[ref_y*len(images):ref_y*len(images)+ref_y, ref_x*len(images):ref_x*len(images)+ref_x] = i
		imagesPad.append(res)

	refImage = imagesPad[refIndex]

	pano = mosaic(imagesPad,refImage)	
	y_nz, x_nz, _ = np.nonzero(pano)
	pano = pano[np.min(y_nz):np.max(y_nz), np.min(x_nz):np.max(x_nz)]

	cv.imwrite("pano.jpg",pano)

	cv.imshow('panorama',pano)
	cv.waitKey(0)
	cv.destroyAllWindows()