import cv2
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("img_path", help="path to image",type=str)
args = parser.parse_args()

# some needed values
image = cv2.imread(args.img_path)
save_image = image
w = h = 1080    # original width and height 
epoch = 150      # number of intermediate steps
sec = 5         # how long the video to be in sec

# points are the coordinates of map in the given image
points = np.array([[480,190],[990,830],[700,1025],[220,240]],dtype = 'float32')
save_points = points
# expected image 
final_image = np.array([[0,0],[512,0],[512,385],[0,385]],dtype = 'float32')

# arr is array which contains by what value we should add in points to get the intermediate step
arr = np.array([[-(480/epoch),-(190/epoch)],[-((990-512)/epoch),-(830/epoch)],[-((700-512)/epoch),-((1025-385)/epoch)],[-(220/epoch),-(240-385)/epoch]],dtype = 'float32')

# creating a mask
# art = np.uint8(0 * np.ones((w, h,3)))
# this is one way to create a mask
# cv2.polylines(art,[np.array(points,dtype = 'int32')],True,(255,255,255))
# cv2.fillPoly(art,[np.array(points,dtype = 'int32')],(255,255,255))
# and(operator) is use to mask
# warped = cv2.bitwise_and(art,image)

i = 0

while i < epoch:
    print("Press Space to stop the animation",end="\r")
    # adding the arr to get the intermediate steps
    newpoints = points + (arr)*i
    #  getting the perspective image for the intermediate step
    M = cv2.getPerspectiveTransform(points, newpoints)
    warped = cv2.warpPerspective(image, M, (w,h))

    art = np.uint8(np.zeros((w, h,3)))
    # this is second way to create a mask
    cv2.drawContours(art,[np.array(newpoints,dtype = 'int32')],-1,(255,255,255),-1)
    # mask
    warped = cv2.bitwise_and(warped,art)
    # adding images to animation
    cv2.imshow('Animation (Press "space" to stop)',warped)
    i = i+1
    if i>epoch-1:
        i = 0
        points = save_points
        image = save_image
    k = cv2.waitKey(10)
    if k==32:
        break

M = cv2.getPerspectiveTransform(save_points, final_image)
warped = cv2.warpPerspective(save_image, M, (512,385))

cv2.imshow('final',warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save the output
file_path = "../results/map.jpg"
cv2.imwrite(file_path, warped)