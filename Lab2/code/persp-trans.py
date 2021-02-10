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
epoch = 20      # number of intermediate steps
sec = 5         # how long the video to be in sec

# points are the coordinates of map in the given image
points = np.array([[480,190],[990,830],[700,1025],[220,240]],dtype = 'float32')
save_points = points
# expected image 
final_image = np.array([[0,0],[512,0],[512,385],[0,385]],dtype = 'float32')

# arr is array which contains by what value we should add in points to get the intermediate step
arr = np.array([[-(480/epoch),-(190/epoch)],[-((990-512)/epoch),-(830/epoch)],[-((700-512)/epoch),-((1025-385)/epoch)],[-(220/epoch),-(240-385)/epoch]],dtype = 'float32')

# video initalization
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# epoch/5 this means fps will be epoch/5 5 because we want to create 5 sec image
video=cv2.VideoWriter('video.mp4', fourcc, int(epoch/sec), (w,h))

# creating a mask
art = np.uint8(0 * np.ones((w, h,3)))
# this is one way to create a mask
cv2.polylines(art,[np.array(points,dtype = 'int32')],True,(255,255,255))
cv2.fillPoly(art,[np.array(points,dtype = 'int32')],(255,255,255))
# and(operator) is use to mask
warped = cv2.bitwise_and(art,image)

# Just for betterment contrast of the images is been increased
buf = warped
contrast = 20       # can be increased if want more contrast image
f = 131*(contrast + 127)/(127*(131-contrast))
alpha_c = f
gamma_c = 127*(1-f)
buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
image = buf

# the 1st image(masked) is added in the video frame
video.write(image)
i = 0

while i < epoch:
    print("Press Space to stop the animation",end="\r")
    # adding the arr to get the intermediate steps
    newpoints = points + (arr)
    #  getting the perspective image for the intermediate step
    M = cv2.getPerspectiveTransform(points, newpoints)
    warped = cv2.warpPerspective(image, M, (w,h))

    art = np.uint8(np.zeros((w, h,3)))
    # this is second way to create a mask
    cv2.drawContours(art,[np.array(newpoints,dtype = 'int32')],-1,(255,255,255),-1)
    # mask
    warped = cv2.bitwise_and(warped,art)
    # adding images to video
    warped = cv2.detailEnhance(warped, sigma_s=5/epoch, sigma_r=0.15/epoch)
    cv2.imshow('Animation (Press "space" to stop)',warped)
    # video.write(warped)
    i = i+1
    points = newpoints
    image = warped
    if i>epoch-1:
        i = 0
        points = save_points
        image = save_image
    k = cv2.waitKey(10)
    if k==32:
        break

# M = cv2.getPerspectiveTransform(points, final_image)
# warped = cv2.warpPerspective(image, M, (w,h))
# art = np.uint8(np.zeros((w, h,3)))
# cv2.drawContours(art,[np.array(newpoints,dtype = 'int32')],-1,(255,255,255),-1)
# warped = cv2.bitwise_and(warped,art)
# warped = cv2.detailEnhance(warped, sigma_s=5/epoch, sigma_r=0.15/epoch)
# video.write(warped)

# creating video and store in ./video.mp4
# video.release()

M = cv2.getPerspectiveTransform(save_points, final_image)
warped = cv2.warpPerspective(save_image, M, (512,385))

buf = warped
contrast = 30
f = 131*(contrast + 127)/(127*(131-contrast))
alpha_c = f
gamma_c = 127*(1-f)

buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

cv2.imshow('final',buf)
cv2.waitKey(0)
cv2.destroyAllWindows()

# save the output
file_path = "../results/map.jpg"
cv2.imwrite(file_path, buf)