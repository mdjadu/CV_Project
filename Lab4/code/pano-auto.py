import cv2
import numpy as np
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str, help="path to directory containing 2 images")
    args = parser.parse_args()

    # Get images from folder
    images = os.listdir(args.path)
    print(images)

    # Load images
    image1 = cv2.imread(args.path + "/" + images[0]) # Reference image
    image2 = cv2.imread(args.path + "/" + images[1])

    # Convert to grayscale for feature detection
    im1Gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    im2Gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    n_features = 500

    orb = cv2.ORB_create(n_features)
    # Get keypoints
    kp1, des1 = orb.detectAndCompute(im1Gray, None)
    kp2, des2 = orb.detectAndCompute(im2Gray, None)

    # Find correspondences between the images
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)

    # Sort the matches based on distance
    matches.sort(key=lambda x: x.distance)

    # Take top 10% of the matches
    n_points = int(len(matches) * .1)

    src_pts = []
    dst_pts = []
    for i in range(n_points):
        src_pts.append(kp2[matches[i].trainIdx].pt)
        dst_pts.append(kp1[matches[i].queryIdx].pt)

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    dst_pts = dst_pts + np.array([image1.shape[1], 0])  # Adding image1.shape[1] to x co-ordinnate to match final output

    # Find homography matrix
    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Do perspective transformation
    res = cv2.warpPerspective(image2, matrix, (image1.shape[1]+image2.shape[1], image1.shape[0]))
    res[:image1.shape[0], -image1.shape[1]:] = image1

    # Remove the extra back edges
    y_nz, x_nz, _ = np.nonzero(res)
    res = res[np.min(y_nz):np.max(y_nz), np.min(x_nz):np.max(x_nz)]

    cv2.imshow("Result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('../results/pano_auto.jpg', res)

