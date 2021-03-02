import cv2
import numpy as np
import argparse
import os


def add_images(image1, image2):
    images_res = np.zeros_like(image1)
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if (image1[i, j] != 0).any():
                images_res[i, j] = image1[i, j]
            else:
                images_res[i, j] = image2[i, j]

    return images_res


def stitch(images, res, n):
    src = images[n-1]
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    dst_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    n_features = 2000

    orb = cv2.ORB_create(n_features)
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(src_gray, None)
    kp2, des2 = orb.detectAndCompute(dst_gray, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    # Match descriptors.
    matches = bf.match(des1, des2)

    matches.sort(key=lambda x: x.distance)

    n_points = int(len(matches) * .05)

    src_pts = []
    dst_pts = []
    for i in range(n_points):
        dst_pts.append(kp2[matches[i].trainIdx].pt)
        src_pts.append(kp1[matches[i].queryIdx].pt)

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)

    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    res1 = cv2.warpPerspective(src, matrix, (res.shape[1], res.shape[0]))

    return add_images(res, res1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str, help="path to directory containing 2 images")
    parser.add_argument("ref", type=int, help="reference image no")
    args = parser.parse_args()

    # Get images from folder
    images_list = os.listdir(args.path)

    # Load images
    images = []
    n_images = len(images_list)
    for i in range(n_images):
        image = cv2.imread(args.path + "/" + images_list[i])
        image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

        images.append(image)

    ref = args.ref
    ref_y = images[ref-1].shape[0]
    ref_x = images[ref-1].shape[1]
    res_y = 0
    res_x = 0
    for image in images:
        res_y += 2*image.shape[0]
        res_x += 2*image.shape[1]
    res_y -= ref_y
    res_x -= ref_x

    res_shape = (res_y, res_x, 3)
    res = np.zeros(res_shape, dtype=np.uint8)
    res_ref_y = (res_y - ref_y) // 2
    res_ref_x = (res_x - ref_x) // 2
    res[res_ref_y:res_ref_y+ref_y, res_ref_x:res_ref_x+ref_x] = images[ref-1]

    transformation = []
    for i in range(ref-1, 0, -1):
        res = stitch(images, res, i)
        # pass
    for i in range(ref+1, n_images+1, 1):
        res = stitch(images, res, i)

    y_nz, x_nz, _ = np.nonzero(res)
    res = res[np.min(y_nz):np.max(y_nz), np.min(x_nz):np.max(x_nz)]

    cv2.imshow("Result", res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('../results/pano_general.jpg', res)
