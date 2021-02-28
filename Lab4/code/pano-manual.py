import cv2
import numpy as np
import argparse
import os


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        params[0].append([x, y])

        # Draw blue point
        cv2.circle(params[2], (x, y), 1, (255, 0, 0), 2)
        cv2.imshow(params[1], params[2])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("path", type=str, help="path to directory containing 2 images")
    args = parser.parse_args()

    # Get images from folder
    images = os.listdir(args.path)
    print(images)

    # Load images
    image1 = cv2.imread(args.path + "/" + images[0])
    image2 = cv2.imread(args.path + "/" + images[1])

    print("Select atleast 4 points in each image in the same order. Then press any key to continue...")

    cv2.imshow("Image1", image1)
    cv2.imshow("Image2", image2)

    cv2.moveWindow("Image1", 100, 100)
    cv2.moveWindow("Image2", 800, 100)

    dst_pts = []
    params1 = [dst_pts, "Image1", image1.copy()]
    cv2.setMouseCallback("Image1", click_event, params1)  # Checking for mouse clicks

    src_pts = []
    params2 = [src_pts, "Image2", image2.copy()]
    cv2.setMouseCallback("Image2", click_event, params2)  # Checking for mouse clicks

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(src_pts) != len(dst_pts):
        print("Error!! Select equal number of points in both images")
        exit(0)

    if len(src_pts) < 4:
        print("Error!! Please select atleast 4 points.")
        exit(0)

    src_pts = np.array(src_pts, dtype=np.float32)
    dst_pts = np.array(dst_pts, dtype=np.float32)
    dst_pts = dst_pts + np.array([image1.shape[1], 0])  # Adding image1.shape[1] to x co-ordinnate to match final output

    # Finding homography matrix using RANSAC method
    matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Doing the transformation
    res = cv2.warpPerspective(image2, matrix, (image1.shape[1]+image2.shape[1], image1.shape[0]))
    res[:image1.shape[0], -image1.shape[1]:] = image1

    y_nz, x_nz, _ = np.nonzero(res)
    res = res[np.min(y_nz):np.max(y_nz), np.min(x_nz):np.max(x_nz)]

    cv2.imshow("Result", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('../results/pano_manual.jpg', res)

