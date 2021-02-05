import cv2
import numpy as np
import argparse


def idx(a):
    return a[0]


def order_points(p):
    p = p.squeeze()
    p = sorted(p, key=idx)
    p_new = np.zeros((4, 2), dtype=np.float32)
    if p[0][1] < p[1][0]:
        p_new[0] = p[0]
        p_new[3] = p[1]
    else:
        p_new[0] = p[1]
        p_new[3] = p[0]

    if p[2][1] < p[3][0]:
        p_new[1] = p[2]
        p_new[2] = p[3]
    else:
        p_new[1] = p[3]
        p_new[2] = p[2]

    return p_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image", type=str)
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    image = cv2.resize(image, (1280, 960))

    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    edged_image = cv2.Canny(blurred_image, 0, 50)

    (contours, _) = cv2.findContours(edged_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for c in contours:
        p = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, p, True)
        if len(approx) == 4:
            target = approx
        break


    approx = order_points(target)
    pts2 = np.float32([[0, 0], [480, 0], [480, 640], [0, 640]])
    M = cv2.getPerspectiveTransform(approx,pts2)
    final_image = cv2.warpPerspective(image, M, (480, 640))
    cv2.imshow("Scanned", final_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()