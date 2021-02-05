import cv2
import numpy as np
import argparse


# Index function needed for sorting
def idx(a):
    return a[0]


# Orders the corner points
def order_points(p):
    p = p.squeeze()
    p = sorted(p, key=idx)
    p_new = np.zeros((4, 2), dtype=np.float32)
    if p[0][1] < p[1][1]:
        p_new[0] = p[0]
        p_new[3] = p[1]
    else:
        p_new[0] = p[1]
        p_new[3] = p[0]

    if p[2][1] < p[3][1]:
        p_new[1] = p[2]
        p_new[2] = p[3]
    else:
        p_new[1] = p[3]
        p_new[2] = p[2]
    return p_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to image", type=str)
    parser.add_argument("output_image_path", help="Path to output image", nargs='?', type=str,
                        default="results/scanned_output.jpg")
    args = parser.parse_args()

    tgt_width = 480
    tgt_height = 640

    src_width = 480*2
    src_height = 640*2

    # Read image
    image = cv2.imread(args.image_path)

    # Resize image
    image = cv2.resize(image, (src_width, src_height))

    # Convert to gray scale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur image
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Canny edge detection
    edged_image = cv2.Canny(blurred_image, 0, 50)

    # Get contours
    (contours, _) = cv2.findContours(edged_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Get biggest quadrilateral
    final = None
    for c in contours:
        eps = 0.02 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) == 4:
            final = approx
            break

    if final is None:  # Check if quadrilateral exists
        print("No document found...")
        exit()

    # Order points
    approx = order_points(final)

    # Output Image coordinates
    pts_dest = np.float32([[0, 0], [tgt_width, 0], [tgt_width, tgt_height], [0, tgt_height]])

    # Homogeneous Matrix
    M = cv2.getPerspectiveTransform(approx, pts_dest)

    # Peerspective Transformation
    final_image = cv2.warpPerspective(image, M, (tgt_width, tgt_height))

    # Display final image
    cv2.imshow("Scanned", final_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save output
    cv2.imwrite(args.output_image_path, final_image)
