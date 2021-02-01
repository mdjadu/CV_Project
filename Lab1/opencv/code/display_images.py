import argparse
import cv2
from os import listdir

parser = argparse.ArgumentParser()
parser.add_argument("dir_path", help="path to directory",
                    type=str)
args = parser.parse_args()

# Get list of file
files = listdir(args.dir_path)
images = []
for f in files:
    if f[-4:] == ".png" or f[-4:] == ".jpg":  # Check for image files
        images.append(f)

cur_idx = 0
img_path = args.dir_path + "\\{}"

while True:
    img = cv2.imread(img_path.format(images[cur_idx]), cv2.IMREAD_COLOR)
    cv2.imshow('Image', img)
    code = cv2.waitKey(0)
    if code == 110:  # n key code
        cur_idx = (cur_idx + 1) % len(images)
    elif code == 112:  # p key code
        cur_idx = (cur_idx - 1) % len(images)
    elif code == 27 or code == 113:  # Escape / q key code
        cv2.destroyAllWindows()
        break