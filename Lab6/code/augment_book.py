import cv2
import numpy as np
import argparse
import math


def draw_line(points, i, j, img):
    cv2.line(img, points[i], points[j], (0, 0, 255), 3)


def face_transform(src, dst_pts):
    # src_pts = [[0, 480], [640, 480], [0, 0], [640, 0]]
    src_pts = [[0, 0], [0, 480], [640, 0], [640, 480]]
    src_pts = np.array(src_pts, dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    dest = cv2.warpPerspective(src, matrix, (1280, 960))

    return dest


def draw_book_on_img(points, src):
    draw_line(points, 0, 1, src)
    draw_line(points, 0, 2, src)
    draw_line(points, 0, 4, src)
    draw_line(points, 1, 3, src)
    draw_line(points, 1, 5, src)
    draw_line(points, 2, 3, src)
    draw_line(points, 2, 6, src)
    draw_line(points, 3, 7, src)
    draw_line(points, 4, 5, src)
    draw_line(points, 4, 6, src)
    draw_line(points, 5, 7, src)
    draw_line(points, 6, 7, src)

def check_point_visible(pointA, pointB, pointC, pointD):
    a, b, c = get_line_params(pointA, pointB)
    sideC = a * pointC[0] + b * pointC[1] + c
    sideD = a * pointD[0] + b * pointD[1] + c
    
    if sideC*sideD > 0:
        return False
    else:
        return True

def get_line_params(pointA, pointB):
    x1, y1 = list(pointA)
    x2, y2 = list(pointB)
    y1 = y1
    y2 = y2
    a = -(y2-y1)/(x2-x1)
    b = 1
    c = x1*(y2-y1)/(x2-x1) - y1

    return a, b, c



def texture_book_on_img(points, src, args):
    front = cv2.imread(args.data + "/book_cover.jpg")
    front = cv2.resize(front, (640, 480))

    side = cv2.imread(args.data + "/side_image.jpg")
    side = cv2.resize(side, (640, 480))

    top_side = np.ones((480, 640, 3), dtype=np.uint8) * 255
    left_side = np.ones((480, 640, 3), dtype=np.uint8) * 255
    right_side = np.ones((480, 640, 3), dtype=np.uint8) * 255

    if check_point_visible(points[2], points[3], points[0], points[6]):
        dst_pts_tside = np.array([points[2], points[3], points[6], points[7]], dtype=np.float32)
        top_side = face_transform(top_side, dst_pts_tside)
        draw_face(top_side, src)

    if check_point_visible(points[1], points[3], points[0], points[5]):
        dst_pts_rside = np.array([points[1], points[5], points[3], points[7]], dtype=np.float32)
        right_side = face_transform(right_side, dst_pts_rside)
        draw_face(right_side, src)

    if check_point_visible(points[0], points[2], points[1], points[4]):
        dst_pts_lside = np.array([points[4], points[0], points[6], points[2]], dtype=np.float32)
        left_side = face_transform(left_side, dst_pts_lside)
        draw_face(left_side, src)
    
    if check_point_visible(points[0], points[1], points[2], points[4]):
        dst_pts_side = np.array([points[4], points[5], points[0], points[1]], dtype=np.float32)
        side = face_transform(side, dst_pts_side)
        draw_face(side, src)

    if points[0][1] != points[2][1]:
        dst_pts_face = np.array([points[0], points[1], points[2], points[3]], dtype=np.float32)
        front = face_transform(front, dst_pts_face)
        draw_face(front, src)

    


def draw_face(front, img):
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (front[i, j] == 0).all():
                continue
            else:
                img[i, j] = front[i][j]


def get_book_coords(args):  
    x = args.x
    y = args.y
    theta = args.theta

    x = 8*x/100
    y = 8*y/100
    theta = - math.pi * theta / 180
    
    width = args.w
    length = args.l
    breadth = args.b

    r_mtx = [[math.cos(theta), math.sin(theta), 0, x], [-math.sin(theta), math.cos(theta), 0, y], [0, 0, 1, 0]]

    book_coords = [[-length/2, -breadth/2, width], [length/2, -breadth/2, width], [-length/2, breadth/2, width], [length/2, breadth/2, width],
                    [-length/2, -breadth/2, 0], [length/2, -breadth/2, 0], [-length/2, breadth/2, 0], [length/2, breadth/2, 0]]

    for i in range(len(book_coords)):
        book_coords[i].append(1)
        book_coords[i] = np.dot(r_mtx,book_coords[i])

    return book_coords


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str)
    parser.add_argument("-w", type=float)
    parser.add_argument("-b", type=float)
    parser.add_argument("-l", type=float)
    parser.add_argument("-x", type=float, default=-1)
    parser.add_argument("-y", type=float, default=-1)
    parser.add_argument("-theta", type=float, default=-1)

    args = parser.parse_args()

    img1 = cv2.imread(args.data + "/img1.jpg")
    img2 = cv2.imread(args.data + "/img2.jpg")
    img3 = cv2.imread(args.data + "/img3.jpg")
    img4 = cv2.imread(args.data + "/img4.jpg")

    img1 = cv2.resize(img1, (1280, 960))
    img2 = cv2.resize(img2, (1280, 960))
    img3 = cv2.resize(img3, (1280, 960))
    img4 = cv2.resize(img4, (1280, 960))

    imgs = [img1, img2, img3, img4]

    view = 3

    """ 
    view = 0 i.e. img1 corresponds to front view of the wall.
    view = 1 i.e. img2 corresponds to top view of the wall.
    view = 2 i.e. img3 corresponds to bottom view of the wall. 
    view = 3 i.e. img4 corresponds to left side view of the wall.
    """

    pixel_coords = [[[337, 135], [613, 128], [888, 129], [349, 408], [625, 409], [885, 404]],
                    [[141, 285], [646, 283], [1145, 287], [225, 511], [661, 512], [1064, 510]],
                    [[257, 70], [633, 73], [996, 72], [225, 246], [649, 246], [1033, 242]]]

    view_4 = [[567, 153], [851, 167], [1110, 178], [581, 468], [846, 455], [1100, 451]]
    view_4 = np.array(view_4, dtype=np.float32)

    world_coords = []
    for _ in range(len(pixel_coords)):
        world_coords.append([[0, 4, 0], [4, 4, 0], [8, 4, 0], [0, 0, 0], [4, 0, 0], [8, 0, 0]])

    world_coords = np.array(world_coords, dtype=np.float32)
    pixel_coords = np.array(pixel_coords, dtype=np.float32)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_coords, pixel_coords, img1.shape[:2][::-1], None, None)

    width = args.w
    length = args.l
    breadth = args.b

    if args.x == -1:
        book_coords = [[0, 0, width], [length, 0, width], [0, breadth, width], [length, breadth, width],
                       [0, 0, 0], [length, 0, 0], [0, breadth, 0], [length, breadth, 0]]
    else:
        book_coords = get_book_coords(args)  

    book_coords = np.array(book_coords, dtype=np.float32)

    _, r, t = cv2.solvePnP(world_coords[0], view_4, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)

    if view != 3:
        pixel_book_coords, _ = cv2.projectPoints(book_coords, rvecs[view], tvecs[view], mtx, dist)
    else:
        pixel_book_coords, _ = cv2.projectPoints(book_coords, r, t, mtx, dist)

    points = []

    for point in pixel_book_coords:
        x = round(point[0][0])
        y = round(point[0][1])
        points.append((x, y))

    draw_book_on_img(points, imgs[view])
    texture_book_on_img(points, imgs[view], args)

    cv2.imshow("output", imgs[view])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite(args.data + "/augment_book.jpg", imgs[view])