import cv2
import numpy as np
import re

def print_points(box, im2):
    box_a = box.reshape(4, 2).astype(np.int32)
    for a, p in enumerate(box_a):
        # print(p)
        im2 = cv2.circle(im2, p, 3, (a * 63, a * 63, a * 63), -1)
        im2 = cv2.putText(im2, str(a) + str(p), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (a * 63, a * 63, a * 63), 6)
        a += 1

def order(w, h , pts, angle):
    # print(angle)
    tl, tr, br, bl = find_corner_points(pts)[0]
    temp_rect = np.zeros((4, 2), dtype="float32")
    pts = pts.reshape(-1, 1, 2)
    if w <= 0.95 * h:  # If card is vertically oriented
        temp_rect[0] = tl
        temp_rect[1] = tr
        temp_rect[2] = br
        temp_rect[3] = bl

    if w >= 1.05 * h:  # If card is horizontally oriented
        if pts[0][0][0] < pts[2][0][0]: #left tilt
            temp_rect[0] = bl
            temp_rect[1] = tl
            temp_rect[2] = tr
            temp_rect[3] = br

        else: #right tilt
            print("right tilt")
            temp_rect[0] = tr
            temp_rect[1] = br
            temp_rect[2] = bl
            temp_rect[3] = tl


        # If the card is 'diamond' oriented, a different algorithm
        # has to be used to identify which point is top left, top right
        # bottom left, and bottom right.

    if w > 0.95 * h and w < 1.05 * h:  # If card is diamond oriented
        # If furthest left point is higher than furthest right point,
        # card is tilted to the left.
        if pts[1][0][1] <= pts[3][0][1]:
            # If card is titled to the left, approxPolyDP returns points
            # in this order: top right, top left, bottom left, bottom right
            temp_rect[0] = pts[1][0]  # Top left
            temp_rect[1] = pts[0][0]  # Top right
            temp_rect[2] = pts[3][0]  # Bottom right
            temp_rect[3] = pts[2][0]  # Bottom left

        # If furthest left point is lower than furthest right point,
        # card is tilted to the right
        if pts[1][0][1] > pts[3][0][1]:
            # If card is titled to the right, approxPolyDP returns points
            # in this order: top left, bottom left, bottom right, top right
            temp_rect[0] = pts[0][0]  # Top left
            temp_rect[1] = pts[3][0]  # Top right
            temp_rect[2] = pts[2][0]  # Bottom right
            temp_rect[3] = pts[1][0]  # Bottom left
    return temp_rect

def find_corner_points(contour, method=1):
    if len(contour) < 4:
        # print("not enough points")
        return None
    contour = contour.reshape(-1, 1, 2)
    if method:
        s = np.sum(contour, axis=2)
        diff = np.diff(contour, axis=-1)

        # Find the point with the lowest sum of coordinates (top-left corner)
        top_left = tuple(contour[np.argmin(s)])

        # Find the point with the highest difference of coordinates (top-right corner)
        top_right = tuple(contour[np.argmin(diff)])

        # Find the point with the highest sum of coordinates (bottom-right corner)
        bottom_right = tuple(contour[np.argmax(s)])

        # Find the point with the lowest difference of coordinates (bottom-left corner)
        bottom_left = tuple(contour[np.argmax(diff)])

        return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32).reshape(4, 2), method

    elif not method:
        left_most = tuple(contour[contour[:, :, 0].argmin()][0])

        right_most = tuple(contour[contour[:, :, 0].argmax()][0])

        up_most = tuple(contour[contour[:, :, 1].argmin()][0])

        bottom_most = tuple(contour[contour[:, :, 1].argmax()][0])

        return np.array([up_most, right_most, bottom_most, left_most], dtype=np.int32).reshape(4, 2), method


def is_rectangle(contour):
    # Calculate contour properties
    perimeter = cv2.arcLength(contour, True)
    vertices = cv2.approxPolyDP(contour, 0.07 * perimeter, True)

    # print("done")
    if len(vertices) != 4:
        print("not 4 vertices")
        return False
    vertices.reshape(4, 2)  # tl, tr, br, bl

    # calculate distance between two points
    height1, height2 = cv2.norm(vertices[::2, :], cv2.NORM_L2), cv2.norm(vertices[1::2, :], cv2.NORM_L2)
    width1, width2 = cv2.norm(vertices[:2, :], cv2.NORM_L2), cv2.norm(vertices[3:, :], cv2.NORM_L2)
    # print(width1/ width2, height1/ height2)
    width, height = (width1 + width2) / 2, (height1 + height2) / 2

    card_ratio = 590 / 860
    # print(card_ratio, width / height)
    # Calculate aspect ratio of the bounding rectangle
    aspect_ratio = width / height
    # Check if the contour has 4 vertices and aspect ratio close to 1

    return aspect_ratio >= card_ratio - 0.2 and aspect_ratio <= card_ratio + 0.2


def hsv_thresh(img, kernel):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _, thresh_H = cv2.threshold(hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh_S = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh_V = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow("thresh_H", cv2.resize(thresh_H, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
    # cv2.imshow("thresh_S", cv2.resize(thresh_S, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
    # cv2.imshow("thresh_V", cv2.resize(thresh_V, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
    neg = cv2.bitwise_not(thresh_H + thresh_S + thresh_V)
    dilation = cv2.dilate(neg, kernel, iterations=4)

    thresh_rgb = cv2.cvtColor(np.dstack((thresh_H, thresh_S, thresh_V)), cv2.COLOR_HSV2BGR)
    r, graystacked = cv2.threshold(cv2.cvtColor(thresh_rgb, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return graystacked, dilation, gray


def get_name(result):
    name = re.sub(r"[-()\"#/@—;:*®_<£/\n>{}`+=~|.!?,]", "", result).lstrip().rstrip()
    return name

def last_resort(img):
    # blur_img = cv2.bilateralFilter(img, 9, 10, 10)
    lato = cv2.cvtColor(np.asarray([(130, 135, 157), (67, 70, 87), (73, 77, 89), (46, 46, 56), (114, 116, 124), (61, 62, 74), (32, 35, 50),
                (129, 139, 153), (131, 142, 180), (20, 20, 35), (87, 102, 137)], dtype=np.uint8).reshape(-1, 1, 3), cv2.COLOR_RGB2HSV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #H: 0-179, S: 0-255, V: 0-255
    last_resort = cv2.inRange(hsv, (0 ,10, 70),(180,  170,  140)) #(0 ,10, 50),(200,  100,  180)
    # cv2.imshow("last_resort", cv2.resize(last_resort, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
    # cv2.waitKey(0)
    return last_resort

def find_optimal_canny_threshold(im, test=0, threshold_range=(0, 601), step=30):
    l = []
    kernel = np.ones((2, 2), np.uint8)
    im = cv2.imread(im)
    # image = cv2.GaussianBlur(cv2.imread(im), (3, 3), 0)
    image = cv2.bilateralFilter(im, 5, 10, 10)
    if test:
        step = 15
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ret, thresh_H = cv2.threshold(hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, thresh_S = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ret, thresh_V = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = thresh_V - thresh_H - thresh_S
        neg = cv2.bitwise_not(t)
        stacked = np.dstack((thresh_H, thresh_S, thresh_V))
        image = cv2.cvtColor(stacked, cv2.COLOR_HSV2BGR)

    for threshold1 in range(threshold_range[0], threshold_range[1] - 1, 3 * step):
        for threshold2 in range(threshold_range[0], threshold_range[1], step):
            edges = cv2.Canny(image, threshold1, threshold2, None, 3, False)
            edges = cv2.dilate(edges, kernel, iterations=4)
            sedges = cv2.rectangle(edges, (50, 50), (1100, 150), (255, 255, 255), -1, 8)
            sedges = cv2.putText(sedges, f"t1= {threshold1} , t2= {threshold2}", (80, 90),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
            sedges = cv2.putText(sedges, f"q = quit, enter = try, any key = next",
                                 (80, 130),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
            cv2.imshow("cannyHSV" if test else "cannyOG",
                       cv2.resize(sedges, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_CUBIC))

            key = cv2.waitKey(0) & 0xFF
            # wait for user input
            if key == 13:
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best_threshold1 = threshold1
                best_threshold2 = threshold2
                l.append((best_threshold1, best_threshold2))
                for c in contours:
                    if cv2.contourArea(c) > 450 and is_rectangle(c):  #
                        image = cv2.drawContours(image, contours, -1, (0, 255, 0), 4)
                cv2.imshow("contours", cv2.resize(image, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))

            elif key == ord("q"):
                return l
            else:

                continue

    return l


def find_optimal_hsv_threshold(im, testvaluelist=None, step=15):  # []
    kernel = np.ones((1, 1), np.uint8)
    l = []
    im = cv2.imread(im)
    area1 = np.multiply(*im.shape[:2]) - 1
    # im = cv2.GaussianBlur(im, (3, 3), 0)
    im = cv2.bilateralFilter(im, 5, 10, 10)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    if testvaluelist:
        for test in testvaluelist:
            ret, thresh_H = cv2.threshold(hsv[:, :, 0], test[0], 255, cv2.THRESH_BINARY)
            ret, thresh_S = cv2.threshold(hsv[:, :, 1], test[1], 255, cv2.THRESH_BINARY)
            ret, thresh_V = cv2.threshold(hsv[:, :, 2], test[3], 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # --- add the result of the above two ---
            t = thresh_V - thresh_H + thresh_S
            neg = cv2.bitwise_not(t)
            # --- some morphology operation to clear unwanted spots ---
            dilation = cv2.dilate(neg, kernel, iterations=5)
            cv2.imshow(f"added {test}", cv2.resize(t, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
            cv2.imshow(f"neg {test}", cv2.resize(dilation, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
        cv2.waitKey(0)
        return
    for threshold1 in range(0, 255, step):
        for threshold2 in range(0, 255, step):
            for threshold3 in range(0, 255, step):
                ret, thresh_H = cv2.threshold(hsv[:, :, 0], threshold1, 255, cv2.THRESH_BINARY)
                ret, thresh_S = cv2.threshold(hsv[:, :, 1], threshold2, 255, cv2.THRESH_BINARY)
                ret, thresh_V = cv2.threshold(hsv[:, :, 2], threshold3, 255, cv2.THRESH_BINARY)
                # --- add the result of the above two ---
                t = thresh_V - thresh_H - thresh_S
                neg = cv2.bitwise_not(t)
                # --- some morphology operation to clear unwanted spots ---
                dilation = cv2.dilate(neg, kernel, iterations=5)
                cv2.imshow("added", cv2.resize(t, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
                cv2.imshow("neg", cv2.resize(dilation, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
                sedges = cv2.rectangle(dilation + t, (50, 50), (1100, 150), (255, 255, 255), -1, 8)
                sedges = cv2.putText(sedges, f"t1={threshold1} , t2={threshold2}, t3={threshold3}", (80, 90),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
                sedges = cv2.putText(sedges, f"q = quit, enter = try, any key = next",
                                     (80, 130),
                                     cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
                cv2.imshow("edges", cv2.resize(sedges, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_CUBIC))

                key = cv2.waitKey(0)
                # wait for user input
                if key == 13:
                    contours1, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours2, hierarchy = cv2.findContours(t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    best_threshold1 = threshold1
                    best_threshold2 = threshold2
                    l.append((best_threshold1, best_threshold2))
                    for c in contours1:
                        area = cv2.contourArea(c)
                        if area > 450 and area < area1 and is_rectangle(c):  #
                            im = cv2.drawContours(im.copy(), contours1, -1, (0, 255, 0), 2)
                    for c in contours2:
                        area = cv2.contourArea(c)
                        if area > 450 and area < area1 and is_rectangle(c):  #
                            im = cv2.drawContours(im, contours2, -1, (0, 0, 255), 2)
                    cv2.imshow("contours", cv2.resize(im, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
                elif key == ord("q"):
                    return l
                elif key == ord("d"):
                    l.pop()
                else:

                    continue

    return l


carta5 = [(20, 130), (20, 250), (30, 250), (40, 250), (90, 120), (120, 60), (120, 120), (130, 60), (200, 70),
          (200, 120), (220, 70), (220, 100), (250, 70)]
carta11 = [(40, 30), (60, 20), (190, 10)]

if __name__ == '__main__':
    # print(find_optimal_canny_threshold('cards/carta3.jpeg', 1, step=1))
    print(last_resort(cv2.imread('cards/carta8.jpeg')))
    # s = set([(20, 160), (20, 240), (30, 170), (30, 250), (50, 190), (50, 200), (50, 210), (60, 210), (60, 240), (60, 250), (80, 190), (150, 110), (160, 70), (230, 90)(70, 50), (100, 30), (130, 20), (150, 10), (190, 10), (190, 20), (200, 20), (210, 20), (210, 40), (220, 10), (220, 20),(20, 170), (20, 180), (200, 70), (200, 150), (210, 120), (220, 80), (230, 70), (230, 80)])
    # print(s)
