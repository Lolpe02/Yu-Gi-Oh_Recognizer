import cv2
import numpy as np
import re

def print_points(box, im2):
    box_a = box.reshape(4, 2).astype(np.int32)
    for a, p in enumerate(box_a):
        print(p)
        im2 = cv2.circle(im2, p, 3, (a * 63, a * 63, a * 63), -1)
        im2 = cv2.putText(im2, str(a) + str(p), p, cv2.FONT_HERSHEY_SIMPLEX, 1, (a * 63, a * 63, a * 63), 6)
        a += 1


def pca(box, pt4):
    # tl, tr, br, bl = box
    # x, y, alpha = cv2.minAreaRect(box)
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(box)
    data_pts = np.empty((sz, 2), dtype=np.float32)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = box[i, 0, 0]
        data_pts[i, 1] = box[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)

    # Store the center of 4 points
    cntr = (np.mean(pt4, axis=0))

    # print(eigenvectors)
    atan = np.arctan(eigenvectors[1, 0] / eigenvectors[0, 0])
    angle = np.degrees(atan)
    # if box[1, 0] - box[0, 0] > box[2, 1] - box[1, 1]: # if width > height
    #
    #     # print(box[1, 0] - box[0, 0], box[2, 1] - box[1, 1])
    #     box = np.array([box[3], box[0], box[1], box[2]], dtype=np.float32)
    #     print("ordering")
    # # detect if rectangle is left or right oriented
    # if pts[0][0] < pts[1][0]:
    #     left = pts[0]
    #     right = pts[1]  # right
    # else:
    #     left = pts[1]
    #     right = pts[0]
    # if pts[2][0] < pts[3][0]:
    #     left2 = pts[2]
    #     right2 = pts[3]
    # else:
    #     left2 = pts[3]
    #     right2 = pts[2]
    # # detect if rectangle is up or down oriented
    # if pts[0][1] < pts[2][1]:
    #     up = pts[0]
    #     down = pts[2]
    # else:
    #     up = pts[2]
    #     down = pts[0]
    # if pts[1][1] < pts[3][1]:
    #     up2 = pts[1]
    #     down2 = pts[3]
    # else:
    #     up2 = pts[3]
    #     down2 = pts[1]
    return cntr, angle


def find_corner_points(contour, method=1):
    if len(contour) < 4:
        print("not enough points")
        return None

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
    vertices = cv2.approxPolyDP(contour, 0.1 * perimeter, True)

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
    # Calculate aspect ratio of the bounding rectangle
    aspect_ratio = width / height
    # Check if the contour has 4 vertices and aspect ratio close to 1

    return True #aspect_ratio >= card_ratio - 0.3 and aspect_ratio <= card_ratio + 0.3


def hsv_thresh(img, kernel):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret, thresh_H = cv2.threshold(hsv[:, :, 0]*2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, thresh_S = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, thresh_V = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # --- add the result of the above two ---

    t = thresh_H + thresh_S + thresh_V
    neg = cv2.bitwise_not(t)

    # --- some morphology operation to clear unwanted spots ---
    dilation = cv2.dilate(neg, kernel, iterations=4)
    # thresh_V = cv2.morphologyEx(thresh_V, cv2.MORPH_ERODE, kernel, iterations=10)  # erode(thresh_V, kernel, iterations=5)
    # thresh_S = cv2.morphologyEx(thresh_S, cv2.MORPH_CLOSE, kernel, iterations=10)  # erode(thresh_V, kernel, iterations=5)

    # cv2.imshow('h', cv2.resize(thresh_H, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
    # cv2.imshow('s', cv2.resize(thresh_S, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
    # cv2.imshow('v', cv2.resize(thresh_V, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))



    # calculate histogram of black and white pixels
    hist = cv2.calcHist([thresh_V], [0], None, [2], [0, 256])
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_V, connectivity=4)
    total_pixels = thresh_V.shape[0] * thresh_V.shape[1]

    # Calculate the number of white pixels (pixels with a value of 255)
    white_pixels = np.sum(thresh_V == 255)
    # cv2.imshow("lables",cv2.resize(np.uint8(255 * (labels+50) / np.max(labels)), None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC ))

    # Calculate the sparsity of white pixels
    white_pixel_ratio = white_pixels / total_pixels
    # print(hist, white_pixels, "\n", len(stats), "\n", len(centroids))

    if white_pixel_ratio < 0.4:
        print("white pixel sparsity is too low")
        return [gray]

    # thresh_V = cv2.morphologyEx(thresh_V, cv2.MORPH_CLOSE, kernel, iterations=20)  # erode(thresh_V, kernel, iterations=5)
    stacked = np.dstack((thresh_H, thresh_S, thresh_V))
    rgbstacked = cv2.cvtColor(stacked, cv2.COLOR_HSV2BGR)
    r, graystacked = cv2.threshold(cv2.cvtColor(rgbstacked, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # cv2.imshow('graystacked', cv2.resize(graystacked, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
    return graystacked, dilation, gray


def get_name(result):
    name = re.sub(r"[-()\"#/@;:*_<£/\n>{}`+=~|.!?,]", "", result).lstrip().rstrip()
    return name


def find_optimal_canny_threshold(im, test=0, threshold_range=(0, 601), step=30):
    best_threshold1 = None
    best_threshold2 = None
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
    print(find_optimal_canny_threshold('carta3.jpeg', 1, step=1))
    # s = set([(20, 160), (20, 240), (30, 170), (30, 250), (50, 190), (50, 200), (50, 210), (60, 210), (60, 240), (60, 250), (80, 190), (150, 110), (160, 70), (230, 90)(70, 50), (100, 30), (130, 20), (150, 10), (190, 10), (190, 20), (200, 20), (210, 20), (210, 40), (220, 10), (220, 20),(20, 170), (20, 180), (200, 70), (200, 150), (210, 120), (220, 80), (230, 70), (230, 80)])
    # print(s)
