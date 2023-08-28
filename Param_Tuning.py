import cv2
import numpy as np


def find_corner_points(contour,method=1):
    if len(contour) < 4:
        print("not enough points")
        return None, None, None, None

    if method :
        # Find the point with the lowest sum of coordinates (top-left corner)
        top_left = tuple(contour[np.argmin(contour.sum(axis=2))])

        # Find the point with the highest difference of coordinates (top-right corner)
        top_right = tuple(contour[np.argmax(np.diff(contour, axis=2))])

        # Find the point with the highest sum of coordinates (bottom-right corner)
        bottom_right = tuple(contour[np.argmax(contour.sum(axis=2))])

        # Find the point with the lowest difference of coordinates (bottom-left corner)
        bottom_left = tuple(contour[np.argmin(np.diff(contour, axis=2))])

        return top_left, top_right, bottom_right, bottom_left
    elif not method:
        top_left = tuple(contour[contour[:, :, 0].argmin()][0])

        top_right = tuple(contour[contour[:, :, 0].argmax()][0])

        bottom_right = tuple(contour[contour[:, :, 1].argmin()][0])

        bottom_left = tuple(contour[contour[:, :, 1].argmax()][0])

        return top_left, top_right, bottom_right, bottom_left

def is_rectangle(contour):
    # Calculate contour properties
    perimeter = cv2.arcLength(contour, True)
    vertices = cv2.approxPolyDP(contour, 0.1 * perimeter, True)
    x, y, width, height = cv2.boundingRect(vertices)
    card_ratio = 590 / 860
    # Calculate aspect ratio of the bounding rectangle
    aspect_ratio = width / height

    # Check if the contour has 4 vertices and aspect ratio close to 1
    return len(vertices) == 4 and aspect_ratio >= card_ratio - 0.3 and aspect_ratio <= card_ratio + 0.3


def find_optimal_canny_threshold(im ,threshold_range=(0, 500), step=30):
    best_threshold1 = None
    best_threshold2 = None
    l = []
    im = cv2.imread(im)
    # image = cv2.GaussianBlur(cv2.imread(im), (3, 3), 0)
    image = cv2.bilateralFilter(im, 5, 10, 10)
    for threshold1 in range(threshold_range[0], threshold_range[1] - 1, step):
        for threshold2 in range(threshold1+ step, threshold_range[1], step):
            edges = cv2.Canny(image, threshold1, threshold2, None, 3, True)
            sedges = cv2.rectangle(edges, (50, 50), (1100, 150), (255, 255, 255), -1, 8)
            sedges = cv2.putText(sedges, f"t1= {threshold1} , t2= {threshold2}", (80, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
            sedges = cv2.putText(sedges, f"q = quit, enter = try, any key = next",
                                (80, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
            cv2.imshow("edges", cv2.resize(sedges, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_CUBIC))

            key = cv2.waitKey(0)
            # wait for user input
            if key == 13:
                contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best_threshold1 = threshold1
                best_threshold2 = threshold2
                l.append((best_threshold1, best_threshold2))
                for c in contours:
                    if cv2.contourArea(c) > 450 :  #and is_rectangle(c)
                        new =  cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)
                cv2.imshow("contours", cv2.resize(new, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
                cv2.waitKey(5000)
            elif key == ord("q"):
                return l
            else:

                continue

    return l
def find_optimal_hsv_threshold(im , step=10):# [(20, 170), (20, 180), (200, 70), (200, 150), (210, 120), (220, 80), (230, 70), (230, 80)]
    kernel = np.ones((1, 1), np.uint8)
    l = []
    im = cv2.imread(im)
    # im = cv2.GaussianBlur(im, (3, 3), 0)
    im = cv2.bilateralFilter(im, 5, 10, 10)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    for threshold1 in range(0, 255 , step):
        for threshold2 in range(0,255, step):

            ret, thresh_H = cv2.threshold(hsv[:, :, 0], threshold1, 255, cv2.THRESH_BINARY  )
            ret, thresh_S = cv2.threshold(hsv[:, :, 1], threshold2, 255, cv2.THRESH_BINARY )

            # --- add the result of the above two ---

            sumt = thresh_H + thresh_S
            neg = cv2.bitwise_not(sumt)
            # --- some morphology operation to clear unwanted spots ---
            dilation = cv2.dilate(sumt, kernel, iterations=10)
            cv2.imshow('thresh', cv2.resize(sumt, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_CUBIC))
            # edges = cv2.Canny(dilation, 120, 400)
            sedges = cv2.rectangle(dilation, (50, 50), (1100, 150), (255, 255, 255), -1, 8)
            sedges = cv2.putText(sedges, f"t1= {threshold1} , t2= {threshold2}", (80, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
            sedges = cv2.putText(sedges, f"q = quit, enter = try, any key = next",
                                (80, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 6)
            cv2.imshow("edges", cv2.resize(sedges, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_CUBIC))

            key = cv2.waitKey(0)
            # wait for user input
            if key == 13:
                contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                best_threshold1 = threshold1
                best_threshold2 = threshold2
                l.append((best_threshold1, best_threshold2))
                for c in contours:
                    if cv2.contourArea(c) > 450 and is_rectangle(c) :  #
                        im =  cv2.drawContours(im.copy(), contours, -1, (0, 255, 0), 2)
                cv2.imshow("contours", cv2.resize(im, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
                cv2.waitKey(5000)
            elif key == ord("q"):
                return l
            else:

                continue

    return l

if __name__ == '__main__':
    print(find_optimal_hsv_threshold('carta11.jpeg', ))