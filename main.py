import cv2
import pytesseract as pt
from PIL import Image
import numpy as np
import Param_Tuning as pmt

# import nltk


pt.pytesseract.tesseract_cmd = r'D:\Program Files\TesseractOCR\tesseract.exe'
# print(pt.get_languages())
namelist = []
kernel = np.ones((2,2), np.uint8)
img = cv2.imread('carta2.jpeg')
blur_img = cv2.bilateralFilter(img, 9, 10, 10)
# print(img.shape)
def hsv_thresh(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ret, thresh_H = cv2.threshold(hsv[:, :, 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, thresh_S = cv2.threshold(hsv[:, :, 1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, thresh_V = cv2.threshold(hsv[:, :, 2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # --- add the result of the above two ---

    t = -thresh_V - thresh_H - thresh_S
    neg = cv2.bitwise_not(t)
    # cv2.imshow('threshsum', cv2.resize(neg, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))

    # --- some morphology operation to clear unwanted spots ---
    dilation = cv2.dilate(neg, kernel, iterations=5)
    # add 3 axis to the 2d array

    stacked = np.dstack((thresh_H, thresh_S, thresh_V))
    rgbstacked = cv2.cvtColor(stacked, cv2.COLOR_HSV2RGB)
    return rgbstacked, dilation, img

# img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
# img = cv2.GaussianBlur(img, (3, 3), 0.1)




canny = cv2.Canny(hsv_thresh(), 0, 700)   #[(0, 480), (30, 480), (120, 150), (120, 450), (140, 481)]
# edges2 = cv2.Canny(neg, 500, 600)
new = cv2.dilate(canny, kernel, iterations=4)
cv2.imshow("canny", cv2.resize(new, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
# cv2.imshow("c2", cv2.resize(edges2, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
# cv2.waitKey(0)
contours, hierarchy = cv2.findContours(new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# print(len(contours))
if contours is None:
    print("no contours found")
    exit()
im2 = img.copy()
count = 0
failed = 0
# Define the dimensions of the output image (width, height)
output_width, output_height = 590, 860
# top_left, top_right, bottom_right, bottom_left
output_corners = np.float32([[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]])

for c in contours:
    if cv2.contourArea(c) > 1800:#: and pmt.is_rectangle(c)
        print("found a rectangle")
        cv2.drawContours(im2, [c], -1, (0, 255, 0), 2)
        # HL, LR, alpha = cv2.minAreaRect(c)
        # always clockwise starting from the top left tl, tr, br, bl

        box = pmt.find_corner_points(c).astype(np.float32)
        a = 0
        vertices = cv2.approxPolyDP(c, 0.3 * cv2.arcLength(c, True), True)
        print(vertices)
        # vertices.reshape(4, 2)
        for p in box.astype(np.int32):
            print(p)
            im2 = cv2.circle(im2, p, 5, (a * 63, a * 63, a * 63), -1)
            im2 = cv2.putText(im2, str(a) + str(p), p, cv2.FONT_HERSHEY_SIMPLEX, 1.6, (a * 63, a * 63, a * 63), 6)
            a += 1

        # TODO: check if the box is clockwise or not and trasform as needed

        if box[1,0] - box[0,0] > box[2,1] - box[1,1]:
            print(box[1, 0] - box[0, 0], box[2, 1] - box[1, 1])
            box = np.array([box[3], box[0], box[1], box[2]], dtype=np.float32)
            print("ordering\n"+str(box))


        count += 1
        transformation_matrix = cv2.getPerspectiveTransform(box, output_corners)
        warped_image = cv2.warpPerspective(img.copy(), transformation_matrix, (output_width, output_height))
        x, y = warped_image.shape[:2]
        # name = warped_image[x//16:x//7,y//12:y*33//40]
        name = warped_image[:x // 7, :]
        # sharpening with blur and kernel
        # blurred = cv2.GaussianBlur(name, (3, 3), 0)
        # name = cv2.addWeighted(name, 1.5, blurred, -0.5, 0)

        # kernel = np.array([[1, -1, -1], [0, 5, -1], [1, -1,-1]])
        # warped_image = cv2.filter2D(warped_image, -1, kernel)
        # resize
        # warped_image = cv2.resize(warped_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        text = pt.image_to_string(name, lang='ita')
        print(text)
        cv2.imshow(f"name of {count}", cv2.resize(name, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))

    else:
        # cv2.drawContours(im2, [c], -1, (0,0,255), 8)
        # print("smt wrongobongo")
        failed += 1


cv2.imshow("contours", cv2.resize(im2, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
print(f"found {count} cards and {failed} failed")
# im2 = img.copy()
# lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
# for line in lines:
#     rho,theta = line[0]
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 1000*(-b))
#     y1 = int(y0 + 1000*(a))
#     x2 = int(x0 - 1000*(-b))
#     y2 = int(y0 - 1000*(a))
#     print((x1, y1), (x2, y2))
#     cv2.line(im2,(x1,y1),(x2,y2),(0,0,255),2)
# cv2.imshow('cards_output', im2)
# cv2.waitKey(0)
# print(np.linspace(-1,11))

# img = img[::-1,::-1] # inverted img
# cv2.namedWindow("Img", cv2.WINDOW_KEEPRATIO)
#
# src_pts = []
# def onClick(event,x,y,flags,param):
#     corners = ["choose top left","choose top right","choose bottom right","choose bottom left", "done"]
#     if event == cv2.EVENT_LBUTTONDOWN:
#
#         if len(src_pts) <4:
#             src_pts.append([x,y])
#             put = cv2.circle(img.copy(), (x,y), 10, (0,0,255), -1)
#             put = cv2.putText(put, corners[len(src_pts)], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#             cv2.imshow("Img", put)
#     else :
#         put = cv2.putText(img.copy(), corners[len(src_pts)], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.imshow("Img", put)
#
# # points for 0.5 scale [[181, 306], [426, 370], [341, 730], [82, 664]]
# # points for 1 scale   [[362, 612], [852, 740],[682, 1460],[164, 1328]]
#
# # homogenous trasform
# # cv2.imshow("Img",img )
# # cv2.setMouseCallback("Img", onClick)
# # cv2.waitKey(0)
# original_corners = np.array([[362, 612], [852, 740],[682, 1460],[164, 1328]], dtype=np.float32)
# # print(src_pts)
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#
# # sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
# # sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
# # sobel = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)


#
# Define the coordinates of the corresponding corners in the output image
# output_corners = np.float32([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]])
#
# Compute the perspective transformation matrix
# transformation_matrix = cv2.getPerspectiveTransform(original_corners, output_corners)
#
# Apply the perspective transformation
# warped_image = cv2.warpPerspective(img, transformation_matrix,(output_width, output_height))
#
#
# #--- convert the image to HSV color space ---
# hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# # find hsv of points
# # for point in [[191, 319], [128, 666], [360, 620], [401, 442]]:
# #     print(hsv[point[1],point[0]]) #why not [point[0],point[1]]? because opencv is BGR not RGB
# # [65, 27, 103],[38, 18, 57],[49, 61, 121],[55, 29, 97]
# cv2.imshow('H', hsv[:,:,0])
# cv2.imshow('S', hsv[:,:,1])
#
# #--- find Otsu threshold on hue and saturation channel ---
# ret, thresh_H = cv2.threshold(hsv[:,:,0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ret, thresh_S = cv2.threshold(hsv[:,:,1], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#
# #--- add the result of the above two ---
# cv2.imshow('thresh', thresh_H + thresh_S)
#
# #--- some morphology operation to clear unwanted spots ---

# neg = cv2.bitwise_not(thresh_H + thresh_S)
# cv2.imshow('neg', thresh_H + thresh_S)

# cv2.imshow('dilation+bitwise not', dilation)
#
# #--- find contours on the result above ---
# (contours, hierarchy) = cv2.findContours(neg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# #--- since there were few small contours found, retain those above a certain area ---
# im2 = img.copy()
# count = 0
# for c in contours:
#     if cv2.contourArea(c) > 650:
#         count+=1
#         cv2.drawContours(im2, [c], -1, (0, 255, 0), -1)
##
# detect corners with new algorithm
# corners = cv2.goodFeaturesToTrack(gray, 10, qualityLevel=0.05, minDistance= 60, mask=None, blockSize=3, gradientSize= 1, useHarrisDetector= False, k=0.4)
# if corners is None:
#     print("no corners found")
#     exit()
# # Draw corners on the original image
# for corner in corners:
#     x, y = corner.ravel()
#     cv2.circle(im2, (int(x), int(y)), 3, (0, 255, 0), -1)
#
# cv2.imshow('cards_output', im2)
# print('There are {} cards'.format(count))
cv2.waitKey(0)
#
# # cv2.imshow('img', warped_image)
# # cv2.waitKey(0)
# # text = pt.image_to_string(warped_image, lang='ita')
# # print(text)
# # print("tibo can u see this?")
