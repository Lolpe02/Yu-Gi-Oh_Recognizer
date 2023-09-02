import cv2
import pytesseract as pt
from PIL import Image
import numpy as np
import Param_Tuning as pmt

import joblib

# import nltk


pt.pytesseract.tesseract_cmd = r'D:\Program Files\TesseractOCR\tesseract.exe'
# print(pt.get_languages())
namelist = []
kernel = np.ones((2,2), np.uint8)
img = cv2.imread('cards/carta12.jpeg')
blur_img = cv2.bilateralFilter(img, 9, 10, 10)
# print(img.shape)


# img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
# img = cv2.GaussianBlur(img, (3, 3), 0.1)



processed = pmt.hsv_thresh(blur_img, kernel)[0]
canny = cv2.Canny(processed,60,300)   #[(0, 480), (30, 480), (120, 150), (120, 450), (140, 481)]

# new = cv2.dilate(canny, kernel, iterations=3)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
new= cv2.morphologyEx(canny, cv2.MORPH_CROSS, kernel, iterations=3)

# cv2.imshow("canny", cv2.resize(new, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
# cv2.imshow("processed", cv2.resize(cv2.bitwise_not(processed) , None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
# test = cv2.waitKey(0) & 0xFF
# if test == ord("q"):
#     exit()

contours, hierarchy = cv2.findContours(cv2.bitwise_not(processed), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
black = np.zeros_like(im2, dtype=np.uint8)

for c in contours:
    if cv2.contourArea(c) > 2000   and  pmt.is_rectangle(c) :#:  np.array([False]).all()
        # print("found a rectangle")
        cv2.drawContours(im2, [c], -1, (0, 255, 0), 2)
        # always clockwise starting from the top left tl, tr, br, bl

        # MY METHOD
        # box, method = pmt.find_corner_points(c, 1)
        # a = 0

        # APPROX METHOD
        box = cv2.approxPolyDP(c, 0.06 * cv2.arcLength(c, True), True)
        if len(box) != 4:
            continue
        # box = pmt.find_corner_points(box)[0]
        # print(box)


        # MIN AREA RECT METHOD
        # x, y, alpha = cv2.minAreaRect(c)
        # box = cv2.boxPoints((x, y, alpha))

        # HULL + minrect METHOD
        # hull = cv2.convexHull(c, returnPoints=False) #indexes of the points
        # defects = cv2.convexityDefects(c, hull) #indexes of the points
        # #order defects by length
        # defects = sorted(defects, key=lambda x: (cv2.norm(c[[x[0][1],x[0][0]]], normType= cv2.NORM_L2),x[0][3]  ), reverse=True)
        # # reduce chosens from 4 dimensions to 3
        # chosens = c[hull].squeeze(axis=2)
        # # cv2.drawContours(black, chosens, -1, (255, 255, 255), 3)
        # boolean_mask = np.full(c.shape[0], True, dtype=np.bool_)
        # for defect in defects:
        #     s, e, f, d = defect[0]
        #     if d < 300:
        #         continue
        #     # start = tuple(c[s][0])
        #     # end = tuple(c[e][0])
        #     # delete from start to end and add the new point
        #
        #     boolean_mask[s+1:e] = False
        # #
        # new_c = c[boolean_mask] #
        # # print(len(boolean_mask),len(c),len(new_c))
        # cv2.drawContours(black, new_c, -1, (255, 255, 255), 3)
        # x, y, alpha = cv2.minAreaRect(new_c)
        # box = cv2.boxPoints((x, y, alpha))
        box =box.reshape(4, 2)
        # center of 4 points

        # cntr = np.array([np.mean(box[:, 0]), np.mean(box[:, 1])])


        # if the first point is on the left of the second point

        pmt.print_points(box, im2)
        print(box.shape)
        # find longest side
        if cv2.norm(box[0], box[1], normType=cv2.NORM_L2) > cv2.norm(box[0], box[3], normType=cv2.NORM_L2):
            angle = np.arctan((box[1][1] - box[0][1]) / (box[1][0] - box[0][0]))
        else:
            angle = np.arctan((box[3][1] - box[0][1]) / (box[3][0] - box[0][0]))
        # print(angle)


        # TODO: check if the box is clockwise or not and trasform as needed
        if angle < 0:
            # print("destra?")
            box = np.array([box[0], box[3], box[2], box[1]])
        else :
            # print("sinistra?")
            box = np.roll(box, -1, axis=0)
            box = np.array([box[0], box[3], box[2], box[1]])
        count += 1

        # affine transformation
        # mat = cv2.getRotationMatrix2D((im2.shape[0]/2,im2.shape[1]/2), -90+alpha, 0.7)#-90+alpha
        # warped_image = cv2.warpAffine(blur_img.copy(), mat, (output_width*2, output_height*2))

        # perspective transformation
        transformation_matrix = cv2.getPerspectiveTransform(box.astype(np.float32), output_corners)
        warped_image = cv2.warpPerspective(blur_img.copy(), transformation_matrix, (output_width, output_height))

        x, y = warped_image.shape[:2]

        # name = warped_image[x//16:x//7
        name = warped_image[:x // 7,:y-y//10]

        loaded_kmeans = joblib.load('kmeans_model.pkl')
        centers = loaded_kmeans.cluster_centers_.astype(np.uint8)
        print(centers)
        segmented = centers[loaded_kmeans.predict(cv2.cvtColor(name, cv2.COLOR_BGR2HSV).reshape(-1, 3))]
        test = cv2.cvtColor(cv2.cvtColor(segmented.reshape(*name.shape), cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)

        # sharpening
        blurredn = cv2.bilateralFilter(name, 9, 5, 5)

        # grayn = cv2.cvtColor(blurredn, cv2.COLOR_BGR2GRAY)
        hsvn = cv2.cvtColor(blurredn, cv2.COLOR_BGR2HSV)

        # thresh_h = cv2.adaptiveThreshold(hsvn[:,:,0]*2, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1)
        # thresh_s = cv2.adaptiveThreshold(hsvn[:,:,1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1)
        # thresh_v = cv2.adaptiveThreshold(hsvn[:,:,2], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 1)

        # cv2.imshow(f"s {count}", thresh_s)
        # s = cv2.bitwise_not(thresh_h+ thresh_s)

        # kernel = np.array([[1, -1, -1], [0, 5, -1], [1, -1,-1]])
        # warped_image = cv2.filter2D(warped_image, -1, kernel)
        # resize
        # warped_image = cv2.resize(warped_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)


        # gold_range = np.array([(130, 94, 48), (243, 219, 173)], dtype=np.uint8).reshape(-1,3)
        # black_range = np.array([(0, 0, 0), (230,230,230)], dtype=np.uint8).reshape(-1,3)
        # viola = (167,111,194)
        # test = cv2.inRange(hsvn,(250,50,50),(220,60,60))
        # newkernel= np.ones((2,2), np.uint8)
        # test = cv2.morphologyEx(test, cv2.MORPH_CLOSE, newkernel, iterations=1)

        cv2.imshow(f"h {count}", test)

        text = pt.image_to_string(test, lang='ita', )
        namelist.append(pmt.get_name(text))

        # cv2.imshow(f"sname of {count}", cv2.resize(s, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC))
        # cv2.imshow(f"name of {count}", cv2.resize(name, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC))

    else:
        # cv2.drawContours(im2, [c], -1, (0,0,255), 8)
        # print("smt wrongobongo")
        failed += 1

# cv2.imshow("black", cv2.resize(black, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
# cv2.imshow("contours", cv2.resize(im2, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
print(f"found {count} cards and {failed} failed")
print(namelist)
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

##
# detect corners with new algorithm
# corners = cv2.goodFeaturesToTrack(gray, 10, qualityLevel=0.05, minDistance= 60, mask=None, blockSize=3, gradientSize= 1, useHarrisDetector= False, k=0.4)
# if corners is None:
#     print("no corners found")
#     exit()

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
