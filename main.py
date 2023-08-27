import cv2
import pytesseract as pt
# from PIL import Image
import numpy as np
# import Perfect_Rotation as pr
# import nltk

def is_rectangle(contour):
    perimeter = cv2.arcLength(contour, True)
    vertices = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    return len(vertices) == 4

# Define the dimensions of the output image (width, height)
output_width = 590 #// 2
output_height = 860 #// 2

pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# print(pt.get_languages())
img = cv2.imread('carta3.jpeg')
# img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
# bilateralFilter keeps edges sharp while blurring the rest
blur = cv2.bilateralFilter(img, 9, 75, 75)
edges = cv2.Canny(blur, 100, 300)
cv2.imshow("edges", edges)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# top_right, top_left, bottom_left, bottom_right
output_corners = np.float32([[output_width - 1, 0], [0, 0], [0, output_height - 1], [output_width - 1, output_height - 1]])
count = 0
for c in contours:
    if cv2.contourArea(c) > 300 and is_rectangle(c):
        im2 = img.copy()
        cv2.drawContours(im2, [c], -1, (0, 255, 0), 2)
        vertices = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)
        vertices = np.array(vertices, np.float32).reshape(4, 2)
        for a, p in enumerate(vertices):
            cv2.circle(im2, p.astype(int), 5,(a *63,a * 63,a * 63), -1)
        #TODO: check if the box is clockwise or not and trsform as needed
        # if box[1,0] - box[0,0] > box[2,1] - box[1,1]:
        #      box = np.concatenate(box[-1] + box[:-1])

        count += 1
        transformation_matrix = cv2.getPerspectiveTransform(vertices, output_corners)
        warped_img = cv2.warpPerspective(im2, transformation_matrix, (output_width, output_height))
        text = pt.image_to_string(warped_img, lang='ita', config=' --psm 4')
        print(text)
        cv2.imshow(f"contour {count}", warped_img)
cv2.waitKey(0)


# cv2.imshow("contours", im2)
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
#
# # cv2.imshow('img', warped_image)
# # cv2.waitKey(0)
# # text = pt.image_to_string(warped_image, lang='ita')
# # print(text)
# # print("tibo can u see this?")
