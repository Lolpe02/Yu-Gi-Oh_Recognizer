import cv2
import pytesseract as pt
from PIL import Image
import numpy as np
# import nltk

pt.pytesseract.tesseract_cmd = r'D:\Program Files\TesseractOCR\tesseract.exe'
print(pt.get_languages())
img = cv2.imread('carta2.jpeg')
# print(np.linspace(-1,11))
img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
# img = cv2.GaussianBlur(img, (9,9),0)s
# img = img[::-1,::-1] # inverted img
# cv2.namedWindow("Img", cv2.WINDOW_KEEPRATIO)

src_pts = []
def onClick(event,x,y,flags,param):
    corners = ["choose top left","choose top right","choose bottom right","choose bottom left", "done"]
    if event == cv2.EVENT_LBUTTONDOWN:

        if len(src_pts) <4:
            src_pts.append([x,y])
            put = cv2.circle(img.copy(), (x,y), 10, (0,0,255), -1)
            put = cv2.putText(put, corners[len(src_pts)], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Img", put)
    else :
        put = cv2.putText(img.copy(), corners[len(src_pts)], (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Img", put)

# points for 0.5 scale [[181, 306], [426, 370], [341, 730], [82, 664]]
# points for 1 scale   [[362, 612], [852, 740],[682, 1460],[164, 1328]]

# homogenous trasform
# cv2.imshow("Img",img )
# cv2.setMouseCallback("Img", onClick)
# cv2.waitKey(0)
original_corners = np.array([[362, 612], [852, 740],[682, 1460],[164, 1328]], dtype=np.float32)
# print(src_pts)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
# sobel = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

# detect corners with new algorithm
# corners = cv2.goodFeaturesToTrack(gray,10,qualityLevel=0.8,minDistance=
# 60, mask=None, blockSize= 6, gradientSize= 19, useHarrisDetector= False, k=0.4)
# if corners is None:
#     print("no corners found")
#     exit()
# # Draw corners on the original image
# for corner in corners:
#     x, y = corner.ravel()
#     cv2.circle(img, (int(x), int(y)), 3, (0, 255, 0), -1)

# Define the dimensions of the output image (width, height)
output_width = 590
output_height = 860

# Define the coordinates of the corresponding corners in the output image
output_corners = np.float32([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]])

# Compute the perspective transformation matrix
transformation_matrix = cv2.getPerspectiveTransform(original_corners, output_corners)

# Apply the perspective transformation
warped_image = cv2.warpPerspective(img, transformation_matrix,(output_width, output_height))


#--- convert the image to HSV color space ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
# find hsv of points
# for point in [[191, 319], [128, 666], [360, 620], [401, 442]]:
#     print(hsv[point[1],point[0]]) #why not [point[0],point[1]]? because opencv is BGR not RGB
# [65, 27, 103],[38, 18, 57],[49, 61, 121],[55, 29, 97]
cv2.imshow('H', hsv[:,:,0])
cv2.imshow('S', hsv[:,:,1])

#--- find Otsu threshold on hue and saturation channel ---
ret, thresh_H = cv2.threshold(hsv[:,:,0], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
ret, thresh_S = cv2.threshold(hsv[:,:,1], 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#--- add the result of the above two ---
cv2.imshow('thresh', thresh_H + thresh_S)

#--- some morphology operation to clear unwanted spots ---
kernel = np.ones((2, 2), np.uint8)
dilation = cv2.dilate(thresh_H + thresh_S, kernel, iterations = 1)
cv2.imshow('dilation', dilation)

#--- find contours on the result above ---
(contours, hierarchy) = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print()
#--- since there were few small contours found, retain those above a certain area ---
im2 = img.copy()
count = 0
for c in contours:
    if cv2.contourArea(c) > 50 and cv2.contourArea(c) < np.multiply(img.shape[0],img.shape[1])-1:
        count+=1
        cv2.drawContours(im2, [c], -1, (0, 255, 0), 2)

cv2.imshow('cards_output', im2)
print('There are {} cards'.format(count))
# cv2.imshow("corners1", img)
cv2.waitKey(0)

# cv2.imshow('img', warped_image)
# cv2.waitKey(0)
# text = pt.image_to_string(warped_image, lang='ita')
# print(text)
# print("tibo can u see this?")
