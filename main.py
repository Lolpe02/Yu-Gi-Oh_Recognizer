import cv2
import pytesseract as pt
from PIL import Image
import numpy as np
# import nltk

pt.pytesseract.tesseract_cmd = r'D:\Program Files\TesseractOCR\tesseract.exe'
print(pt.get_languages())
img = cv2.imread('carta2.jpeg')
# print(np.linspace(-1,11))
# img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
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
# gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5)
# sobely = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5)
# sobel = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)

# detect corners with new algorithm
# corners = cv2.goodFeaturesToTrack(gray,15,qualityLevel=0.9,minDistance=
# 80, mask=None, blockSize= 10, gradientSize= 1, useHarrisDetector= True, k=0.0004)
# if corners is None:
#     print("no corners found")
#     exit()
# Draw corners on the original image
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



# img[dst > 0.0000004 * dst.max()] =
# cv2.imshow("corners2", img)
# cv2.waitKey(0)

cv2.imshow('img', warped_image)
cv2.waitKey(0)
text = pt.image_to_string(warped_image, lang='ita')
# print(text)
# print("tibo can u see this?")
