import cv2
import pytesseract as pt
from PIL import Image
import numpy as np

pt.pytesseract.tesseract_cmd = r'D:\Program Files\TesseractOCR\tesseract.exe'
print(pt.get_languages())
img = cv2.imread('carta2.jpeg')
cv2.resize(img, None, fx=0.3, fy=0.3)
# img = img[::-1,::-1] inverted img
# cv2.namedWindow("Img", cv2.WINDOW_KEEPRATIO)

src_pts = []
def onClick(event,x,y,flags,param):
    corners = ["top left","top right","bottom right","bottom left"]
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(src_pts) <4:
            src_pts.append([x,y])
            a = cv2.circle(img.copy(), (x,y), 10, (0,0,255), -1)
            a = cv2.addText(a, corners[len(src_pts)])
            cv2.imshow("Img", a)

# homogenous trasform
# cv2.imshow("Img",img)
# cv2.setMouseCallback("Img", onClick)
# cv2.waitKey(0)
# print(src_pts)
original_corners = np.array([[381, 630], [840, 759], [668, 1454], [176, 1316]], dtype=np.float32)

# Define the dimensions of the output image (width, height)
output_width = 590
output_height = 860

# Define the coordinates of the corresponding corners in the output image
output_corners = np.float32([[0, 0], [output_width - 1, 0], [output_width - 1, output_height - 1], [0, output_height - 1]])

# Compute the perspective transformation matrix
transformation_matrix = cv2.getPerspectiveTransform(original_corners, output_corners)

# Apply the perspective transformation
warped_image = cv2.warpPerspective(img, transformation_matrix,(output_width, output_height))

cv2.imshow('img', img)
cv2.waitKey(0)
text = pt.image_to_string(img, lang='ita')
print(text)
