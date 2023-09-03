import cv2
import pytesseract as pt
import numpy as np
import Param_Tuning as pmt

pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
namelist = []
kernel = np.ones((2,2), np.uint8)
img = cv2.imread('cards/carta1.jpeg')

blur_img = cv2.bilateralFilter(img, 9, 10, 10)
processed = pmt.hsv_thresh(blur_img, kernel)[0]

canny = cv2.Canny(processed, 60, 300)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
new = cv2.morphologyEx(canny, cv2.MORPH_CROSS, kernel, iterations=3)
contours, hierarchy = cv2.findContours(new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours is None:
    print("no contours found")
    exit()

im2 = img.copy()
count = 0
failed = 0
# Dimensions of the output image
output_width, output_height = 590, 860
# top_left, top_right, bottom_right, bottom_left
output_corners = np.float32([[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]])

for c in contours:
    if cv2.contourArea(c) > 2000 and pmt.is_rectangle(c):
        cv2.drawContours(im2, [c], -1, (0, 255, 0), 2)

        # APPROX METHOD
        box = cv2.approxPolyDP(c, 0.06 * cv2.arcLength(c, True), True)
        if len(box) != 4:
            continue
        box = box.reshape(4, 2)
        x, y, w, h = cv2.boundingRect(c)
        box = pmt.order(w, h, box)

        # pmt.print_points(box, im2)

        # Find the longest side
        if cv2.norm(box[0], box[1], normType=cv2.NORM_L2) > cv2.norm(box[0], box[3], normType=cv2.NORM_L2):
            angle = np.arctan((box[1][1] - box[0][1]) / (box[1][0] - box[0][0]))
        else:
            angle = np.arctan((box[3][1] - box[0][1]) / (box[3][0] - box[0][0]))

        if angle < 0:
            box = np.array([box[0], box[3], box[2], box[1]])
        else:
            box = np.roll(box, -1, axis=0)
            box = np.array([box[0], box[3], box[2], box[1]])
        count += 1

        # perspective transformation
        transformation_matrix = cv2.getPerspectiveTransform(box.astype(np.float32), output_corners)
        warped_image = cv2.warpPerspective(img, transformation_matrix, (output_width, output_height))

        x, y = warped_image.shape[:2]

        name = warped_image[:x // 7, :y - y // 10]
        gray = cv2.cvtColor(name, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        text = pt.image_to_string(thresh, lang='ita', config=' --psm 1')
        namelist.append(pmt.get_name(text))
    else:
        failed += 1

print(f"found {count} cards and {failed} failed")
print(namelist)
cv2.waitKey(0)
