import cv2
import pytesseract as pt
import numpy as np
import Param_Tuning as pmt
import joblib
import os

pt.pytesseract.tesseract_cmd = r'D:\Program Files\TesseractOCR\tesseract.exe'
namelist = []
kernel = np.ones((2,2), np.uint8)
img = cv2.imread('cards/carta12.jpeg')
# vert = -1.56
blur_img = cv2.bilateralFilter(img, 9, 10, 10)
processed = pmt.hsv_thresh(blur_img, kernel)[0]

# def mainloop(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder, filename))
#         if img is not None:
#             continue

cv2.imshow("processed", cv2.resize(processed, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))

canny = cv2.Canny(processed, 60, 300)
cv2.imshow("canny", cv2.resize(canny, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
new= cv2.morphologyEx(canny, cv2.MORPH_CROSS, kernel, iterations=3)

# cv2.imshow("canny", cv2.resize(new, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
# k = cv2.waitKey(0) & 0xFF
# if k == ord("q"):
#     exit()

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
output_corners = np.float32([[0, 0], [output_width-1, 0], [output_width-1, output_height-1], [0, output_height-1]])

# def mainloop(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img = cv2.imread(os.path.join(folder, filename))
#         if img is not None:
#             continue


for c in contours:
    if cv2.contourArea(c) > 4000 and pmt.is_rectangle(c):
        cv2.drawContours(im2, [c], -1, (0, 255, 0), 2)
        # always clockwise starting from the top left tl, tr, br, bl

        # APPROX METHOD
        box = cv2.approxPolyDP(c, 0.06 * cv2.arcLength(c, True), True)
        if len(box) != 4:
            continue
        box = box.reshape(4, 2)
        pmt.print_points(box, im2)
        # Find the longest side
        if cv2.norm(box[0], box[1], normType=cv2.NORM_L2) > cv2.norm(box[0], box[3], normType=cv2.NORM_L2):
            angle = np.arctan((box[1][1] - box[0][1]) / (box[1][0] - box[0][0]))
        else:
            angle = np.arctan((box[3][1] - box[0][1]) / (box[3][0] - box[0][0]))
        x, y, w, h = cv2.boundingRect(c)
        box = pmt.order(w, h, box, angle)

        count += 1

        # perspective transformation
        transformation_matrix = cv2.getPerspectiveTransform(box.astype(np.float32), output_corners)
        # print(transformation_matrix)
        warped_image = cv2.warpPerspective(blur_img.copy(), transformation_matrix, (output_width, output_height))

        x, y = warped_image.shape[:2]
        name = warped_image[:x // 7,y//20:y-y//5]

        loaded_kmeans = joblib.load('kmeans_model.pkl')
        mod = 1
        # centers = loaded_kmeans.cluster_centers_.astype(np.uint8)
        c_list = [(0, 0, 0), (58, 60, 74), (230, 227, 226), (27, 156, 139), (175, 53, 130), (138, 71, 188), (91, 121, 174),
             (186, 149, 84), (171, 104, 75)] #sorted(, key= lambda x : sum(x)/3)
        if mod : # HSV
            a = cv2.cvtColor(np.asarray(c_list, dtype=np.uint8).reshape(-1, 1, 3), cv2.COLOR_BGR2HSV)
            predicted = loaded_kmeans[mod].predict(cv2.cvtColor(name, cv2.COLOR_BGR2HSV).reshape(-1, 3))
            segmented = a[predicted]
            test = cv2.cvtColor(segmented.reshape(*name.shape), cv2.COLOR_HSV2BGR)  #
        elif not mod: # RGB
            a = np.asarray(c_list, dtype=np.uint8).reshape(-1, 1, 3)
            predicted = loaded_kmeans[mod].predict(name.reshape(-1, 3))
            segmented = a[predicted]
            test = segmented.reshape(*name.shape)

        # grayn = cv2.resize(cv2.cvtColor(blurredn, cv2.COLOR_BGR2GRAY), None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)
        # hsvn = cv2.cvtColor(name, cv2.COLOR_BGR2HSV)
        # _, test = cv2.threshold(hsvn[:,:,2], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2.imshow(f"v of {count}", cv2.resize(hsvn[:,:,2], None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC))

        text = pt.image_to_string(test, lang='ita', )
        namelist.append(pmt.get_name(text))

        cv2.imshow(f"sname of {count}", cv2.resize(test, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC))
    else:
        failed += 1
cv2.imshow("contours", cv2.resize(im2, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC))
print(f"found {count} cards and {failed} failed")
#filter out empty strings
namelist = list(filter(None, namelist))
print(namelist)
cv2.waitKey(0)
