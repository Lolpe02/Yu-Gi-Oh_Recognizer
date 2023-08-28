import cv2
import pytesseract as pt
# from PIL import Image
import numpy as np
# import Perfect_Rotation as pr
# import nltk

# Define the dimensions of the output image (width, height)
output_width = 590 #// 2
output_height = 860 #// 2

pt.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# print(pt.get_languages())
for i in range(12):
    img = cv2.imread(f'carta{i + 1}.jpeg')
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 150)
    cnts, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img2 = img.copy()
    filtered_cnts = []
    for cnt, h in zip(cnts, hierarchy[0]):
        if h[-1] == -1:
            filtered_cnts.append(cnt)
    filtered_cnts.sort(key=lambda x: cv2.boundingRect(x)[2] * cv2.boundingRect(x)[3], reverse=True)
    x, y, max_w, max_h = cv2.boundingRect(filtered_cnts[0])
    vertices_all = []
    bounding_rects = []
    for cnt in filtered_cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > max_w * max_h - 5000:
            if cv2.contourArea(cnt) > 500:
                vertices = np.array(cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True))
                vertices = vertices.reshape(vertices.shape[0], 2)
            else:
                vertices = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
            vertices_all.append(vertices)
            bounding_rects.append((x, y, w, h))
    for x, y, w, h in bounding_rects:
        cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
    for vertices in vertices_all:
        for a, corner in enumerate(vertices):
            cv2.circle(img2, corner, 5, (a * 63, a * 63, a * 63), -1)
    cv2.imshow('Detected Cards', img2)
    cv2.waitKey(0)