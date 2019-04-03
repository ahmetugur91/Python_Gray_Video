import numpy as np
import cv2

image_src = cv2.imread("board.jpg")
gray = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(gray, 250, 255, 0)

image, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
largest_area = sorted(contours, key=cv2.contourArea)[-1]
mask = np.zeros(image_src.shape, np.uint8)
cv2.drawContours(mask, [largest_area], 0, (255, 255, 255, 255), -1)
dst = cv2.bitwise_and(image_src, mask)
mask = 255 - mask
roi = cv2.add(dst, mask)

roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(roi_gray, 250, 255, 0)
image, contours, hierarchy = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

max_x = 0
max_y = 0
min_x = image_src.shape[1]
min_y = image_src.shape[0]

for c in contours:
    if 150 < cv2.contourArea(c) < 100000:
        x, y, w, h = cv2.boundingRect(c)
        min_x = min(x, min_x)
        min_y = min(y, min_y)
        max_x = max(x + w, max_x)
        max_y = max(y + h, max_y)

roi = roi[min_y:max_y, min_x:max_x]
cv2.imshow("a", roi)

while (1):
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
