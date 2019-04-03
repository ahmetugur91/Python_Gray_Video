import cv2
import numpy as np

image = cv2.imread("board.jpg")


image = cv2.resize(image, (500, 800))


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

retval, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

threshold = cv2.medianBlur(threshold, 11)

contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
c = max(contours, key=cv2.contourArea)

for c in contours:
    cv2.drawContours(image, [c], 0, (0, 0, 255), 2)

cv2.imshow("Result", image)

cv2.waitKey(0)
