import numpy as np
import cv2


img = cv2.imread('board.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)

key = 0
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)

    cv2.putText(img, str(key), (x+5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
    key = key + 1

cv2.imshow('Corner', img)

while(1):
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break