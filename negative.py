import numpy as np
import cv2

# load the image
image = cv2.imread("4.jpg", 1)
image = cv2.resize(image, (500, 800))
# red color boundaries (R,B and G)
lower = [120, 120, 120]
upper = [255, 255, 255]

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

# find the colors within the specified boundaries and apply
# the mask
mask = cv2.inRange(image, lower, upper)
output = cv2.bitwise_and(image, image, mask=mask)

# cv2.imshow("mask",output)

ret, thresh = cv2.threshold(mask, 127, 255, 0)
thresh = cv2.medianBlur(thresh, 11)

# cv2.imshow("",thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

if len(contours) != 0:
    # draw in blue the contours that were founded
    c = max(contours, key=cv2.contourArea)
    cv2.drawContours(image, c, -1, 255, 3)
    cv2.drawContours(output, c, -1, 255, 3)

    # find the biggest area


    # x, y, w, h = cv2.boundingRect(c)
    # draw the book contour (in green)
    # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the images
# cv2.imshow("Result", np.hstack([image, output]))
cv2.imshow("Original",image)
cv2.imshow("Cropped",output)

cv2.waitKey(0)
