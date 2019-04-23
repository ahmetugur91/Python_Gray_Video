import math
import numpy as np
import cv2


def roundto(number):
    floatPart = number - int(number)
    if floatPart >= 0.5:
        return int(number) + 1
    return int(number)


# load the image
image = cv2.imread("3.jpg", 1)
image = cv2.resize(image, (500, 800))
# white color boundaries (R,B and G)
lower = [120, 120, 120]
upper = [255, 255, 255]

# create NumPy arrays from the boundaries
lower = np.array(lower, dtype="uint8")
upper = np.array(upper, dtype="uint8")

# find the colors within the specified boundaries and apply the mask
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

    epsilon = 0.1 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    # print(approx)

    point1 = (approx[0][0][0], approx[0][0][1])
    point2 = (approx[1][0][0], approx[1][0][1])
    point3 = (approx[3][0][0], approx[3][0][1])
    point4 = (approx[2][0][0], approx[2][0][1])

    cv2.circle(image, point1, 5, (0, 0, 255), -1)
    cv2.circle(image, point2, 5, (0, 0, 255), -1)
    cv2.circle(image, point3, 5, (0, 0, 255), -1)
    cv2.circle(image, point4, 5, (0, 0, 255), -1)

    # cv2.line(image, point1, point2, 255, 5)
    # cv2.line(image, point3, point4, 255, 5)
    #
    # cv2.line(image, point1, point3, 255, 1)
    # cv2.line(image, point2, point4, 255, 1)

    topLen = int(math.sqrt(math.pow(point1[0] - point3[0], 2) - math.pow(point1[1] - point3[1], 2)))
    topPieceLen = int(topLen / 15)

    bottomLen = int(math.sqrt(math.pow(point2[0] - point4[0], 2) - math.pow(point2[1] - point4[1], 2)))
    bottomPieceLen = int(bottomLen / 15)

    print("topLen = " + str(topLen) + ", bottomLen = " + str(bottomLen))
    print("topPieceLen = " + str(topPieceLen) + ", bottomPieceLen = " + str(bottomPieceLen))

    a = roundto((point3[0] - point1[0]) / 15)
    b = roundto((point3[1] - point1[1]) / 15)

    q = roundto((point4[0] - point2[0]) / 15)
    w = roundto((point4[1] - point2[1]) / 15)

    for i in range(1, 15, 1):
        # print(i)
        pointDotTop = (point1[0] + i * a, point1[1] + i * b)
        pointDotBottom = (point2[0] + i * q, point2[1] + i * w)

        print(pointDotTop, pointDotBottom)
        cv2.line(image, pointDotTop, pointDotBottom, 255, 1)

        cv2.circle(image, pointDotTop, 1, (0, 0, 255), -1)
        cv2.circle(image, pointDotBottom, 1, (0, 0, 255), -1)



    # middleTopX = int((point1[0] + point3[0]) / 2)
    # middleTopY = int((point1[1] + point3[1]) / 2)
    # middleTopPoint = (middleTopX, middleTopY)
    #
    # middleBottomX = int((point2[0] + point4[0]) / 2)
    # middleBottomY = int((point2[1] + point4[1]) / 2)
    # middleBottomPoint = (middleBottomX, middleBottomY)

    # cv2.line(output, middleTopPoint, middleBottomPoint, 255, 2)
    # print(middleTopPoint,middleBottomPoint)

    # cv2.drawContours(image, c, -1, 255, 3)
    # cv2.drawContours(output, c, -1, 255, 3)

    # find the biggest area

    # x, y, w, h = cv2.boundingRect(c)
    # draw the book contour (in green)
    # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

# show the images
# cv2.imshow("Result", np.hstack([image, output]))
cv2.imshow("Original", image)
# cv2.imshow("Cropped", output)

cv2.waitKey(0)
