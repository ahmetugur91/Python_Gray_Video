import math
import copy
import numpy as np
import cv2


def roundto(number):
    floatPart = number - int(number)
    if floatPart >= 0.5:
        return int(number) + 1
    return int(number)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


showCornerPoints = False
showIndexNumbers = False
showCenterDots = True
drawLines = False
drawBorders = True

# load the image
image = cv2.imread("1.jpg", 1)
image = cv2.resize(image, (500, 800))
image_original = copy.copy(image)
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

intersectionPoints = []
boxes = []

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

    if showCornerPoints:
        cv2.circle(image, point1, 5, (0, 0, 255), -1)
        cv2.circle(image, point2, 5, (0, 0, 255), -1)
        cv2.circle(image, point3, 5, (0, 0, 255), -1)
        cv2.circle(image, point4, 5, (0, 0, 255), -1)

    # cv2.line(image, point1, point2, 255, 5)
    # cv2.line(image, point3, point4, 255, 5)
    #
    # cv2.line(image, point1, point3, 255, 1)
    # cv2.line(image, point2, point4, 255, 1)

    topLen = int(math.sqrt(abs(math.pow(point1[0] - point3[0], 2) - math.pow(point1[1] - point3[1], 2))))
    topPieceLen = int(topLen / 15)

    bottomLen = int(math.sqrt(abs(math.pow(point2[0] - point4[0], 2) - math.pow(point2[1] - point4[1], 2))))
    bottomPieceLen = int(bottomLen / 15)

    leftLen = int(math.sqrt(abs(math.pow(point1[0] - point2[0], 2) - math.pow(point1[1] - point2[1], 2))))
    leftPieceLen = int(leftLen / 15)

    rightLen = int(math.sqrt(abs(math.pow(point3[0] - point4[0], 2) - math.pow(point3[1] - point4[1], 2))))
    rightPieceLen = int(rightLen / 15)

    print("topLen = " + str(topLen) + ", bottomLen = " + str(bottomLen) + ", leftLen = " + str(leftLen) + ", rightLen = " + str(rightLen))
    print("topPieceLen = " + str(topPieceLen) + ", bottomPieceLen = " + str(bottomPieceLen) + ", leftPieceLen = " + str(leftPieceLen) + ", rightPieceLen = " + str(rightPieceLen))

    topPieceLenX = roundto((point3[0] - point1[0]) / 15)
    topPieceLenY = roundto((point3[1] - point1[1]) / 15)

    bottomPieceLenX = roundto((point4[0] - point2[0]) / 15)
    bottomPieceLenY = roundto((point4[1] - point2[1]) / 15)

    leftPieceLenX = roundto((point2[0] - point1[0]) / 15)
    leftPieceLenY = roundto((point2[1] - point1[1]) / 15)

    rightPieceLenX = roundto((point4[0] - point3[0]) / 15)
    rightPieceLenY = roundto((point4[1] - point3[1]) / 15)

    for yrow in range(0, 16, 1):
        pointDotTop = (point1[0] + yrow * topPieceLenX, point1[1] + yrow * topPieceLenY)
        intersectionPoints.append(pointDotTop)
        cv2.circle(image, pointDotTop, 1, (255, 0, 0), -1)

    for xrow in range(1, 16, 1):
        # print(i)
        pointDotLeft = (point1[0] + xrow * leftPieceLenX, point1[1] + xrow * leftPieceLenY)
        pointDotRight = (point3[0] + xrow * rightPieceLenX, point3[1] + xrow * rightPieceLenY)

        intersectionPoints.append(pointDotLeft)
        cv2.circle(image, pointDotLeft, 1, (255, 0, 0), -1)

        for yrow in range(1, 15, 1):
            pointDotTop = (point1[0] + yrow * topPieceLenX, point1[1] + yrow * topPieceLenY)
            pointDotBottom = (point2[0] + yrow * bottomPieceLenX, point2[1] + yrow * bottomPieceLenY)

            # print(pointDotTop, pointDotBottom)
            cv2.line(image, pointDotTop, pointDotBottom, 255, 1)
            cv2.line(image, pointDotLeft, pointDotRight, 255, 1)

            cv2.circle(image, pointDotTop, 1, (0, 0, 255), -1)
            cv2.circle(image, pointDotBottom, 1, (0, 0, 255), -1)
            cv2.circle(image, pointDotLeft, 1, (0, 0, 255), -1)
            cv2.circle(image, pointDotRight, 1, (0, 0, 255), -1)

            x, y = line_intersection((pointDotTop, pointDotBottom), (pointDotLeft, pointDotRight))
            cv2.circle(image, (roundto(x), roundto(y)), 1, (255, 0, 0), -1)
            intersectionPoints.append((roundto(x), roundto(y)))
            # print(x, y)

        intersectionPoints.append(pointDotRight)
        cv2.circle(image, pointDotRight, 1, (255, 0, 0), -1)

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

# for index numbers on image
counter = 0
for point in intersectionPoints:
    print(point)

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.3
    fontColor = (0, 255, 255)
    lineType = 1

    cv2.putText(image, str(counter),
                point,
                font,
                fontScale,
                fontColor,
                lineType)

    counter = counter + 1

for row in range(0, 15, 1):
    for col in range(0, 15, 1):
        current = col + row * 16
        point1 = intersectionPoints[current]
        point2 = intersectionPoints[current + 17]
        # cv2.line(image, point1, point2, 200, 1)
        middleX = int((point1[0] + point2[0]) / 2)
        middleY = int((point1[1] + point2[1]) / 2)
        cv2.circle(image, (middleX, middleY), 1, (255, 0, 0), -1)

# cropped = image_original[100:200, 140:440]
# cv2.imshow("cropped", cropped)


cv2.imshow('Processed', image)
# cv2.imshow("Cropped", output)

cv2.waitKey(0)
