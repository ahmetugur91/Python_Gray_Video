from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import math
import copy
import numpy as np
import cv2
from collections import Counter


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


def color_from_rgb(r, g, b):
    if r < 100 and g < 100 and b > 125:
        return "blue"
    if r < 100 and g > 125 and b < 100:
        return "green"
    if r > 120 and g < 100 and b < 100:
        return "red"
    if r > 120 and g > 120 and b > 120:
        return "white"


def player_decision(r, g, b):
    if color_from_rgb(r, g, b) == "green":
        return 1
    if color_from_rgb(r, g, b) == "blue":
        return 2
    if color_from_rgb(r, g, b) == "white":
        return 0
    return -1


showPointDots = False
showCornerPoints = True
showIndexNumbers = False
showCenterDots = True
drawGridLines = True
showBorders = True
showIntersectionPoints = False

# load the image
image = cv2.imread("22.jpg", 1)
# image = cv2.imread("1_dot.jpg", 1)
image = cv2.resize(image, (800, 1000))
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

# cv2.imshow("", thresh)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

intersectionPoints = []
boxes = []

if len(contours) != 0:
    # draw in blue the contours that were founded
    c = max(contours, key=cv2.contourArea)

    # get corner points
    epsilon = 0.1 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)
    # print(approx)

    cornerPoint1 = (approx[0][0][0], approx[0][0][1])
    cornerPoint2 = (approx[1][0][0], approx[1][0][1])
    cornerPoint3 = (approx[3][0][0], approx[3][0][1])
    cornerPoint4 = (approx[2][0][0], approx[2][0][1])

    # persfektif görünüş
    pts1 = np.float32([cornerPoint1, cornerPoint2, cornerPoint3, cornerPoint4])
    pts2 = np.float32([[0, 0], [500, 0], [0, 600], [500, 600]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    perspective = cv2.warpPerspective(image, matrix, (500, 600))

    cv2.imshow("Orginal perspective", perspective)

    cv2.line(image, cornerPoint1, cornerPoint4, 255, 2)
    cv2.line(image, cornerPoint3, cornerPoint2, 255, 2)

    pieceX = abs(roundto((cornerPoint1[0] - cornerPoint4[0]) / 100))
    pieceY = abs(roundto((cornerPoint1[1] - cornerPoint4[1]) / 100))
    print("PieceX ", pieceX)

    newCornerPoint1 = (0, 0)
    prevColor = None
    for iteration in range(0, 101, 1):
        dot = (cornerPoint1[0] + iteration * pieceX, cornerPoint1[1] + iteration * pieceY)
        color = image_original[dot[1], dot[0]]

        curColor = color_from_rgb(color[2], color[1], color[0])

        if prevColor == "red" and curColor == "white":
            newCornerPoint1 = dot
            break

        prevColor = curColor

        print(str(iteration) + "=>" + str(curColor) + "=>" + str(dot))

        cv2.circle(image, dot, 1, (255, 0, 0), -1)
        cv2.putText(image, str(iteration), dot, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    # Corner 2
    pieceX = abs(roundto((cornerPoint2[0] - cornerPoint3[0]) / 100))
    pieceY = abs(roundto((cornerPoint2[1] - cornerPoint3[1]) / 100))
    print("PieceX ", pieceX)

    newCornerPoint2 = (0, 0)
    prevColor = None
    for iteration in range(0, 101, 1):
        dot = (cornerPoint2[0] + iteration * pieceX, cornerPoint2[1] - iteration * pieceY)
        color = image_original[dot[1], dot[0]]

        curColor = color_from_rgb(color[2], color[1], color[0])

        if prevColor == "red" and curColor == "white":
            newCornerPoint2 = dot
            break

        prevColor = curColor

        print(str(iteration) + "=>" + str(curColor) + "=>" + str(dot))

        cv2.circle(image, dot, 1, (255, 0, 0), -1)
        cv2.putText(image, str(iteration), dot, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    pieceX = abs(roundto((cornerPoint3[0] - cornerPoint2[0]) / 100))
    pieceY = abs(roundto((cornerPoint3[1] - cornerPoint2[1]) / 100))
    print("PieceX ", pieceX)

    newCornerPoint3 = (0, 0)
    prevColor = None
    for iteration in range(0, 101, 1):
        dot = (cornerPoint3[0] - iteration * pieceX, cornerPoint3[1] + iteration * pieceY)
        color = image_original[dot[1], dot[0]]

        curColor = color_from_rgb(color[2], color[1], color[0])

        if prevColor == "red" and curColor == "white":
            newCornerPoint3 = dot
            break

        prevColor = curColor

        print(str(iteration) + "=>" + str(curColor) + "=>" + str(dot))

        cv2.circle(image, dot, 1, (255, 0, 0), -1)
        cv2.putText(image, str(iteration), dot, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    # Corner 4
    pieceX = abs(roundto((cornerPoint4[0] - cornerPoint1[0]) / 100))
    pieceY = abs(roundto((cornerPoint4[1] - cornerPoint1[1]) / 100))
    print("PieceX ", pieceX)

    newCornerPoint4 = (0, 0)
    prevColor = None
    for iteration in range(0, 101, 1):
        dot = (cornerPoint4[0] - iteration * pieceX, cornerPoint4[1] - iteration * pieceY)
        color = image_original[dot[1], dot[0]]

        curColor = color_from_rgb(color[2], color[1], color[0])

        if prevColor == "red" and curColor == "white":
            newCornerPoint4 = dot
            break

        prevColor = curColor

        print(str(iteration) + "=>" + str(curColor) + "=>" + str(dot))

        cv2.circle(image, dot, 1, (255, 0, 0), -1)
        cv2.putText(image, str(iteration), dot, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    # print(newCornerPoint1)
    # print(newCornerPoint2)
    # print(newCornerPoint3)
    # print(newCornerPoint4)
    #
    # cv2.imshow("s", image)
    # cv2.waitKey(0)
    # exit()

    cornerPoint1 = newCornerPoint1
    cornerPoint2 = newCornerPoint2
    cornerPoint3 = newCornerPoint3
    cornerPoint4 = newCornerPoint4

    if showCornerPoints:
        cv2.circle(image, cornerPoint1, 5, (0, 0, 255), -1)
        cv2.circle(image, cornerPoint2, 5, (0, 0, 255), -1)
        cv2.circle(image, cornerPoint3, 5, (0, 0, 255), -1)
        cv2.circle(image, cornerPoint4, 5, (0, 0, 255), -1)

    if showBorders:
        cv2.line(image, cornerPoint1, cornerPoint2, 255, 2)
        cv2.line(image, cornerPoint3, cornerPoint4, 255, 2)
        cv2.line(image, cornerPoint1, cornerPoint3, 255, 2)
        cv2.line(image, cornerPoint2, cornerPoint4, 255, 2)

    topLen = int(
        math.sqrt(abs(math.pow(cornerPoint1[0] - cornerPoint3[0], 2) - math.pow(cornerPoint1[1] - cornerPoint3[1], 2))))
    bottomLen = int(
        math.sqrt(abs(math.pow(cornerPoint2[0] - cornerPoint4[0], 2) - math.pow(cornerPoint2[1] - cornerPoint4[1], 2))))
    leftLen = int(
        math.sqrt(abs(math.pow(cornerPoint1[0] - cornerPoint2[0], 2) - math.pow(cornerPoint1[1] - cornerPoint2[1], 2))))
    rightLen = int(
        math.sqrt(abs(math.pow(cornerPoint3[0] - cornerPoint4[0], 2) - math.pow(cornerPoint3[1] - cornerPoint4[1], 2))))

    print("topLen = " + str(topLen) + ", bottomLen = " + str(bottomLen) + ", leftLen = " + str(
        leftLen) + ", rightLen = " + str(rightLen))

    topPieceLenX = roundto((cornerPoint3[0] - cornerPoint1[0]) / 15)
    topPieceLenY = roundto((cornerPoint3[1] - cornerPoint1[1]) / 15)

    bottomPieceLenX = roundto((cornerPoint4[0] - cornerPoint2[0]) / 15)
    bottomPieceLenY = roundto((cornerPoint4[1] - cornerPoint2[1]) / 15)

    leftPieceLenX = roundto((cornerPoint2[0] - cornerPoint1[0]) / 15)
    leftPieceLenY = roundto((cornerPoint2[1] - cornerPoint1[1]) / 15)

    rightPieceLenX = roundto((cornerPoint4[0] - cornerPoint3[0]) / 15)
    rightPieceLenY = roundto((cornerPoint4[1] - cornerPoint3[1]) / 15)

    for yrow in range(0, 16, 1):
        pointDotTop = (cornerPoint1[0] + yrow * topPieceLenX, cornerPoint1[1] + yrow * topPieceLenY)
        intersectionPoints.append(pointDotTop)
        if showIndexNumbers:
            cv2.circle(image, pointDotTop, 1, (255, 0, 0), -1)

    for xrow in range(1, 16, 1):
        # print(i)
        pointDotLeft = (cornerPoint1[0] + xrow * leftPieceLenX, cornerPoint1[1] + xrow * leftPieceLenY)
        pointDotRight = (cornerPoint3[0] + xrow * rightPieceLenX, cornerPoint3[1] + xrow * rightPieceLenY)

        intersectionPoints.append(pointDotLeft)

        if showPointDots:
            cv2.circle(image, pointDotLeft, 1, (255, 0, 0), -1)

        for yrow in range(1, 15, 1):
            pointDotTop = (cornerPoint1[0] + yrow * topPieceLenX, cornerPoint1[1] + yrow * topPieceLenY)
            pointDotBottom = (cornerPoint2[0] + yrow * bottomPieceLenX, cornerPoint2[1] + yrow * bottomPieceLenY)

            # print(pointDotTop, pointDotBottom)
            if drawGridLines:
                cv2.line(image, pointDotTop, pointDotBottom, 255, 1)
                cv2.line(image, pointDotLeft, pointDotRight, 255, 1)

            if showPointDots:
                cv2.circle(image, pointDotTop, 1, (0, 0, 255), -1)
                cv2.circle(image, pointDotBottom, 1, (0, 0, 255), -1)
                cv2.circle(image, pointDotLeft, 1, (0, 0, 255), -1)
                cv2.circle(image, pointDotRight, 1, (0, 0, 255), -1)

            x, y = line_intersection((pointDotTop, pointDotBottom), (pointDotLeft, pointDotRight))
            intersectionPoints.append((roundto(x), roundto(y)))
            if showIntersectionPoints:
                cv2.circle(image, (roundto(x), roundto(y)), 1, (255, 0, 0), -1)
            # print(x, y)

        intersectionPoints.append(pointDotRight)
        if showPointDots:
            cv2.circle(image, pointDotRight, 1, (255, 0, 0), -1)

if showIndexNumbers:
    counter = 0
    for point in intersectionPoints:
        # print(point)
        cv2.putText(image, str(counter), point, cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        counter = counter + 1

matrix = []
matrixRow = []

for row in range(0, 15, 1):
    matrixRow = []
    for col in range(0, 15, 1):

        centerColors = []

        current = col + row * 16
        cornerPoint1 = intersectionPoints[current]
        cornerPoint2 = intersectionPoints[current + 17]
        # cv2.line(image, point1, point2, 200, 1)
        middleX = int((cornerPoint1[0] + cornerPoint2[0]) / 2)
        middleY = int((cornerPoint1[1] + cornerPoint2[1]) / 2)
        if showCenterDots:
            cv2.circle(image, (middleX, middleY), 1, (255, 0, 0), -1)
        # print(image_original[middleY, middleX])

        rgb = image_original[middleY, middleX]
        centerColors.append((rgb[2], rgb[1], rgb[0]))

        rgb = image_original[middleY + 1, middleX + 1]
        cv2.circle(image, (middleX + 1, middleY + 1), 1, (255, 0, 0), -1)
        centerColors.append((rgb[2], rgb[1], rgb[0]))

        rgb = image_original[middleY - 1, middleX - 1]
        cv2.circle(image, (middleX - 1, middleY - 1), 1, (255, 0, 0), -1)
        centerColors.append((rgb[2], rgb[1], rgb[0]))

        rgb = image_original[middleY, middleX + 1]
        cv2.circle(image, (middleX, middleY + 1), 1, (255, 0, 0), -1)
        centerColors.append((rgb[2], rgb[1], rgb[0]))

        rgb = image_original[middleY, middleX - 1]
        cv2.circle(image, (middleX, middleY - 1), 1, (255, 0, 0), -1)
        centerColors.append((rgb[2], rgb[1], rgb[0]))

        rgb = image_original[middleY + 1, middleX]
        cv2.circle(image, (middleX + 1, middleY), 1, (255, 0, 0), -1)
        centerColors.append((rgb[2], rgb[1], rgb[0]))

        rgb = image_original[middleY - 1, middleX]
        cv2.circle(image, (middleX - 1, middleY), 1, (255, 0, 0), -1)
        centerColors.append((rgb[2], rgb[1], rgb[0]))

        matrixRow.append(centerColors)

    matrix.append(matrixRow)

# print(matrix)
for row in matrix:
    print(row)

print(matrix[0][1])
output = ""
row_dump = []
for row in matrix:
    dump = []
    for col in row:
        player_colors = []
        for clr in col:
            player_colors.append(player_decision(*clr))

        b = Counter(player_colors)
        most = b.most_common(1)[0][0]
        dump.append(str(most))
    row_dump.append("\t".join(dump))

output = "\n".join(row_dump)

print(output)

f = open("data.txt", "w")
f.write(output)
f.close()

cv2.imshow('Original Image', image_original)
cv2.imshow('Processed', image)
# cv2.imshow("Cropped", output)

cv2.waitKey(0)
