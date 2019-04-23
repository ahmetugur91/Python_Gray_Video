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


showPointDots = False
showCornerPoints = True
showIndexNumbers = False
showCenterDots = True
drawGridLines = True
showBorders = True
showIntersectionPoints = False

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    image = frame.copy()
    image = cv2.resize(image, (800, 1000))

    # load the image
    # image = cv2.imread("4.jpg", 1)
    # image = cv2.imread("1_dot.jpg", 1)
    # image = cv2.resize(image, (800, 1000))
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
        try:
            # draw in blue the contours that were founded
            c = max(contours, key=cv2.contourArea)

            # get corner points
            epsilon = 0.1 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)


            if len(approx) != 4:
                continue



            cornerPoint1 = (approx[0][0][0], approx[0][0][1])
            cornerPoint2 = (approx[1][0][0], approx[1][0][1])
            cornerPoint3 = (approx[3][0][0], approx[3][0][1])
            cornerPoint4 = (approx[2][0][0], approx[2][0][1])

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

            topLen = int(math.sqrt(abs(math.pow(cornerPoint1[0] - cornerPoint3[0], 2) - math.pow(cornerPoint1[1] - cornerPoint3[1], 2))))
            bottomLen = int(math.sqrt(abs(math.pow(cornerPoint2[0] - cornerPoint4[0], 2) - math.pow(cornerPoint2[1] - cornerPoint4[1], 2))))
            leftLen = int(math.sqrt(abs(math.pow(cornerPoint1[0] - cornerPoint2[0], 2) - math.pow(cornerPoint1[1] - cornerPoint2[1], 2))))
            rightLen = int(math.sqrt(abs(math.pow(cornerPoint3[0] - cornerPoint4[0], 2) - math.pow(cornerPoint3[1] - cornerPoint4[1], 2))))

            print("topLen = " + str(topLen) + ", bottomLen = " + str(bottomLen) + ", leftLen = " + str(leftLen) + ", rightLen = " + str(rightLen))

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
                    matrixRow.append((rgb[0], rgb[1], rgb[2]))
                matrix.append(matrixRow)

            # print(matrix)
            for row in matrix:
                print(row)
            # cropped = image_original[100:200, 140:440]
            # cv2.imshow("cropped", cropped)

            # cv2.imshow('Original Image', image_original)
            cv2.imshow('Processed', image)
            # cv2.imshow("Cropped", output)

            # cv2.waitKey(0)
        except IndexError:
            print(IndexError)

    cv2.imshow('frame', image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
