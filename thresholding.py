import cv2

image = cv2.imread("4.jpg")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = cv2.resize(image, (500, 800))

# convert the image to grayscale, blur it, and find edges
# in the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

# loop over the contours
screenCnt = None
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # if our approximated contour has four points, then we
    # can assume that we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is not None:
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)

print(screenCnt)
cv2.imshow("", image)
cv2.waitKey(0)
