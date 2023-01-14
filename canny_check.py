import cv2

para=[300,400]
def nothing(*arg):
    pass
cv2.namedWindow('Trackbar')
cv2.resizeWindow('Trackbar', 400, 100)
cv2.createTrackbar('low', 'Trackbar', para[0], 1000, nothing)
cv2.createTrackbar('high', 'Trackbar', para[1], 1000, nothing)

source=cv2.imread("test.jpg")
img=source.copy()
cv2.namedWindow("canny", 0)
while(1):
    img = source.copy()
    low = cv2.getTrackbarPos('low', 'Trackbar')
    high = cv2.getTrackbarPos('high', 'Trackbar')
    img = cv2.Canny(img, low, high)
    cv2.imshow("canny", img)
    cv2.waitKey(20)