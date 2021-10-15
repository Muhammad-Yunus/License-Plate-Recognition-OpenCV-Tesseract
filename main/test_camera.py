import cv2

cap = cv2.VideoCapture(0)

ret, img = cap.read()

if ret :
    cv2.imwrite("image_test.jpg", img)
    print("image saved succesfully!")
else :
    print("can't capture image")