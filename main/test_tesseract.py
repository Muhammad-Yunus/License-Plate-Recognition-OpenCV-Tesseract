import pytesseract
import cv2

img = cv2.imread("image_text.jpg")
text = pytesseract.image_to_string(img)

print(text)