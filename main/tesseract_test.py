import cv2
import pytesseract

plate_img = cv2.imread("plate/plate7.jpg")
print(plate_img.shape)
configuration = ("--oem 1 --psm 7") 
plate_text = pytesseract.image_to_string(plate_img, config=configuration)
plate_text = list(plate_text.strip())
original_letter =  ['i', 'I', 'O', 'Q', 'A', 'S', 'Z', 'z', 'o', 'G', 'B']
transform_letter = ['1', '1', '0', '0', '4', '5', '2', '2', '0', '6', '8']
if len(plate_text) > 5:
    for i, letter in enumerate(original_letter) :
        if letter in plate_text[2:-3] :
            plate_text[2:-3] = [item.replace(letter, transform_letter[i]) for item in plate_text[2:-3]]
plate_text = ''.join(plate_text)
print("PREDICTED TEXT ", plate_text)
