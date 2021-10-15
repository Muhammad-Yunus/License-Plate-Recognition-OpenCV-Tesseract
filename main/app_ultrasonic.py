import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import os
import requests
import time 

#Libraries
import RPi.GPIO as GPIO
 
#GPIO Mode (BOARD / BCM)
GPIO.setmode(GPIO.BCM)
 
#set GPIO Pins
GPIO_TRIGGER = 17
GPIO_ECHO = 18
servoPIN = 27
 
#set GPIO direction (IN / OUT)
GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
GPIO.setup(GPIO_ECHO, GPIO.IN)
GPIO.setup(servoPIN, GPIO.OUT)
p = GPIO.PWM(servoPIN, 50) # GPIO 17 for PWM with 50Hz
p.start(5) # Initialization DutyCycle

def ServoGate():
    print("Gate Open")
    p.ChangeDutyCycle(5)
    time.sleep(0.5)
    print("Gate Close")
    p.ChangeDutyCycle(10)
    time.sleep(0.5)
    p.ChangeDutyCycle(5)
    time.sleep(0.5)

def distance():

    maxTime = 0.4 #ms

    GPIO.output(GPIO_TRIGGER,False)
    time.sleep(0.01)

    # set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)
 
    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
 
    StartTime = time.time()
    
    # save StartTime
    timeout = StartTime + maxTime
    while GPIO.input(GPIO_ECHO) == 0 and StartTime < timeout:
        StartTime = time.time()

    StopTime = time.time()

    # save time of arrival
    timeout = StopTime + maxTime
    while GPIO.input(GPIO_ECHO) == 1 and StopTime < timeout:
        StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34300) / 2
 
    return distance

def preprocessing(img, h1, h2, w1, w2):
    # get roi & resize 
    y1, y2, x1, x2 = int(h1), int(h2), int(w1), int(w2)
    roi = img[y1:y2, x1:x2]
    scale = 300/roi.shape[0] # roi.shape[0] merupakan h image roi
    roi = cv2.resize(roi, (0,0), fx=scale, fy=scale)
    
    #convert to gray -> binary
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    __, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh, roi, [x1, y1, x2, y2], scale

# calculate contour & filter contour

def get_contours(thresh):
    contours, ___ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    h, w, c = img.shape
    size = h*w
    contours = [cc for i, cc in enumerate(contours) if contour_char_OK(cc, size)]
    return contours

def contour_char_OK(cc, size=1000000):
    x, y, w, h = cv2.boundingRect(cc)
    area = cv2.contourArea(cc)
    
    if w < 3 or h < 5 or area < 80: 
        return False
    
    validDimentson = w/h > 0.11 and w/h < 0.7 # filter rasio ukuran lebar / tinggi
    validAreaRatio = area/(w*h)  > 0.1 # filter rasio luasan (luas area putih / hitam)
    return validDimentson and validAreaRatio

def sort_contours(contours, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(cnt) for cnt in contours]
    
    cnts, boundingBoxes = zip(*sorted(zip(contours, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
    return cnts, boundingBoxes

# draw label box pada original image
def drawPred(frame, label, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 255), 2)

    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - round(1.5*labelSize[1])), 
                         (max(right, left + labelSize[0]), top + baseLine), (255, 0, 255), -1)
    
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    return frame


# geometric tranformation & crop plat nomor
def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return np.array(rect)

def transform(img, pts, padding=0):
    pad = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * padding
    #pad[2:, -1] = 12
    #pad[1:-1, 0] = 12
    rect = np.float32(order_points(pts) + pad)
    (tl, tr, br, bl) = rect

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth, 0],
        [maxWidth, maxHeight],
        [0, maxHeight]], dtype = "float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return np.int0(warped), rect

# filter box yang tidak memiliki kedekatan jarak dan ukuran
def filter_boxes(boxes):
    boxes = np.array(boxes)
    boxes = nearest_box_position(boxes, label='x', n=3) #nearest box for closest `x` 
    boxes = nearest_box_position(boxes, label='y', n=2) #nearest box for closest `y`
    boxes = nearest_box_size(boxes, label='w', n=2) #nearest box for closest `w` 
    boxes = nearest_box_size(boxes, label='h', n=3) #nearest box for closest `h`
    return boxes

def nearest_box_position(boxes, label='x', n=2):
    pt = 0 if label == 'x' else 1
    mean = np.mean(boxes[:, 0, pt])
    std = np.std(boxes[:, 0, pt])
    boxes = np.array([box for box in boxes if abs(box[0, pt] - mean) < n*std])
    return boxes

def nearest_box_size(boxes, label='w', n=2):
    ptx = (1,0) if label == 'w' else (3,0)
    pt = 0 if label == 'w' else 1
    mean = np.mean(boxes[:, ptx[0], pt] - boxes[:, ptx[1], pt])
    std = np.std(boxes[:, ptx[0], pt] - boxes[:, ptx[1], pt])
    boxes = np.array([box for box in boxes if abs((box[ptx[0], pt] - box[ptx[1], pt]) - mean) < n*std])
    return boxes

# menjadi 4 titik koordinat plat nomor
def get_plate_4_coord(contours):
    contours, rects = sort_contours(contours)
    boxes = []
    for cnt in contours:
        boxes.append(np.int0(cv2.boxPoints(cv2.minAreaRect(cnt))))
    
    # sort box tl, tr, br, bl
    boxes = [order_points(box) for box in boxes]
    
    # filter box
    boxes = filter_boxes(boxes)
    max_x_id = np.argmax(boxes[:,1,0])
    min_x_id = np.argmin(boxes[:,0,0])
    
    l, r = boxes[min_x_id], boxes[max_x_id]
    bl, tl = l[3], l[0]
    br, tr = r[2], r[1]
    
    # adjusment bl(y) & br(y) agar lebih turun
    bl[1] = bl[1] + 10
    br[1] = br[1] + 10

    # adjusment tl(y) & tr(y) agar lebih naik
    tl[1] = tl[1] - 10
    tr[1] = tr[1] - 10

    # adjusment bl(x) & tl(x) agar lebih kiri
    bl[0] = bl[0] - 10
    tl[0] = tl[0] - 10

    # adjusment br(x) & tr(x) agar lebih kanan
    br[0] = br[0] + 10
    tr[0] = tr[0] + 10
    return np.int0([tl, tr, br, bl])

# menghitung lokasi box pada gambar original dari gambar ROI
def get_box_original(plate_rect, roi_rect, ratio):
    (x1, y1), (x2, y2) = plate_rect[0], plate_rect[2]
    x1_, y1_, x2_, y2_ = roi_rect
    l = int(x1/ratio) + x1_
    t = int(y1/ratio) + y1_
    r = int(x2/ratio) + x1_
    b = int(y2/ratio) + y1_
    return l, t, r, b

# Plate Number Transformator 
def plate_transform(plate_text):

    plate_text = list(plate_text.strip())
    original_letter =  ['i', 'I', 'O', 'Q', 'A', 'S', 'Z', 'z', 'o', 'G', 'B']
    transform_letter = ['1', '1', '0', '0', '4', '5', '2', '2', '0', '6', '8']
    if len(plate_text) > 5:
        for i, letter in enumerate(original_letter) :
            if letter in plate_text[2:-3] :
                plate_text[2:-3] = [item.replace(letter, transform_letter[i]) for item in plate_text[2:-3]]
    return ''.join(plate_text)

# main program
cap = cv2.VideoCapture(0)

THRESHOLD_DISTANCE = 30 #cm - jika objek di jarak < 30 cm , maka camera akan capture

try :
    while True :
        dist = distance()
        print ("Measured Distance = %.1f cm" % dist)
        
        if dist < THRESHOLD_DISTANCE :
            ret, img = cap.read()

            if not ret :
                print("can't capture image")
            else :
                e1 = cv2.getTickCount()
                h, w, c = img.shape
                thresh, roi, roi_rect, ratio = preprocessing(img, h1=0.50*h, h2=0.85*h, w1=0.3*w, w2=0.7*w)
                
                e2 = cv2.getTickCount()
                cv_time =  (e2 - e1)/ cv2.getTickFrequency()
                print("Preprocessing time : %.4fs\n" % cv_time )

                # get contour
                e1 = cv2.getTickCount()
                contours = get_contours(thresh)
                pts = []

                if len(contours) < 3 :
                    print("[STOP] tidak dapat mendeteksi plat nomor, contour terlalu sedikit..")
                else :
                    # get 4 point coordinate plate | pts = tl, tr, br, bl
                    pts = get_plate_4_coord(contours)

                    # geometric transform & crop plate number box
                    plate_img, plate_rect = transform(roi, pts, padding=0)
                    plate_img = np.uint8(plate_img)

                    e2 = cv2.getTickCount()
                    cv_time =  (e2 - e1)/ cv2.getTickFrequency()
                    print("Detecting License Plate Location time : %.4fs\n" % cv_time )
                    
                    # recognize plate character
                    e1 = cv2.getTickCount()
                    configuration = ("--oem 1 --psm 7")
                    plate_text = pytesseract.image_to_string(plate_img, config=configuration)
                    #plate_text = plate_transform(plate_text)
                    print("PREDICTED TEXT ", plate_text.strip())
                    e2 = cv2.getTickCount()
                    cv_time =  (e2 - e1)/ cv2.getTickFrequency()
                    print("OCR (Tesseract) time : %.4fs\n" % cv_time )

                    # draw label on image
                    e1 = cv2.getTickCount()
                    left, top, right, bottom = get_box_original(plate_rect, roi_rect, ratio)
                    img = drawPred(img, plate_text.strip(), left, top, right, bottom)
                    e2 = cv2.getTickCount()
                    cv_time =  (e2 - e1)/ cv2.getTickFrequency()
                    print("Postprocessing (Draw Box) time : %.4fs\n" % cv_time )


                    # send json to server
                    e1 = cv2.getTickCount()
                    index = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
                    print("INDEX ", index)
                    url = 'http://localhost:1880/parking'
                    parking_json = {
                                    "cred" : {
                                        "username" : "admin",
                                        "password" : "admin123"
                                            },
                                    "data" : {
                                            'timestamp' : int(time.time()),
                                            'predicted_plate' : plate_text.strip(),
                                            'car_image' : 'car_%s.jpg' % index,
                                            'plate_image' : 'plat_%s.jpg' % index,
                                            'client_name' : 'IN'
                                            }
                                    }

                    x = requests.post(url, json = parking_json)
                    print("PARKING ", x.headers['isSave'])
                    if x.headers['isSave'] == 'True' :
                        ServoGate()
                    e2 = cv2.getTickCount()
                    cv_time =  (e2 - e1)/ cv2.getTickFrequency()
                    print("Send JSON Data to Server time : %.4fs\n" % cv_time )

                    # send files to server
                    e1 = cv2.getTickCount()
                    car_name = 'car_%s.jpg' % index
                    plat_name = 'plat_%s.jpg' % index

                    cv2.imwrite('image_data/' + car_name, img)
                    cv2.imwrite('image_data/' + plat_name, plate_img)

                    url = 'http://localhost:1880/parking_data'

                    file_list = [  
                        ('car_image', (car_name, open('image_data/' + car_name, 'rb'), 'image/jpg')),
                        ('plate_image', (plat_name, open('image_data/' + plat_name, 'rb'), 'image/jpg'))
                    ]

                    data_json = {
                                "username" : "admin",
                                "password" : "admin123"
                                }

                    x = requests.post(url, files = file_list, data = data_json)

                    print("PARKING_DATA", x.text)
                    e2 = cv2.getTickCount()
                    cv_time =  (e2 - e1)/ cv2.getTickFrequency()
                    print("Send Processed Image to Server time : %.4fs\n" % cv_time )
            
        time.sleep(1)
except Exception as e:
    print("Exit from loop!")
    print("[ERROR]", e)
    p.stop()
    GPIO.cleanup()
    cap.release()