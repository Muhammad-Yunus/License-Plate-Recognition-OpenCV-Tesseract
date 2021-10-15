import requests

url = 'http://192.168.43.253:1880/parking_data'

file_list = [  
       ('car_image', ('car_001.jpg', open('car_001.jpg', 'rb'), 'image/jpg')),
       ('plate_image', ('plat_001.jpg', open('plat_001.jpg', 'rb'), 'image/jpg'))
   ]

x = requests.post(url, files = file_list)

print(x.text)