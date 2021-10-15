# api json http://192.168.43.253:1880/parking
# json : timestamp, predicted_plate, car_image, plate_image, client_name
#
# api file http://192.168.43.253:1880/parking_data

import requests
import time 

url = 'http://192.168.43.253:1880/parking'
parking_json = {'timestamp' : int(time.time()),
                'predicted_plate' : 'D 123 XX',
                'car_image' : 'car_001.jpg',
                'plat_image' : 'plat_001.jpg',
                'client_name' : 'rpi_001'}

x = requests.post(url, json = parking_json)

print(x.text)