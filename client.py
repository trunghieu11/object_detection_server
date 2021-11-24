from __future__ import print_function
import jsonpickle
import requests
import json
import cv2
import numpy as np
import time

addr = 'http://localhost:1234'
# addr = "http://138.197.104.52:1234/"
test_url = addr + '/api/test'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

tic = time.time()
# send http request with image and receive response
response = requests.post(test_url, data=open('example_bike.jpg', 'rb').read(), headers=headers)
print("API Call time: {}".format(time.time() - tic))

if response.status_code == 200:
    frame = jsonpickle.decode(response.text)
    frame = frame.tobytes()

    # convert string of image data to uint8
    nparr = np.frombuffer(frame, np.uint8)

    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    cv2.imwrite("processed_image.jpg", img)
else:
    print("Error with status_code={}".format(response.status_code))