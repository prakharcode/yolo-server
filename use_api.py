import cv2
import sys
import numpy as np
import base64
from requests import post



img = cv2.imread(sys.argv[1])
_, img_encoded = cv2.imencode('.jpg', img)
response = post('http://127.0.0.1:5000/', data=img_encoded.tostring())
print(response)
