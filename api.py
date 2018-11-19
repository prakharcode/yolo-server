from yolo.yolo_model import YOLO
from utils import detect_image
from flask import Flask, Response, request as req
from flask_restful import Resource, Api
import jsonpickle
import numpy as np
import cv2
from keras import backend as K

def load_model():
    return YOLO(0.9, 0.5, 'tiny_yolo.h5')


app = Flask(__name__)
api = Api(app)

class Detection(Resource):
    def post(self):
        try:
            model = load_model()
        except:
            print("Model Failed to load")

        try:
            nparr = np.fromstring(req.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except:
            print("Image failed to load")
        try:
            response = detect_image(img, model)
            print(response[1:])
            response_pickled = jsonpickle.encode(response)
        except:
            print("meh")
        return Response(response=response_pickled, status=200, mimetype="application/json")

api.add_resource(Detection, '/')

if __name__ == '__main__':
    app.run(debug = True)
