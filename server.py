import os
from yolo.yolo_model import YOLO
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from utils import detect_image
import base64
import cv2
import numpy as np
import keras.backend as K
up_loc = os.path.join('static','uploads')
res_loc = os.path.join('static','result')
UPLOAD_FOLDER = os.path.join(os.getcwd(), up_loc)
RESULT_FOLDER = os.path.join(os.getcwd(), res_loc)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['RESULT_FOLDER'] = RESULT_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def hello():
    K.clear_session()
    if request.method == 'POST':
        algo = request.form['algo']
        if 'file_input' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file_input']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            file_data = file.stream.read()
            nparr = np.fromstring(file_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if algo:
                yolo = YOLO(0.6, 0.5, 'yolo.h5')
                # result = detect_image(cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename)), yolo)
                result = detect_image(img, yolo)
                print(result[1]) # result will be list of tuple where each tuple is ( class, prob, [box coords])
                img_str = cv2.imencode('.jpg', result[0])[1].tostring()
                encoded = base64.b64encode(img_str).decode("utf-8")
                mime = "image/jpg;"
                out_image = f"data:{mime}base64,{encoded}"
                # cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'],filename), result[0])
            else:
                yolo = YOLO(0.8, 0.5, 'tiny_yolo.h5')
                # result = detect_image(cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename)), yolo)
                result = detect_image(img, yolo)
                print(result[1]) # result will be list of tuple where each tuple is ( class, prob, [box coords])
                img_str = cv2.imencode('.jpg', result[0])[1].tostring()
                encoded = base64.b64encode(img_str).decode("utf-8")
                mime = "image/jpg;"
                out_image = f"data:{mime}base64,{encoded}"
                # cv2.imwrite(os.path.join(app.config['RESULT_FOLDER'],filename), result[0])
            return render_template('result.html', out_image=out_image)
        else:
            return "File extension not supported"
    return render_template('index.html')

if __name__ == '__main__':
    app.secret_key = "key_key"
    app.run(debug=True)
