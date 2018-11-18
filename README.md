Highly recommended to have Tensorflow-gpu up and running.

After moving inside the directory. To run the project:

1. Install virtual environment
```bash
pip install virtualenv
```

2. Initiate virtual environment
```bash
virtualenv venv
```

3. Activate virtualenv
```bash
. venv/bin/activate
```

4. Install the requirements
```bash
pip install -r requirements.tex
```

5. Download and convert the weights for yolov3 and yolov3-tiny
```bash
wget https://pjreddie.com/media/files/yolov3.weights
 ./yad2k.py yolov3.cfg yolov3.weights yolo.h5  
wget https://pjreddie.com/media/files/yolov3-tiny.weights
./yad2k.py yolov3-tiny.cfg yolov3-tiny.weights tint_yolo.h5
```

6. Run the server
```bash
python server.py
```
