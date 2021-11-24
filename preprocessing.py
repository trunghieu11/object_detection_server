import jsonpickle
import numpy as np
import cv2
import tensorflow as tf
import grpc

from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis.predict_pb2 import PredictRequest
from flask import Flask, request, Response

# Initialize the Flask application
app = Flask(__name__)


# route http posts to this method
@app.route('/api/test', methods=['POST'])
def test():
    r = request
    
    # ============ Preprocessing ============
    # convert bytes of image data to uint8
    nparr = np.frombuffer(r.data, np.uint8)
    
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # preprocessing
    resized_image = preprocess_image(img)

    # ============ Create a request ============
    rq = PredictRequest()
    rq.model_spec.name = "yolov4-416"
    rq.model_spec.signature_name = "serving_default"
    rq.inputs["input_1"].CopyFrom(tf.make_tensor_proto(resized_image.astype(np.float32), shape=resized_image.shape))

    # ============ Sending a request ============
    channel = grpc.insecure_channel('209.97.162.90:8500')
    predict_service = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    response = predict_service.Predict(rq, timeout=60.0)

    # ============ Extracting the result ============
    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    class_ids=[]
    confidences=[]
    boxes=[]
    height, width, channels = img.shape

    for out in tf.make_ndarray(response.outputs["tf.concat_16"]):
        for detection in out:
            scores = detection[4:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1:
                x1 = int(detection[0] * width)
                y1 = int(detection[1] * height)
                x2 = int(detection[2] * width)
                y2 = int(detection[3] * height)
                
                boxes.append([x1, y1, x2, y2]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    for i in range(len(boxes)):
        if i in indexes:
            x1, y1, x2, y2 = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x1, y1),(x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 + 30), font, 5, color, 2)

    # ============ Returning the result ============
    # encode image
    _, frame = cv2.imencode('.jpg', img)

    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(frame)

    return Response(response=response_pickled, status=200, mimetype="application/json")

@app.route("/")
def ping_service():
    return "Hello, I'm preprocessing service"

def preprocess_image(image):
    resized_image = cv2.resize(image, (416, 416), interpolation=cv2.INTER_AREA)
    resized_image = resized_image[np.newaxis, ...]
    resized_image = resized_image / 255.0 # important
    return resized_image