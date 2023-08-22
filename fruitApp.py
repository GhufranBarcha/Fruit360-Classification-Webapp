from flask import Flask, render_template, request
import joblib
import numpy as np
import cv2
import tensorflow as tf
import base64



app = Flask(__name__)

## import model and encoder 
model1   =  joblib.load("fruit360model.pkl")
encoder1 = joblib.load("LabelEncoder.pkl")

labels = {i: label for i, label in enumerate(encoder1.classes_)}
def preprocessing(image):
    input_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized_input_image = cv2.resize(input_img, (250, 250))
    flattened_input_image = resized_input_image.flatten().tolist()
    return np.array([flattened_input_image]) / 255

@app.route('/')
def main():
    return render_template("index.html")



@app.route('/upload', methods=['POST', 'GET'])
def upload():
    img = request.files["image"].read()
    img_array = np.frombuffer(img, np.uint8)
    uploaded_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    prediction = model1.predict(preprocessing(uploaded_image))
    predicted_label = labels[np.argmax(prediction)]

    _, img_encoded = cv2.imencode(".jpg", uploaded_image)
    img_base64 = base64.b64encode(img_encoded).decode("utf-8")

    return render_template("index.html", predicted=predicted_label, image=img_base64)



if __name__ == "__main__":
    app.run(port = 3000 ,debug =True)    