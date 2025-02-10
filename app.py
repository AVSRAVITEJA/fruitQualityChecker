from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load trained model
MODEL_PATH = "banana_classifier.h5"
model = load_model(MODEL_PATH)

# Class labels
class_labels = {0: "Export Quality", 1: "Fresh", 2: "Rotten"}

# Set IP Webcam stream URL (Replace with your actual IP Webcam address)
MOBILE_CAM_URL = "http://192.168.226.50:8080/video"  # Update this with your IP Webcam stream

# Function to preprocess frames for ML model
def preprocess_frame(frame):
    image = cv2.resize(frame, (128, 128))  # Resize to model input size
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to capture frames from mobile camera
def generate_frames():
    cap = cv2.VideoCapture(MOBILE_CAM_URL)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Preprocess frame
        processed_frame = preprocess_frame(frame)

        # Predict the class
        predictions = model.predict(processed_frame)
        predicted_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        label = f"{class_labels[predicted_class]} ({confidence:.2f})"

        # Display the result on the frame
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Route to serve video stream
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
