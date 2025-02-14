from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

MODEL_PATH = "banana_quality_model_1.h5"
model = load_model(MODEL_PATH)

class_labels = {0: "overripe", 1: "ripe", 2: "rotten", 3: "unripe"}

MOBILE_CAM_URL = "http://100.65.66.233:8080/video"  

def preprocess_frame(frame):
    image = cv2.resize(frame, (150, 150))  
    image = img_to_array(image) / 255.0  
    image = np.expand_dims(image, axis=0)  
    image=image.reshape(1,150,150,3) #error was here fixed it by reshaping the image
    return image

def generate_frames():
    cap = cv2.VideoCapture(MOBILE_CAM_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 4)  

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

       
        processed_frame = preprocess_frame(frame)

        try:
            predictions = model.predict(processed_frame)  
            predicted_class = np.argmax(predictions[0])  
            confidence = float(np.max(predictions[0]))  
            label = f"{class_labels[predicted_class]} ({confidence:.2f})"  
        except Exception as e:
            label = f"Error: {e}"

        
        cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """Render the HTML page that displays the video feed."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream the processed video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
   
    app.run(debug=True, host='0.0.0.0')
