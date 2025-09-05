import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



import os
import cv2
import numpy as np

def clear_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

def capture_and_crop_face():
    # Directory to save cropped face images
    save_directory = r"C:\Users\aluvo\OneDrive\Desktop\Capturr\integrated"

    # Clear the directory before saving the new cropped face image
    clear_directory(save_directory)

    # Initialize webcam capture
    cap = cv2.VideoCapture(0)

    # Load the Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from webcam.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangle around the faces and crop the first face detected
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_crop = gray[y:y+h, x:x+w]
            break

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            if 'face_crop' in locals():
                # Save the cropped face image in grayscale
                save_path = os.path.join(save_directory, 'cropped_face.jpg')
                cv2.imwrite(save_path, face_crop)
                print(f"Cropped face saved as '{save_path}'")
            break

    # When everything done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

capture_and_crop_face()



app = Flask(__name__)

model1 = load_model(r"C:\Users\aluvo\Downloads\monday_face.h5")
#model1 = load_model(r"C:\Users\aluvo\Downloads\sunday_face.h5")
model2 = load_model(r"C:\Users\aluvo\Downloads\final_ecg_model.h5")

classes = 30
threshold_face = 0.5
threshold_ecg = 0.7
print(threshold_face)
print(threshold_ecg)

@app.route("/")
def about():
    return render_template("home.html")

@app.route("/about")
def home():
    return render_template("home.html")

@app.route("/info")
def information():
    return render_template("information.html")

@app.route("/upload")
def test():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def upload():
    if request.method == 'POST':
        f1 = request.files['file1']
        f2 = request.files['file2']

        basepath = os.path.dirname(__file__)
        file1path = os.path.join(basepath, "uploads", f1.filename)
        file2path = os.path.join(basepath, "uploads", f2.filename)
        f1.save(file1path)
        f2.save(file2path)

        img1 = image.load_img(file1path, target_size=(224, 224))
        img2 = image.load_img(file2path, target_size=(224, 224))
        x1 = image.img_to_array(img1)
        x2 = image.img_to_array(img2)
        x1 = np.expand_dims(x1, axis=0)
        x2 = np.expand_dims(x2, axis=0)
        x1 = preprocess_input(x1)
        x2 = preprocess_input(x2)

        pred1 = model1.predict(x1)
        pred2 = model2.predict(x2)
        y_pred1 = np.argmax(pred1)
        y_pred2 = np.argmax(pred2)

        confidence1 = np.max(pred1)
        confidence2 = np.max(pred2)

        index = ['Authentication For Weapon Successful. "Person {}" Unlocked..'.format(i) for i in range(1, classes + 1)]

        if confidence1 >= threshold_face and confidence2 >= threshold_ecg:
            if y_pred1 < classes and y_pred2 < classes and index[y_pred1] == index[y_pred2]:
                result = index[y_pred1]
            else:
                result = "XXXX-----Authentication failed-----XXXX"
        else:
            if confidence1 < threshold_face:
                result = "Face or ECG Authentication Failed"
            elif confidence2 < threshold_ecg:
                result = "Face or ECG Authentication Failed"

        return result
    return "No files uploaded"

if __name__ == "__main__":
    app.run(debug=False)
