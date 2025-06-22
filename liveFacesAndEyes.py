import cv2 
import numpy as np

# HAAR Cascade files (use absolute paths)
cascade_face = r"C:\Users\SAURABH\ComputerVision-Projects\FaceEyeDetection\cascades\haarcascade_frontalface_default.xml"
cascade_eye = r"C:\Users\SAURABH\ComputerVision-Projects\FaceEyeDetection\cascades\haarcascade_eye.xml"

face_classifier = cv2.CascadeClassifier(cascade_face)
eye_classifier = cv2.CascadeClassifier(cascade_eye)

# Check if classifiers loaded correctly
if face_classifier.empty() or eye_classifier.empty():
    print("❌ Error: Haarcascade XML files could not be loaded. Check file paths.")
    exit()

def face_and_eye_detector(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_classifier.detectMultiScale(gray, 1.2, 5)

    if len(faces) == 0:
        return image  # Return the original image if no face is detected

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 3)  # Red box for face

        # Crop the detected face
        area_gray = gray[y:y+h, x:x+w]
        area_original = image[y:y+h, x:x+w]

        # Detect eyes within the face region
        eyes = eye_classifier.detectMultiScale(area_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(area_original, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)  # Green box for eyes

    # Flip image horizontally (for mirror effect)
    image = cv2.flip(image, 1)
    return image		

# Open webcam
capture = cv2.VideoCapture(0)

while True:
    response, frame = capture.read()
    
    if not response:
        print("❌ Error: Failed to read frame from webcam.")
        break

    processed_frame = face_and_eye_detector(frame)
    cv2.imshow("Live Face and Eye Classifier", processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
