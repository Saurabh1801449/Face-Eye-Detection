import cv2
import os

# Load Haarcascade files (Use absolute paths)
face_cascade = cv2.CascadeClassifier("C:/Users/SAURABH/ComputerVision-Projects/FaceEyeDetection/cascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:/Users/SAURABH/ComputerVision-Projects/FaceEyeDetection/cascades/haarcascade_eye.xml")

# Load Image
image_path = "C:/Users/SAURABH/ComputerVision-Projects/FaceEyeDetection/images/Hillary.jpg"
image = cv2.imread(image_path)

if image is None:
    print(f"❌ Error: Could not load image at '{image_path}'. Check the file path.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

if len(faces) == 0:  # ✅ Fixed empty tuple check
    print("❌ No faces detected!")
    
else:
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = image[y:y + h, x:x + w]

        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    # Show result
    cv2.imshow("Face & Eye Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
