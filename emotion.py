import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
emotion_detector = FER()

while True:
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # Load the pre-trained model for emotion recognition

        # Output image's information

        # Preprocess the face region for emotion detection
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (48, 48))
            face_img = face_img / 255.0
            face_img = face_img.reshape(1, 48, 48, 1)
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Predict the emotion
            predictions = emotion_detector.detect_emotions(face_img)
            cv2.imshow('Emotion Detection', predictions)
            # Process predictions (e.g., find max confidence)
            # Display emotion on the face or print the result
        plt.show()
    else:
        break

cap.release()
cv2.destroyAllWindows()