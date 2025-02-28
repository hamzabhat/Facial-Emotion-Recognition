import tensorflow as tf
import cv2
import numpy as np
import tf_keras
from tensorflow import keras
from keras import layers
from tf_keras import layers

# Corrected path with raw string or double backslashes
cascade_path = r' '
model_path = r' '

new_model = tf_keras.models.load_model(model_path)

# OpenCV setup
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8  # Adjusted for a more professional look
thickness = 2

# Prepare a face ROI placeholder
face_roi = np.zeros((224, 224, 3), dtype=np.uint8)

# Open webcam using DirectShow backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Load Haarcascade using the correct absolute path
faceCascade = cv2.CascadeClassifier(cascade_path)

# Create a resizable window and set a larger size
window_name = 'Face Emotion Recognition'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 1000, 800)  # Set desired window size

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optionally, flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

    # Default variables in case no face is detected
    x, y, w, h = 0, 0, 0, 0

    for (x, y, w, h) in faces:
        # Draw a preliminary blue rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detect inner features (optional, can be used for refined detection)
        faces_inner = faceCascade.detectMultiScale(roi_gray)
        if len(faces_inner) == 0:
            print("Face not detected")
        else:
            for (ex, ey, ew, eh) in faces_inner:
                face_roi = roi_color[ey: ey + eh, ex: ex + ew]
                # Break after first inner face to avoid overwriting
                break
        # Break after processing the first face
        break

    # Preprocess the face image for the model only if a face region is captured
    if face_roi.size != 0:
        try:
            final_image = cv2.resize(face_roi, (224, 224))
        except Exception as e:
            print("Error resizing face ROI:", e)
            continue
    else:
        # If no face is detected, use a blank image
        final_image = np.zeros((224, 224, 3), dtype=np.uint8)

    final_image = np.expand_dims(final_image, axis=0)  # Add batch dimension
    final_image = final_image / 255.0  # Normalize pixel values

    Predictions = new_model.predict(final_image)
    emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    status = emotion_labels[np.argmax(Predictions)]

    # Define colors based on emotion (customize as needed)
    if status == "Neutral":
        text_color = (0, 255, 0)
        box_color = (0, 255, 0)
    else:
        text_color = (0, 0, 255)
        box_color = (0, 0, 255)

    # Improved text box design: Draw a semi-transparent rectangle at the top left corner
    overlay = frame.copy()
    box_x, box_y, box_w, box_h = 10, 10, 250, 70
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (50, 50, 50), -1)
    alpha = 0.6  # Transparency factor
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Display emotion text in the text box
    cv2.putText(frame, f'Emotion: {status}', (box_x + 10, box_y + 45), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Draw a refined bounding box around the face with a thicker border
    if w and h:
        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 3)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
