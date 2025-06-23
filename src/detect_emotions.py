import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow logging

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import time  # For FPS calculation
import csv  # For logging data
from datetime import datetime # For timestamps

# --- Configuration ---
MODEL_PATH = 'emotion_detection_model_improved (2).h5' # <<< MAKE SURE THIS PATH IS CORRECT
HAARCASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
MODEL_INPUT_SHAPE = (48, 48, 1) 
RESIZE_INPUT_FRAME = True
FRAME_WIDTH = 640 # Width to resize to if RESIZE_INPUT_FRAME is True
EPS = 1e-9 # Epsilon to prevent division by zero in FPS calculation

# --- Logging Configuration ---
LOG_FILE = 'emotion_log.csv'
# *** IMPORTANT: Set a meaningful location name for where this camera is placed ***
CAMERA_LOCATION = "Museum Entrance" # E.g., "Exhibit Hall A", "Cafe Area", "Exit"

# --- Load Models ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Emotion detection model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    exit()

face_classifier = cv2.CascadeClassifier(HAARCASCADE_PATH)
if face_classifier.empty():
    print(f"Error loading Haar Cascade from {HAARCASCADE_PATH}")
    exit()
print("Haar Cascade face classifier loaded successfully.")

# --- Initialize CSV Log File ---
try:
    # Open in append mode ('a'). Create file if it doesn't exist.
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        # Write header only if the file is new/empty
        if f.tell() == 0:
            writer.writerow(['Timestamp', 'CameraLocation', 'Emotion', 'Confidence'])
    print(f"Logging data to {LOG_FILE}")
except IOError as e:
    print(f"Error initializing log file {LOG_FILE}: {e}. Exiting.")
    exit()

# --- Initialize Video Capture ---
cap = cv2.VideoCapture(0) # Use 0 for default camera, change if needed
if not cap.isOpened():
    print("Error: Could not open video capture device.")
    exit()

print("Starting video stream and emotion detection...")
print("Press 'q' to quit.")
prev_time = time.time() # Initialize prev_time for FPS calculation

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame. End of stream?")
        break

    # --- Frame Preprocessing ---
    frame_display = frame.copy() # Keep original frame for displaying results

    # Determine the frame to process for detection
    if RESIZE_INPUT_FRAME:
        # Calculate scale based on FRAME_WIDTH
        # Avoid division by zero if frame width is somehow 0
        if frame.shape[1] == 0: continue
        scale = FRAME_WIDTH / frame.shape[1]
        # Calculate new height maintaining aspect ratio
        new_height = int(frame.shape[0] * scale)
        # Resize the frame used for processing
        frame_processed = cv2.resize(frame, (FRAME_WIDTH, new_height), interpolation=cv2.INTER_AREA)
    else:
        frame_processed = frame # Process the original frame if not resizing
        scale = 1.0 # Scale is 1 if not resizing

    # Define the target frame for drawing bounding boxes and text
    # This should be the frame you intend to SHOW at the end.
    draw_target = frame_display # We draw on the original high-res frame

    # --- Grayscale and Face Detection (use the potentially resized frame_processed) ---
    gray = cv2.cvtColor(frame_processed, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray,
        scaleFactor=1.1, # Slightly less aggressive than 1.2
        minNeighbors=5,
        minSize=(30, 30) # Minimum face size to detect
    )

    face_rois = []
    # Store locations relative to the frame they were detected on (frame_processed)
    face_locations_processed = []

    # --- Prepare Face ROIs for Batch ---
    for (x, y, w, h) in faces:
        # Extract ROI from the grayscale processed frame
        roi_gray = gray[y:y+h, x:x+w]

        if roi_gray.size == 0: # Check if ROI is empty
            continue

        # Resize ROI to the model's expected input size
        roi_gray_resized = cv2.resize(roi_gray, (MODEL_INPUT_SHAPE[0], MODEL_INPUT_SHAPE[1]), interpolation=cv2.INTER_AREA)

        # Normalize and format for the model
        roi = roi_gray_resized.astype('float') / 255.0
        roi = img_to_array(roi)

        # Ensure correct channel dimension (add channel dim if model expects grayscale)
        if MODEL_INPUT_SHAPE[2] == 1 and len(roi.shape) < 3:
             roi = np.expand_dims(roi, axis=-1) # Add channel dimension: (48, 48) -> (48, 48, 1)
        # Optional: Handle if model expects 3 channels (e.g., convert grayscale back to BGR)
        # elif MODEL_INPUT_SHAPE[2] == 3:
        #     roi_color = cv2.cvtColor(roi_gray_resized, cv2.COLOR_GRAY2BGR)
        #     roi = roi_color.astype('float') / 255.0
        #     roi = img_to_array(roi)

        face_rois.append(roi)
        # Store coords relative to frame_processed
        face_locations_processed.append((x, y, w, h))

    # --- Batch Emotion Prediction ---
    if face_rois: # Only predict if faces were detected
        rois_batch = np.array(face_rois, dtype='float32') # Ensure correct dtype for TF
        try:
            predictions = model.predict(rois_batch, verbose=0) # verbose=0 suppresses Keras progress bar per batch

            # --- Process Predictions, Draw Results, and Log Data ---
            # Iterate using coords relative to frame_processed
            for i, (x_proc, y_proc, w_proc, h_proc) in enumerate(face_locations_processed):
                label_index = predictions[i].argmax() # Get index of highest probability
                label = EMOTION_LABELS[label_index]
                confidence = float(predictions[i][label_index]) # Get the confidence score, ensure it's a Python float

                # --- <<< LOGGING THE DATA >>> ---
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] # Timestamp with milliseconds
                try:
                    # Append data to the CSV file
                    with open(LOG_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp, CAMERA_LOCATION, label, confidence])
                except IOError as e:
                    print(f"Warning: Could not write to log file {LOG_FILE}: {e}")
                # --- <<< END OF LOGGING >>> ---

                # --- Drawing on the display frame ---
                # Map coordinates back from frame_processed to the display frame (draw_target)
                x_draw = int(x_proc / scale)
                y_draw = int(y_proc / scale)
                w_draw = int(w_proc / scale)
                h_draw = int(h_proc / scale)

                # Draw bounding box
                cv2.rectangle(draw_target, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), (255, 165, 0), 2) # Orange box
                # Prepare text label
                label_text = f"{label}: {confidence:.2f}"
                # Put text above the bounding box
                cv2.putText(draw_target, label_text, (x_draw, y_draw - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2) # Green text

        except Exception as e:
            print(f"Error during prediction or processing: {e}")
            # Optional: Draw basic red boxes on error to indicate detection failure post-processing
            for i, (x_proc, y_proc, w_proc, h_proc) in enumerate(face_locations_processed):
                x_draw = int(x_proc / scale)
                y_draw = int(y_proc / scale)
                w_draw = int(w_proc / scale)
                h_draw = int(h_proc / scale)
                cv2.rectangle(draw_target, (x_draw, y_draw), (x_draw + w_draw, y_draw + h_draw), (0, 0, 255), 1) # Red box


    # --- Calculate and Display FPS ---
    current_time = time.time()
    # Calculate FPS based on time difference, add EPS to avoid division by zero
    fps = 1 / (current_time - prev_time + EPS)
    prev_time = current_time
    # Draw FPS on the display frame (draw_target)
    cv2.putText(draw_target, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Red FPS text

    # --- Display Frame ---
    # Show the frame with all drawings
    cv2.imshow('Emotion Recognition - Press Q to Quit', draw_target)

    # --- Exit Condition ---
    # Wait for 1ms, check if 'q' was pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quit signal received.")
        break

# --- Cleanup ---
print("Releasing video capture and closing windows...")
cap.release()
cv2.destroyAllWindows()
print(f"Emotion logging complete. Data saved in {LOG_FILE}")
print("Done.")