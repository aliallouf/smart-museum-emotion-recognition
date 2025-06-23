# smart-museum-emotion-recognition
## Project Overview

This project focuses on developing a real-time emotion recognition system designed for smart museum environments. Leveraging computer vision and machine learning, the system detects facial emotions from live video streams, logs the emotional responses of visitors, and provides automated analysis of this data. The ultimate goal is to enhance the visitor experience, analyze engagement with exhibits, and provide data-driven insights that could optimize museum layouts or content.

This initiative is developed as part of the **IoT Devices Programming** class for the **Master's Degree in Telecommunication** at the **University of Calabria (UNICAL)**.

## Features

Our Smart Museum Emotion Recognition system includes the following key functionalities:

* **Real-time Facial Emotion Detection:** Utilizes a pre-trained deep learning model (TensorFlow/Keras) and OpenCV for accurate, real-time emotion classification from video input (e.g., from an IoT camera device).
* **Comprehensive Emotion Data Logging:** Automatically records detected emotions, their confidence levels, associated timestamps, and the specific camera location (e.g., "Museum Entrance," "Exhibit Hall A") to a CSV file (`emotion_log.csv`).
* **Automated Data Analysis:** Processes the collected `emotion_log.csv` data to generate insightful summaries, including:
    * Overall sentiment distribution (Positive, Negative, Neutral).
    * Counts and percentages of individual emotions (Happy, Sad, Angry, etc.).
    * Emotion trends analyzed by specific museum locations.
    * Hourly emotion distributions to identify peak emotional periods.
* **IoT Dashboard Integration Potential:** The analysis script (`analyze_emotions.py`) outputs results in JSON format, making it readily consumable by IoT dashboards or visualization tools like Node-RED for dynamic display and further analysis.

---
