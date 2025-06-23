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
## Technologies Used

This project is built using a combination of powerful tools and libraries:

* **Python:** The primary programming language for all scripts.
* **TensorFlow/Keras:** Used for building, training, and deploying the deep learning model for facial emotion recognition.
* **OpenCV (`cv2`):** Essential for real-time video stream processing, face detection (using Haar Cascades), and overlaying detection results.
* **Pandas:** Utilized in `analyze_emotions.py` for efficient data manipulation and analysis of the emotion log.
* **NumPy:** Fundamental library for numerical operations, especially array manipulation within the machine learning pipeline.
* **Matplotlib:** (Primarily in `model.py`) for visualizing training history (loss and accuracy plots).
* **Scikit-learn:** (Specifically `sklearn.model_selection.train_test_split` in `model.py`) for splitting data into training and testing sets.

**Potential IoT Deployment Platforms:**
While the core logic is software-based, this system is designed with deployment on IoT edge devices in mind. Potential platforms include:
* **Raspberry Pi:** A popular low-cost, high-performance single-board computer suitable for edge AI applications.
* **NVIDIA Jetson Series:** Powerful embedded AI computing platforms optimized for deep learning inference at the edge, ideal for more demanding real-time processing.

## Installation

To get a copy of this project up and running on your local machine, follow these steps.

### Prerequisites

* Python 3.8+ (recommended)
* `pip` (Python package installer)

### Clone the Repository

First, clone the repository to your local machine using Git:

```bash
git clone [https://github.com/your-username/smart-museum-emotion-recognition.git](https://github.com/your-username/smart-museum-emotion-recognition.git)
cd smart-museum-emotion-recognition
