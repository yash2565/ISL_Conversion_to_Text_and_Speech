# ISL_Conversion_to_Text_and_Speech


🎨 Sign Language Recognition Project 🌐

🔧 Overview

This project is designed to recognize hand gestures and translate them into corresponding sign language symbols using computer vision and machine learning techniques. The system aims to bridge the communication gap for individuals who rely on sign language by providing a practical and efficient recognition tool.

🏆 Highlights

The project leverages a robust neural network architecture and state-of-the-art image processing techniques to achieve accurate gesture recognition. It includes tools for training custom gesture models, real-time recognition, and visualization.

🎯 Objectives

Gesture Dataset Preparation: Create a comprehensive database of hand gestures.

Model Training: Train a CNN-based model to recognize gestures effectively.

Real-Time Recognition: Implement a real-time recognition system using webcam input.

Visualization Tools: Provide tools to visualize gesture data and recognition outcomes.

🔢 Features

Data Augmentation: Flip and preprocess images to enhance model training.

Visualization: Display confusion matrices and accuracy plots.

Custom Gesture Creation: Add new gestures and retrain models as needed.

Interactive GUI: A user-friendly interface for real-time gesture recognition.

🥇 Methodology

Data Collection: Capture images of hand gestures using a webcam.

Preprocessing: Prepare data by resizing and flipping images.

Model Training: Train a convolutional neural network (CNN) using Keras.

Testing and Evaluation: Evaluate the model using metrics like accuracy and confusion matrices.

Deployment: Use the trained model for real-time gesture recognition.

📊 Outcomes

The project demonstrates high accuracy in recognizing hand gestures, enabling effective translation of sign language symbols. The results include:

Well-trained CNN models stored in .h5 format.

Interactive real-time recognition through a GUI application.

Visual performance metrics such as confusion matrices and accuracy plots.

📈 Tools and Technologies

Programming Language: Python

Libraries: TensorFlow, Keras, OpenCV

Visualization: Matplotlib

Database: SQLite (for gesture storage)

🗋 Usage

Clone the repository and install dependencies from requirements_cpu.txt or requirements_gpu.txt.

Use create_gestures.py to generate custom gesture datasets.

Train models with cnn_keras.py or cnn_tf.py.

Launch the GUI application using signGUI_main.py.

📄 Files Overview

cnn_keras.py and cnn_tf.py: Model training scripts.

create_gestures.py: Tool for creating gesture datasets.

recognize_gesture.py: Script for real-time gesture recognition.

signGUI_main.py: Main GUI application.

gesture_db.db: SQLite database for gestures.

🛠️ Future Work

Enhance the model for multi-lingual sign language recognition.

Integrate advanced deep learning models for better accuracy.

Extend the GUI for more user-friendly experiences.

📊 Conclusion

This project is a significant step towards improving accessibility for individuals relying on sign language. By leveraging modern computer vision techniques, it provides an efficient and scalable solution for gesture recognition.

