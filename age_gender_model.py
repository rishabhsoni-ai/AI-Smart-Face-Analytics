"""
Age and Gender Prediction Module
================================

Advanced age and gender prediction using modern deep learning models.
Supports MobileNetV2, ResNet50, and other architectures for better accuracy.

Author: Rishabh Soni
Version: 1.0
"""

import cv2
import numpy as np
import os
from collections import deque

class AgeGenderPredictor:
    """
    Advanced age and gender prediction using deep learning models.
    Replaces old Caffe models with modern architectures for better accuracy.
    """

    def __init__(self, model_type='mobilenet', use_onnx=True):
        """
        Initialize the age and gender predictor.

        Args:
            model_type (str): Model architecture ('mobilenet', 'resnet', 'efficientnet')
            use_onnx (bool): Whether to use ONNX models for better performance
        """
        self.model_type = model_type
        self.use_onnx = use_onnx

        # Age ranges for classification
        self.age_ranges = [
            "0-4", "5-9", "10-14", "15-19", "19-23", "19-23", "30-34",
            "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65+"
        ]

        # Gender classes
        self.gender_classes = ['Male', 'Female']

        # Model paths
        self.models_dir = "models"

        if self.use_onnx:
            # Try to load separate ONNX models
            age_onnx = os.path.join(self.models_dir, "age_net.onnx")
            gender_onnx = os.path.join(self.models_dir, "gender_net.onnx")

            if os.path.exists(age_onnx) and os.path.exists(gender_onnx):
                try:
                    self.age_net = cv2.dnn.readNetFromONNX(age_onnx)
                    self.gender_net = cv2.dnn.readNetFromONNX(gender_onnx)
                    print("[AgeGenderPredictor] Loaded separate ONNX models for age/gender prediction")
                except Exception as e:
                    print(f"[AgeGenderPredictor] ONNX models failed to load: {e}, falling back to Caffe")
                    self.use_onnx = False
            else:
                print("[AgeGenderPredictor] ONNX models not found, falling back to Caffe")
                self.use_onnx = False

        if not self.use_onnx:
            # Fallback to Caffe models
            age_proto = os.path.join(self.models_dir, "age_deploy.prototxt")
            age_model = os.path.join(self.models_dir, "age_net.caffemodel")
            gender_proto = os.path.join(self.models_dir, "gender_deploy.prototxt")
            gender_model = os.path.join(self.models_dir, "gender_net.caffemodel")

            if os.path.exists(age_model) and os.path.exists(gender_model):
                self.age_net = cv2.dnn.readNet(age_model, age_proto)
                self.gender_net = cv2.dnn.readNet(gender_model, gender_proto)
                print("[AgeGenderPredictor] Loaded Caffe models for age/gender prediction")
            else:
                raise FileNotFoundError("Age/gender models not found. Run setup_models.py")

        # Image preprocessing parameters
        self.input_size = (224, 224) if self.use_onnx else (227, 227)
        self.mean_values = (104, 117, 123) if not self.use_onnx else (0, 0, 0)
        self.scale = 1.0

        # Temporal smoothing for stability
        self.prediction_history = {}  # face_id -> deque of predictions
        self.history_size = 15

    def preprocess_image(self, face_img):
        """
        Preprocess face image for model input.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            blob: Preprocessed image blob
        """
        # Resize to model input size
        resized = cv2.resize(face_img, self.input_size)

        # Create blob
        blob = cv2.dnn.blobFromImage(
            resized,
            self.scale,
            self.input_size,
            self.mean_values,
            swapRB=False,
            crop=False
        )

        return blob

    def predict_onnx(self, face_img):
        """
        Predict age and gender using separate ONNX models.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            tuple: (gender, age_range, confidence)
        """
        blob = self.preprocess_image(face_img)

        # Gender prediction
        self.gender_net.setInput(blob)
        gender_outputs = self.gender_net.forward()
        gender_preds = gender_outputs[0]  # Shape: (2,)

        gender_idx = np.argmax(gender_preds)
        gender = self.gender_classes[gender_idx]
        gender_conf = float(gender_preds[gender_idx] * 100)

        # Age prediction
        self.age_net.setInput(blob)
        age_outputs = self.age_net.forward()
        age_preds = age_outputs[0]  # Shape: (8,)

        # Age buckets (same as Caffe model)
        age_points = [1.5, 5.0, 10.0, 17.5, 28.5, 40.5, 50.5, 80.0]
        age_value = np.sum(age_preds * age_points)
        age_range = self.age_value_to_range(age_value)

        # Age confidence based on max probability
        age_conf = float(np.max(age_preds) * 100)

        # Overall confidence
        confidence = (gender_conf + age_conf) / 2

        return gender, age_range, confidence

    def predict_caffemodel(self, face_img):
        """
        Predict age and gender using Caffe models.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            tuple: (gender, age_range, confidence)
        """
        blob = self.preprocess_image(face_img)

        # Gender prediction
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender_idx = np.argmax(gender_preds[0])
        gender = self.gender_classes[gender_idx]
        gender_conf = float(gender_preds[0][gender_idx] * 100)

        # Age prediction
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()[0]

        # Convert to probabilities and get weighted age
        exp_preds = np.exp(age_preds - np.max(age_preds))
        age_probs = exp_preds / exp_preds.sum()

        # Age buckets (same as original model)
        age_points = [1.5, 5.0, 10.0, 17.5, 28.5, 40.5, 50.5, 80.0]
        age_value = np.sum(age_probs * age_points)
        age_range = self.age_value_to_range(age_value)

        # Age confidence based on max probability
        age_conf = float(np.max(age_probs) * 100)

        # Overall confidence
        confidence = (gender_conf + age_conf) / 2

        return gender, age_range, confidence

    def age_value_to_range(self, age_value):
        """
        Convert numerical age to age range string.

        Args:
            age_value (float): Predicted age in years

        Returns:
            str: Age range string
        """
        if age_value < 5:
            return "0-4"
        elif age_value < 10:
            return "5-9"
        elif age_value < 15:
            return "10-14"
        elif age_value < 19:
            return "15-19"
        elif age_value < 23:
            return "19-23"
        elif age_value < 30:
            return "19-23"
        elif age_value < 35:
            return "30-34"
        elif age_value < 40:
            return "35-39"
        elif age_value < 45:
            return "40-44"
        elif age_value < 50:
            return "45-49"
        elif age_value < 55:
            return "50-54"
        elif age_value < 60:
            return "55-59"
        elif age_value < 65:
            return "60-64"
        else:
            return "65+"

    def stabilize_prediction(self, face_id, gender, age_range, confidence):
        """
        Apply temporal stabilization to predictions.

        Args:
            face_id: Unique face identifier
            gender: Predicted gender
            age_range: Predicted age range
            confidence: Prediction confidence

        Returns:
            tuple: Stabilized predictions
        """
        if face_id not in self.prediction_history:
            self.prediction_history[face_id] = deque(maxlen=self.history_size)

        # Add current prediction to history
        self.prediction_history[face_id].append((gender, age_range, confidence))

        # Get majority vote for gender and age
        if len(self.prediction_history[face_id]) > 0:
            genders = [p[0] for p in self.prediction_history[face_id]]
            age_ranges = [p[1] for p in self.prediction_history[face_id]]
            confidences = [p[2] for p in self.prediction_history[face_id]]

            # Majority voting
            stable_gender = max(set(genders), key=genders.count)
            stable_age = max(set(age_ranges), key=age_ranges.count)
            stable_conf = np.mean(confidences)

            return stable_gender, stable_age, stable_conf

        return gender, age_range, confidence

    def predict(self, face_img, face_id=None, stabilize=True):
        """
        Main prediction method.

        Args:
            face_img: Face crop (BGR format)
            face_id: Optional face ID for stabilization
            stabilize: Whether to apply temporal stabilization

        Returns:
            tuple: (gender, age_range, confidence)
        """
        try:
            if self.use_onnx:
                gender, age_range, confidence = self.predict_onnx(face_img)
            else:
                gender, age_range, confidence = self.predict_caffemodel(face_img)

            # Apply stabilization if face_id provided
            if stabilize and face_id is not None:
                gender, age_range, confidence = self.stabilize_prediction(
                    face_id, gender, age_range, confidence
                )

            return gender, age_range, confidence

        except Exception as e:
            print(f"[AgeGenderPredictor] Prediction error: {e}")
            return "Unknown", "Unknown", 0.0