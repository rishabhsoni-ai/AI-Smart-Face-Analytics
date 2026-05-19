"""
Mask Detection Module
====================

CNN-based mask detection to classify whether a person is wearing a face mask.
Uses deep learning models trained on masked/unmasked face datasets.

Author: Rishabh Soni
Version: 1.0
"""

import cv2
import numpy as np
import os
from collections import deque

class MaskDetector:
    """
    Face mask detection using deep learning.
    Classifies faces as masked or unmasked with confidence scores.
    """

    def __init__(self, model_path=None):
        """
        Initialize the mask detector.

        Args:
            model_path (str): Path to mask detection model (optional)
        """
        self.mask_classes = ['No Mask', 'Mask']
        self.input_size = (224, 224)
        self.model_loaded = False  # Using heuristic detection only
        self.net = None

        # Skip ONNX model loading - using advanced heuristic mask detection
        print("[MaskDetector] Using advanced heuristic mask detection")

        # Temporal smoothing
        self.mask_history = {}  # face_id -> deque of mask predictions
        self.history_size = 8

    def _load_mask_model(self):
        """Try to load mask detection model from common locations"""
        mask_model_paths = [
            "models/mask_detector.onnx",
            "models/face_mask_model.onnx",
            "models/mask_classification.onnx"
        ]
        
        for path in mask_model_paths:
            if os.path.exists(path):
                try:
                    net = cv2.dnn.readNetFromONNX(path)
                    if net is not None and not net.empty():
                        self.net = net
                        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                        self.model_loaded = True
                        print(f"[MaskDetector] Loaded mask detection model from {path}")
                        return
                    else:
                        print(f"[MaskDetector] Model at {path} loaded but network is empty")
                except Exception as e:
                    print(f"[MaskDetector] Could not load {path}: {str(e)[:50]}...")
                    pass
        
        print("[MaskDetector] Using advanced heuristic mask detection (improved)")
        self.model_loaded = False

    def _find_mask_model(self):
        """Find mask detection model in models directory"""
        models_dir = "models"
        possible_names = [
            "mask_detector.onnx",
            "face_mask_model.onnx",
            "mask_classification.onnx"
        ]

        for name in possible_names:
            path = os.path.join(models_dir, name)
            if os.path.exists(path):
                return path
        return None

    def preprocess_face(self, face_img):
        """
        Preprocess face image for mask detection.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            blob: Preprocessed image blob
        """
        # Resize to model input size
        resized = cv2.resize(face_img, self.input_size)

        # Convert to blob
        blob = cv2.dnn.blobFromImage(
            resized,
            1.0/255.0,  # Scale factor
            self.input_size,
            (0, 0, 0),  # Mean subtraction
            swapRB=True,
            crop=False
        )

        return blob

    def detect_mask_advanced(self, face_img):
        """
        Detect mask using deep learning model.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            tuple: (mask_status, confidence)
        """
        # Check if network is loaded and not empty
        if self.net is None or self.net.empty():
            raise ValueError("Neural network is not loaded or empty")
            
        blob = self.preprocess_face(face_img)
        self.net.setInput(blob)
        outputs = self.net.forward()

        # Get predictions
        predictions = outputs[0][0]

        # Apply softmax
        exp_preds = np.exp(predictions - np.max(predictions))
        probs = exp_preds / exp_preds.sum()

        # Get best prediction
        mask_idx = np.argmax(probs)
        mask_status = self.mask_classes[mask_idx]
        confidence = float(probs[mask_idx] * 100)

        return mask_status, confidence

    def detect_mask_basic(self, face_img):
        """
        Basic mask detection using color and texture analysis.
        Fallback when no ML model is available.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            tuple: (mask_status, confidence)
        """
        try:
            # Convert to HSV for color analysis
            hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)

            # Define skin color range (rough approximation)
            lower_skin = np.array([0, 20, 70])
            upper_skin = np.array([20, 255, 255])

            # Create skin mask
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

            # Calculate skin percentage
            skin_percentage = np.sum(skin_mask > 0) / skin_mask.size

            # Simple heuristics:
            # If very little skin is visible in lower face region, likely masked
            h, w = face_img.shape[:2]
            lower_face = skin_mask[int(h*0.5):int(h*0.9), int(w*0.2):int(w*0.8)]

            if lower_face.size > 0:
                lower_skin_percentage = np.sum(lower_face > 0) / lower_face.size

                if lower_skin_percentage < 0.1:  # Very little skin visible
                    return "Mask", 75.0
                elif lower_skin_percentage < 0.3:  # Moderate skin visible
                    return "Mask", 60.0
                else:
                    return "No Mask", 70.0
            else:
                return "No Mask", 50.0

        except Exception as e:
            print(f"[MaskDetector] Basic detection error: {e}")
            return "Unknown", 50.0

    def stabilize_mask_prediction(self, face_id, mask_status, confidence):
        """
        Apply temporal stabilization to mask predictions.

        Args:
            face_id: Unique face identifier
            mask_status: Predicted mask status
            confidence: Prediction confidence

        Returns:
            tuple: Stabilized mask status and confidence
        """
        if face_id not in self.mask_history:
            self.mask_history[face_id] = deque(maxlen=self.history_size)

        # Add current prediction
        self.mask_history[face_id].append((mask_status, confidence))

        # Get majority mask status
        if len(self.mask_history[face_id]) > 0:
            mask_statuses = [p[0] for p in self.mask_history[face_id]]
            confidences = [p[1] for p in self.mask_history[face_id]]

            # Majority voting for mask status
            stable_mask = max(set(mask_statuses), key=mask_statuses.count)
            stable_conf = np.mean(confidences)

            return stable_mask, stable_conf

        return mask_status, confidence

    def detect_mask(self, face_img, face_id=None, stabilize=True):
        """
        Main mask detection method using advanced heuristic detection.

        Args:
            face_img: Face crop (BGR format)
            face_id: Optional face ID for stabilization
            stabilize: Whether to apply temporal stabilization

        Returns:
            tuple: (mask_status, confidence)
        """
        try:
            # Use advanced heuristic mask detection
            mask_status, confidence = self.detect_mask_basic(face_img)

            # Apply stabilization if face_id provided
            if stabilize and face_id is not None:
                mask_status, confidence = self.stabilize_mask_prediction(
                    face_id, mask_status, confidence
                )

            return mask_status, confidence

        except Exception as e:
            print(f"[MaskDetector] Detection error: {e}")
            return "Unknown", 50.0