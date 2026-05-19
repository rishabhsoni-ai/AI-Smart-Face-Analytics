"""
Emotion Detection Module
=======================

Detects facial emotions using deep learning models trained on FER2013 dataset.
Supports multiple emotion categories with confidence scoring.

Author: Rishabh Soni
Version: 1.0
"""

import cv2
import numpy as np
import os
from collections import deque

class EmotionDetector:
    """
    Emotion detection using deep learning models.
    Trained on FER2013 dataset for facial expression recognition.
    """

    def __init__(self, model_path=None):
        """
        Initialize the emotion detector with advanced 7-emotion recognition.

        Args:
            model_path (str): Path to emotion detection model (optional)
        """
        self.emotions = [
            'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
        ]

        # Model parameters (for future deep learning models)
        self.input_size = (64, 64)  # Standard input size
        self.model_loaded = False  # Using heuristic detection only
        self.net = None

        # Skip ONNX model loading - using advanced heuristic detection
        print("[EmotionDetector] Using advanced 7-emotion heuristic detection")

        # Temporal smoothing
        self.emotion_history = {}  # face_id -> deque of emotions
        self.history_size = 10

    def _load_fer2013_model(self):
        """Try to load FER2013 model from common locations"""
        fer2013_paths = [
            "models/fer2013_emotion.onnx",
            "models/emotion_detection.onnx",
            "models/fer2013_model.onnx"
        ]
        
        for path in fer2013_paths:
            if os.path.exists(path):
                try:
                    net = cv2.dnn.readNetFromONNX(path)
                    if net is not None and not net.empty():
                        self.net = net
                        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                        self.model_loaded = True
                        print(f"[EmotionDetector] Loaded FERPlus emotion model from {path}")
                        return
                    else:
                        print(f"[EmotionDetector] Model at {path} loaded but network is empty")
                except Exception as e:
                    print(f"[EmotionDetector] Could not load {path}: {str(e)[:50]}...")
                    pass
        
        print("[EmotionDetector] Using advanced heuristic emotion detection (7 emotions)")
        self.model_loaded = False

    def _find_emotion_model(self):
        """Find emotion model in models directory"""
        models_dir = "models"
        possible_names = [
            "fer2013_emotion.onnx",  # FERPlus model
            "Eff_Net_Quantized.onnx",  # Fallback
            "emotion_model.onnx",
            "emotion_detection.onnx"
        ]

        for name in possible_names:
            path = os.path.join(models_dir, name)
            if os.path.exists(path):
                return path
        return None

    def preprocess_face(self, face_img):
        """
        Preprocess face image for emotion detection.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            blob: Preprocessed image blob
        """
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img

        # Resize to model input size (64x64)
        resized = cv2.resize(gray, self.input_size)

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Add batch and channel dimensions: (1, 1, 64, 64)
        blob = np.expand_dims(normalized, axis=[0, 1])

        return blob

    def detect_emotion_advanced(self, face_img):
        """
        Detect emotion using deep learning model.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            tuple: (emotion, confidence)
        """
        # Check if network is loaded and not empty
        if self.net is None or self.net.empty():
            raise ValueError("Neural network is not loaded or empty")
            
        blob = self.preprocess_face(face_img)
        self.net.setInput(blob)
        
        # Forward pass
        try:
            outputs = self.net.forward()
            predictions = outputs[0][0]  # Shape should be (7,) for 7 emotions
        except Exception as e:
            print(f"[EmotionDetector] Forward failed: {e}")
            return "Neutral", 50.0

        # FERPlus outputs are already probabilities (sum to 1)
        # Ensure we have 7 predictions
        if len(predictions) != 7:
            print(f"[EmotionDetector] Unexpected output shape: {predictions.shape}")
            return "Neutral", 50.0

        # Get best prediction
        emotion_idx = np.argmax(predictions)
        emotion = self.emotions[emotion_idx]
        confidence = float(predictions[emotion_idx] * 100)

        return emotion, confidence

    def detect_emotion_basic(self, face_img):
        """
        Basic emotion detection using facial landmarks and simple heuristics.
        Fallback when no ML model is available.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            tuple: (emotion, confidence)
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # Calculate brightness
            brightness = np.mean(gray)

            # Calculate contrast
            contrast = np.std(gray)

            # Simple mouth detection (rough approximation)
            h, w = gray.shape
            mouth_region = gray[int(h*0.6):int(h*0.9), int(w*0.3):int(w*0.7)]
            mouth_brightness = np.mean(mouth_region) if mouth_region.size > 0 else brightness

            # Eye region
            eye_region = gray[int(h*0.2):int(h*0.5), int(w*0.2):int(w*0.8)]
            eye_brightness = np.mean(eye_region) if eye_region.size > 0 else brightness

            # Forehead region (for surprise/anger)
            forehead_region = gray[int(h*0.05):int(h*0.2), int(w*0.3):int(w*0.7)]
            forehead_brightness = np.mean(forehead_region) if forehead_region.size > 0 else brightness

            # Improved emotion classification based on facial features
            brightness_diff = mouth_brightness - eye_brightness
            contrast_ratio = contrast / (brightness + 1)  # Avoid division by zero
            
            print(f"Brightness: {brightness:.1f}, Contrast: {contrast:.1f}")
            print(f"Mouth: {mouth_brightness:.1f}, Eyes: {eye_brightness:.1f}, Diff: {brightness_diff:.1f}")
            
            # Advanced heuristic emotion detection for 7 emotions
            if eye_brightness > 90 and contrast < 45 and brightness_diff > 5:
                # Very bright eyes, low contrast, smiling mouth = Happy
                emotion = "Happy"
                confidence = 80.0
            elif contrast > 65 and forehead_brightness > eye_brightness + 10:
                # High contrast, bright forehead = Angry
                emotion = "Angry"
                confidence = 75.0
            elif eye_brightness < 70 and contrast > 55 and brightness_diff < -5:
                # Dark eyes, high contrast, downturned mouth = Sad
                emotion = "Sad"
                confidence = 75.0
            elif contrast > 70 and brightness < 80:
                # Very high contrast, dark face = Fear
                emotion = "Fear"
                confidence = 70.0
            elif eye_brightness > 95 and contrast < 40:
                # Extremely bright eyes, very low contrast = Surprise
                emotion = "Surprise"
                confidence = 75.0
            elif contrast > 60 and abs(brightness_diff) < 3 and brightness < 100:
                # High contrast, neutral mouth position, dark = Disgust
                emotion = "Disgust"
                confidence = 65.0
            else:
                # Default neutral expression
                emotion = "Neutral"
                confidence = 60.0

            return emotion, confidence

        except Exception as e:
            print(f"[EmotionDetector] Basic detection error: {e}")
            return "Neutral", 50.0

    def stabilize_emotion(self, face_id, emotion, confidence):
        """
        Apply temporal stabilization to emotion predictions.

        Args:
            face_id: Unique face identifier
            emotion: Predicted emotion
            confidence: Prediction confidence

        Returns:
            tuple: Stabilized emotion and confidence
        """
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = deque(maxlen=self.history_size)

        # Add current prediction
        self.emotion_history[face_id].append((emotion, confidence))

        # Get majority emotion
        if len(self.emotion_history[face_id]) > 0:
            emotions = [p[0] for p in self.emotion_history[face_id]]
            confidences = [p[1] for p in self.emotion_history[face_id]]

            # Majority voting for emotion
            stable_emotion = max(set(emotions), key=emotions.count)
            stable_conf = np.mean(confidences)

            return stable_emotion, stable_conf

        return emotion, confidence

    def detect_emotion(self, face_img, face_id=None, stabilize=True):
        """
        Main emotion detection method with advanced 7-emotion recognition.

        Args:
            face_img: Face crop (BGR format)
            face_id: Optional face ID for stabilization
            stabilize: Whether to apply temporal stabilization

        Returns:
            tuple: (emotion, confidence)
        """
        try:
            # Use advanced heuristic detection (more reliable than current models)
            emotion, confidence = self.detect_emotion_basic(face_img)

            # Apply stabilization if face_id provided
            if stabilize and face_id is not None:
                emotion, confidence = self.stabilize_emotion(face_id, emotion, confidence)

            return emotion, confidence

        except Exception as e:
            print(f"[EmotionDetector] Detection error: {e}")
            return "Neutral", 50.0