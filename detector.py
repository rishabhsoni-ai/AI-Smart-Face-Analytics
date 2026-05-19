"""
Face Detection Module
====================

Handles face detection and tracking using YuNet ONNX model with centroid-based tracking.
Provides stable face IDs across frames for consistent identification.

Author: Rishabh Soni
Version: 1.0
"""

import cv2
import numpy as np
import os
from collections import OrderedDict
from scipy.spatial import distance as dist

class CentroidTracker:
    """
    Centroid-based multi-object tracker for maintaining consistent face IDs.
    Tracks faces across frames using centroid distance calculations.
    """

    def __init__(self, max_disappeared=50, max_distance=50):
        """
        Initialize the centroid tracker.

        Args:
            max_disappeared (int): Maximum frames a face can be missing before deregistering
            max_distance (int): Maximum pixel distance for centroid matching
        """
        self.next_object_id = 0
        self.objects = OrderedDict()  # object_id -> (centroid_x, centroid_y)
        self.disappeared = OrderedDict()  # object_id -> frames_missing
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        """Register a new face with a unique ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1
        return self.next_object_id - 1

    def deregister(self, object_id):
        """Remove a face that has been missing too long"""
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        """
        Update tracker with new face bounding boxes.

        Args:
            rects: List of (startX, startY, endX, endY) tuples

        Returns:
            dict: object_id -> (centroid_x, centroid_y)
        """
        if len(rects) == 0:
            # No faces detected, increment disappeared counters
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # Calculate centroids for input rectangles
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            # First frame with faces, register all
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            # Match existing faces to new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Calculate distance matrix
            D = dist.cdist(np.array(object_centroids), input_centroids)

            # Find best matches using Hungarian algorithm approach
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing faces
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            # Handle unmatched new faces
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

class FaceDetector:
    """
    Face detection using YuNet ONNX model with integrated tracking.
    Provides real-time face detection with stable IDs.
    """

    def __init__(self, input_size=(320, 320), confidence_threshold=0.6, nms_threshold=0.3):
        """
        Initialize the face detector.

        Args:
            input_size (tuple): Input size for YuNet model (width, height)
            confidence_threshold (float): Minimum confidence for face detection
            nms_threshold (float): Non-maximum suppression threshold
        """
        self.input_size = input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        # Initialize YuNet face detector
        model_path = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet.onnx")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YuNet model not found at {model_path}. Ensure models are included in the repo.")

        self.face_detector = cv2.FaceDetectorYN.create(
            model_path, "", self.input_size,
            self.confidence_threshold, self.nms_threshold, 5000
        )

        if self.face_detector is None:
            raise RuntimeError("Failed to initialize YuNet face detector. Check model path.")

        # Initialize tracker
        self.tracker = CentroidTracker(max_disappeared=30, max_distance=80)

        print("[FaceDetector] Initialized with YuNet ONNX model")

    def update_tracking(self, frame):
        """
        Detect faces and update tracking.

        Args:
            frame: Input image/frame (BGR format)

        Returns:
            list: [(face_id, bbox), ...] where bbox is (x1, y1, x2, y2)
        """
        # Set input size for current frame
        height, width = frame.shape[:2]
        self.face_detector.setInputSize((width, height))

        # Detect faces
        _, faces = self.face_detector.detect(frame)

        rects = []
        if faces is not None:
            for face in faces:
                x, y, w, h = map(int, face[:4])
                rects.append((x, y, x + w, y + h))

        # Update tracker
        tracked_objects = self.tracker.update(rects)

        # Convert centroids back to bounding boxes
        final_detections = []
        for obj_id, centroid in tracked_objects.items():
            # Find closest rectangle to this centroid
            best_rect = None
            min_dist = float('inf')

            for rect in rects:
                rect_center_x = (rect[0] + rect[2]) / 2
                rect_center_y = (rect[1] + rect[3]) / 2
                distance = np.sqrt((centroid[0] - rect_center_x)**2 +
                                 (centroid[1] - rect_center_y)**2)

                if distance < min_dist:
                    min_dist = distance
                    best_rect = rect

            if best_rect and min_dist < 100:  # Reasonable distance threshold
                final_detections.append((obj_id, best_rect))

        return final_detections

    def get_face_count(self):
        """Get current number of tracked faces"""
        return len(self.tracker.objects)

    def reset_tracking(self):
        """Reset the tracker (useful for new video sequences)"""
        self.tracker = CentroidTracker(max_disappeared=30, max_distance=80)

class AgeGenderDetectorPro:
    def __init__(self, input_size=(320, 320)):
        # Face Detection (YuNet)
        self.face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
        
        # Age/Gender (Caffe Weights)
        self.age_net = cv2.dnn.readNet("models/age_net.caffemodel", "models/age_deploy.prototxt")
        self.gender_net = cv2.dnn.readNet("models/gender_net.caffemodel", "models/gender_deploy.prototxt")
        
        # Metadata for Age Regression
        # These are the midpoints of the age buckets
        self.age_points = [1.5, 5.0, 10.0, 17.5, 28.5, 40.5, 50.5, 80.0]
        self.gender_list = ['Male', 'Female']
        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        
        # Tracking & Stability
        self.tracker = CentroidTracker(max_disappeared=30)
        self.history_size = 30
        self.predictions_history = {} # object_id -> deque

    def update_tracking(self, frame):
        self.face_detector.setInputSize((frame.shape[1], frame.shape[0]))
        _, faces = self.face_detector.detect(frame)
        
        rects = []
        if faces is not None:
            for face in faces:
                x1, y1, w, h = map(int, face[:4])
                rects.append((x1, y1, x1+w, y1+h))
        
        tracked_objects = self.tracker.update(rects)
        
        final_detections = []
        for obj_id, centroid in tracked_objects.items():
            best_rect = None
            min_dist = 9999
            for rect in rects:
                r_centerX = (rect[0] + rect[2]) / 2
                r_centerY = (rect[1] + rect[3]) / 2
                d = np.sqrt((centroid[0]-r_centerX)**2 + (centroid[1]-r_centerY)**2)
                if d < min_dist:
                    min_dist = d
                    best_rect = rect
            
            if best_rect:
                final_detections.append((obj_id, best_rect))
        
        return final_detections

    def analyze_face(self, face_img):
        try:
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            
            # Gender Classification
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            gender = self.gender_list[gender_preds[0].argmax()]
            
            # Precise Age Calculation (Weighted Average of Buckets)
            self.age_net.setInput(blob)
            age_preds = self.age_net.forward()[0]
            
            # Softmax to get clean probabilities
            exp_preds = np.exp(age_preds - np.max(age_preds))
            probs = exp_preds / exp_preds.sum()
            
            # Calculate Weighted Age
            weighted_age = np.sum(probs * self.age_points)
            
            return gender, weighted_age
        except:
            return None, None

    def age_to_range(self, age):
        """Convert exact age to age range (5-year intervals)"""
        if age < 0:
            return "Unknown"
        elif age < 5:
            return "0-4"
        elif age < 10:
            return "5-9"
        elif age < 15:
            return "10-14"
        elif age < 20:
            return "15-19"
        elif age < 25:
            return "20-24"
        elif age < 30:
            return "25-29"
        elif age < 35:
            return "30-34"
        elif age < 40:
            return "35-39"
        elif age < 45:
            return "40-44"
        elif age < 50:
            return "45-49"
        elif age < 55:
            return "50-54"
        elif age < 60:
            return "55-59"
        elif age < 65:
            return "60-64"
        else:
            return "65+"

    def get_stable_prediction(self, obj_id, gender, age):
        if obj_id not in self.predictions_history:
            self.predictions_history[obj_id] = deque(maxlen=self.history_size)
        
        if gender is not None and age is not None:
            self.predictions_history[obj_id].append((gender, age))
        
        if not self.predictions_history[obj_id]:
            return "Scanning...", "Unknown"

        g_votes = [p[0] for p in self.predictions_history[obj_id]]
        a_values = [p[1] for p in self.predictions_history[obj_id]]
        
        stable_gender = Counter(g_votes).most_common(1)[0][0]
        # Use a rolling average for age to make it super smooth but responsive
        stable_age_exact = int(np.mean(a_values))
        # Convert to age range
        stable_age_range = self.age_to_range(stable_age_exact)
        
        return stable_gender, stable_age_range

    def draw_ui(self, frame, obj_id, box, gender, age):
        x1, y1, x2, y2 = map(int, box)
        color = (255, 191, 0) if gender == 'Male' else (203, 192, 255)
        
        # Cyber-Tech UI
        l, t = 20, 3
        cv2.line(frame, (x1, y1), (x1+l, y1), color, t)
        cv2.line(frame, (x1, y1), (x1, y1+l), color, t)
        cv2.line(frame, (x2, y1), (x2-l, y1), color, t)
        cv2.line(frame, (x2, y1), (x2, y1+l), color, t)
        cv2.line(frame, (x1, y2), (x1+l, y2), color, t)
        cv2.line(frame, (x1, y2), (x1, y2-l), color, t)
        cv2.line(frame, (x2, y2), (x2-l, y2), color, t)
        cv2.line(frame, (x2, y2), (x2, y2-l), color, t)

        # Label: Age Range format
        label = f"#{obj_id} | {gender} | AGE: {age}"
        font = cv2.FONT_HERSHEY_DUPLEX
        (w, h), _ = cv2.getTextSize(label, font, 0.5, 1)
        
        cv2.rectangle(frame, (x1, y1-h-15), (x1+w+10, y1), color, -1)
        cv2.putText(frame, label, (x1+5, y1-10), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
