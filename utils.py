"""
Utilities Module
================

Helper functions and classes for the AI Smart Face Analytics System.
Includes FPS counter, data logger, and UI drawing utilities.

Author: AI Assistant
Version: 1.0
"""

import cv2
import time
import csv
import os
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np

class FPSCounter:
    """
    FPS (Frames Per Second) counter for performance monitoring.
    """

    def __init__(self):
        self.prev_time = time.time()
        self.fps = 0.0
        self.frame_count = 0

    def update(self):
        """Update FPS calculation"""
        current_time = time.time()
        self.frame_count += 1

        # Update FPS every second
        if current_time - self.prev_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.prev_time)
            self.frame_count = 0
            self.prev_time = current_time

    def get_fps(self):
        """Get current FPS value"""
        return self.fps

class DataLogger:
    """
    Data logger for collecting and analyzing detection results.
    Maintains statistics and can export to CSV/dashboard.
    """

    def __init__(self):
        self.detections = []
        self.analytics = {
            'total_faces': 0,
            'gender_counts': defaultdict(int),
            'age_counts': defaultdict(int),
            'emotion_counts': defaultdict(int),
            'mask_counts': defaultdict(int),
            'lighting_counts': defaultdict(int),
            'avg_confidence': defaultdict(list),
            'frame_count': 0
        }

    def log_detection(self, result, frame_number=0):
        """
        Log a detection result.

        Args:
            result (dict): Detection result from analysis pipeline
            frame_number (int): Current frame number
        """
        # Add frame number and timestamp
        log_entry = result.copy()
        log_entry['frame_number'] = frame_number
        log_entry['timestamp'] = datetime.now().isoformat()

        self.detections.append(log_entry)

        # Update analytics
        self.analytics['total_faces'] += 1

        # Update counters
        if result['gender'] != 'Unknown':
            self.analytics['gender_counts'][result['gender']] += 1

        if result['age_range'] != 'Unknown':
            self.analytics['age_counts'][result['age_range']] += 1

        if result['emotion'] != 'Unknown':
            self.analytics['emotion_counts'][result['emotion']] += 1

        if result['mask_status'] != 'Unknown':
            self.analytics['mask_counts'][result['mask_status']] += 1

        if result['lighting_quality'] != 'Unknown':
            self.analytics['lighting_counts'][result['lighting_quality']] += 1

        # Update confidence averages
        for key, value in result['confidence_scores'].items():
            if key not in self.analytics['avg_confidence']:
                self.analytics['avg_confidence'][key] = []
            self.analytics['avg_confidence'][key].append(value)

    def get_analytics(self):
        """
        Get current analytics summary.

        Returns:
            dict: Analytics data
        """
        import copy
        analytics = copy.deepcopy(self.analytics)

        # Calculate averages
        for key in analytics['avg_confidence']:
            if analytics['avg_confidence'][key]:
                analytics['avg_confidence'][key] = np.mean(analytics['avg_confidence'][key])
            else:
                analytics['avg_confidence'][key] = 0.0

        # Calculate average age
        if analytics['age_counts']:
            age_ranges = list(analytics['age_counts'].keys())
            age_counts = list(analytics['age_counts'].values())

            # Convert age ranges to midpoints for average calculation
            age_midpoints = []
            for age_range in age_ranges:
                if age_range == '65+':
                    midpoint = 70
                elif '-' in age_range:
                    start, end = age_range.split('-')
                    midpoint = (int(start) + int(end)) / 2
                else:
                    midpoint = 0
                age_midpoints.append(midpoint)

            if age_midpoints and age_counts:
                analytics['avg_age'] = np.average(age_midpoints, weights=age_counts)
            else:
                analytics['avg_age'] = 0.0
        else:
            analytics['avg_age'] = 0.0

        return analytics

    def save_to_csv(self, csv_path):
        """
        Save all detections to CSV file.

        Args:
            csv_path (str): Path to output CSV file
        """
        if not self.detections:
            print("[DataLogger] No detections to save")
            return

        try:
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'frame_number', 'face_id', 'name', 'gender',
                    'age_range', 'emotion', 'mask_status', 'lighting_quality',
                    'recognition_conf', 'age_gender_conf', 'emotion_conf', 'mask_conf'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for detection in self.detections:
                    row = {
                        'timestamp': detection.get('timestamp', ''),
                        'frame_number': detection.get('frame_number', 0),
                        'face_id': detection.get('face_id', 0),
                        'name': detection.get('name', 'Unknown'),
                        'gender': detection.get('gender', 'Unknown'),
                        'age_range': detection.get('age_range', 'Unknown'),
                        'emotion': detection.get('emotion', 'Unknown'),
                        'mask_status': detection.get('mask_status', 'Unknown'),
                        'lighting_quality': detection.get('lighting_quality', 'Unknown'),
                        'recognition_conf': detection.get('confidence_scores', {}).get('recognition', 0),
                        'age_gender_conf': detection.get('confidence_scores', {}).get('age_gender', 0),
                        'emotion_conf': detection.get('confidence_scores', {}).get('emotion', 0),
                        'mask_conf': detection.get('confidence_scores', {}).get('mask', 0)
                    }
                    writer.writerow(row)

            print(f"[DataLogger] Saved {len(self.detections)} detections to {csv_path}")

        except Exception as e:
            print(f"[DataLogger] Error saving to CSV: {e}")

    def reset(self):
        """Reset all logged data"""
        self.detections = []
        self.analytics = {
            'total_faces': 0,
            'gender_counts': defaultdict(int),
            'age_counts': defaultdict(int),
            'emotion_counts': defaultdict(int),
            'mask_counts': defaultdict(int),
            'lighting_counts': defaultdict(int),
            'avg_confidence': defaultdict(float),
            'frame_count': 0
        }

def draw_analytics_overlay(frame, analytics):
    """
    Draw compact analytics overlay on frame (minimal size).

    Args:
        frame: Video frame
        analytics: Analytics data from DataLogger
    """
    try:
        h, w = frame.shape[:2]

        # Create small overlay in top-right corner
        overlay_x = w - 220
        overlay_y = 50
        overlay_w = 210
        overlay_h = 90

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (overlay_x, overlay_y),
                     (overlay_x + overlay_w, overlay_y + overlay_h),
                     (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.85, overlay, 0.15, 0, frame)

        # Draw border
        cv2.rectangle(frame, (overlay_x, overlay_y),
                     (overlay_x + overlay_w, overlay_y + overlay_h),
                     (100, 150, 255), 1)

        # Compact analytics text
        y_offset = overlay_y + 16
        line_height = 16
        font_size = 0.35

        cv2.putText(frame, f"Faces: {analytics.get('total_faces', 0)}",
                   (overlay_x + 8, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 200, 255), 1)

        # Gender stats
        gender_counts = analytics.get('gender_counts', {})
        if gender_counts:
            male_count = gender_counts.get('Male', 0)
            female_count = gender_counts.get('Female', 0)
            cv2.putText(frame, f"M:{male_count} F:{female_count}",
                       (overlay_x + 8, y_offset + line_height), cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 200, 255), 1)

        # Top emotion
        emotion_counts = analytics.get('emotion_counts', {})
        if emotion_counts:
            top_emotion = max(emotion_counts.items(), key=lambda x: x[1])
            cv2.putText(frame, f"Top: {top_emotion[0]}",
                       (overlay_x + 8, y_offset + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 200, 255), 1)

        # Mask stats
        mask_counts = analytics.get('mask_counts', {})
        if mask_counts:
            mask_count = mask_counts.get('Mask', 0)
            no_mask_count = mask_counts.get('No Mask', 0)
            cv2.putText(frame, f"Masked: {mask_count}/{no_mask_count}",
                       (overlay_x + 8, y_offset + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX, font_size, (100, 200, 255), 1)

    except Exception as e:
        print(f"[Utils] Analytics overlay error: {e}")

def create_directories():
    """Create necessary directories for the project"""
    dirs = ['models', 'data', 'output', 'logs']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"[Utils] Created directory: {dir_name}")

def get_video_properties(cap):
    """
    Get video properties from capture object.

    Args:
        cap: OpenCV VideoCapture object

    Returns:
        dict: Video properties
    """
    return {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }

def validate_image_file(file_path):
    """
    Validate if file is a valid image.

    Args:
        file_path (str): Path to image file

    Returns:
        bool: True if valid image
    """
    try:
        img = cv2.imread(file_path)
        return img is not None and img.size > 0
    except:
        return False

def validate_video_file(file_path):
    """
    Validate if file is a valid video.

    Args:
        file_path (str): Path to video file

    Returns:
        bool: True if valid video
    """
    try:
        cap = cv2.VideoCapture(file_path)
        valid = cap.isOpened()
        cap.release()
        return valid
    except:
        return False