"""
AI Smart Face Analytics System
============================

Advanced computer vision system for comprehensive face analysis including:
- Face Detection & Tracking
- Age & Gender Prediction
- Emotion Detection
- Face Recognition
- Mask Detection
- Lighting Analysis
- Real-time Analytics Dashboard

Author: Rishabh Soni
Version: 1.0
"""

import cv2
import threading
import time
import numpy as np
import argparse
import csv
import os
from datetime import datetime
from collections import defaultdict
import pandas as pd

# Import our modular components
from detector import FaceDetector
from age_gender_model import AgeGenderPredictor
from emotion_model import EmotionDetector
from face_recognition import FaceRecognizer
from mask_detection import MaskDetector
from lighting_analysis import LightingAnalyzer
from utils import FPSCounter, DataLogger, draw_analytics_overlay

# Global variables for thread communication
frame_to_analyze = None
last_results = []
analysis_lock = threading.Lock()
is_running = True
frame_count = 0

class AISmartFaceAnalytics:
    """Main orchestrator for the AI Smart Face Analytics System"""

    def __init__(self):
        self.face_detector = FaceDetector()
        self.age_gender_predictor = AgeGenderPredictor()
        self.emotion_detector = EmotionDetector()
        self.face_recognizer = FaceRecognizer()
        self.mask_detector = MaskDetector()
        self.lighting_analyzer = LightingAnalyzer()
        self.fps_counter = FPSCounter()
        self.data_logger = DataLogger()

    def analyze_face(self, face_img, face_id):
        """Comprehensive face analysis pipeline"""
        results = {
            'face_id': face_id,
            'name': 'Unknown',
            'gender': 'Unknown',
            'age_range': 'Unknown',
            'emotion': 'Unknown',
            'mask_status': 'Unknown',
            'lighting_quality': 'Unknown',
            'confidence_scores': {},
            'timestamp': datetime.now().isoformat()
        }

        try:
            name, rec_conf = self.face_recognizer.recognize_face(face_img)
            results['name'] = name
            results['confidence_scores']['recognition'] = rec_conf
        except Exception as e:
            print(f"Recognition error for face {face_id}: {e}")

        try:
            gender, age_range, ag_conf = self.age_gender_predictor.predict(face_img)
            results['gender'] = gender
            results['age_range'] = age_range
            results['confidence_scores']['age_gender'] = ag_conf
        except Exception as e:
            print(f"Age/Gender error for face {face_id}: {e}")

        try:
            emotion, emo_conf = self.emotion_detector.detect_emotion(face_img)
            results['emotion'] = emotion
            results['confidence_scores']['emotion'] = emo_conf
        except Exception as e:
            print(f"Emotion error for face {face_id}: {e}")

        try:
            mask_status, mask_conf = self.mask_detector.detect_mask(face_img)
            results['mask_status'] = mask_status
            results['confidence_scores']['mask'] = mask_conf
        except Exception as e:
            print(f"Mask error for face {face_id}: {e}")

        try:
            lighting_quality = self.lighting_analyzer.analyze_lighting(face_img)
            results['lighting_quality'] = lighting_quality
        except Exception as e:
            print(f"Lighting error for face {face_id}: {e}")

        return results

def analysis_thread(analytics_system):
    """Background analysis thread for real-time processing"""
    global frame_to_analyze, last_results, is_running, frame_count

    print("[AI] Analysis thread started - Multi-Modal Face Analytics...")

    while is_running:
        current_frame = None
        with analysis_lock:
            if frame_to_analyze is not None:
                current_frame = frame_to_analyze.copy()
                frame_to_analyze = None
                frame_count += 1

        if current_frame is not None:
            # Face Detection & Tracking
            tracked_faces = analytics_system.face_detector.update_tracking(current_frame)

            new_results = []
            for face_id, bbox in tracked_faces:
                x1, y1, x2, y2 = map(int, bbox)

                # Crop face with padding for analysis
                pad = 20
                face_img = current_frame[max(0, y1-pad):min(y2+pad, current_frame.shape[0]),
                                    max(0, x1-pad):min(x2+pad, current_frame.shape[1])]

                if face_img.size > 0:
                    # Comprehensive face analysis
                    analysis_results = analytics_system.analyze_face(face_img, face_id)
                    analysis_results['bbox'] = bbox
                    new_results.append(analysis_results)

                    # Log to data logger
                    analytics_system.data_logger.log_detection(analysis_results, frame_count)

            with analysis_lock:
                last_results = new_results
        else:
            time.sleep(0.01)

def calculate_frame_analytics(results):
    """Calculate analytics for current frame only (not cumulative)"""
    frame_analytics = {
        'total_faces': len(results),
        'gender_counts': {},
        'emotion_counts': {},
        'mask_counts': {}
    }
    
    for result in results:
        # Count genders in current frame
        gender = result.get('gender', 'Unknown')
        if gender != 'Unknown':
            frame_analytics['gender_counts'][gender] = frame_analytics['gender_counts'].get(gender, 0) + 1
        
        # Count emotions in current frame
        emotion = result.get('emotion', 'Unknown')
        if emotion != 'Unknown':
            frame_analytics['emotion_counts'][emotion] = frame_analytics['emotion_counts'].get(emotion, 0) + 1
        
        # Count masks in current frame
        mask = result.get('mask_status', 'Unknown')
        if mask != 'Unknown':
            frame_analytics['mask_counts'][mask] = frame_analytics['mask_counts'].get(mask, 0) + 1
    
    return frame_analytics

def process_image(analytics_system, image_path, csv_path=None):
    """Process single image file"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Face Detection & Analysis for single image
    tracked_faces = analytics_system.face_detector.update_tracking(frame)

    results = []
    for face_id, bbox in tracked_faces:
        x1, y1, x2, y2 = map(int, bbox)

        # Crop face with padding
        pad = 20
        face_img = frame[max(0, y1-pad):min(y2+pad, frame.shape[0]),
                        max(0, x1-pad):min(x2+pad, frame.shape[1])]

        if face_img.size > 0:
            analysis_results = analytics_system.analyze_face(face_img, face_id)
            analysis_results['bbox'] = bbox
            results.append(analysis_results)

            # Log to data logger
            analytics_system.data_logger.log_detection(analysis_results, 0)

    # Render results with dynamic face-following attributes
    for result in results:
        draw_advanced_ui(frame, result)

    # Analytics overlay (current image only)
    frame_analytics = calculate_frame_analytics(results)
    draw_analytics_overlay(frame, frame_analytics)

    # Save to CSV if requested
    if csv_path:
        analytics_system.data_logger.save_to_csv(csv_path)

    cv2.imshow("AI Smart Face Analytics - Image Analysis", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def draw_advanced_ui(frame, result):
    """Draw face attributes dynamically next to face detection"""
    face_id = result['face_id']
    bbox = result['bbox']
    name = result['name']
    gender = result['gender']
    age_range = result['age_range']
    emotion = result['emotion']
    mask_status = result['mask_status']
    lighting_quality = result['lighting_quality']
    confidence_scores = result['confidence_scores']

    x1, y1, x2, y2 = map(int, bbox)

    # Color coding based on gender
    color = (255, 191, 0) if gender == 'Male' else (203, 192, 255)  # Orange for male, purple for female

    # Draw clean bounding box with cyber-tech style
    l, t = 15, 3
    cv2.line(frame, (x1, y1), (x1+l, y1), color, t)
    cv2.line(frame, (x1, y1), (x1, y1+l), color, t)
    cv2.line(frame, (x2, y1), (x2-l, y1), color, t)
    cv2.line(frame, (x2, y1), (x2, y1+l), color, t)
    cv2.line(frame, (x1, y2), (x1+l, y2), color, t)
    cv2.line(frame, (x1, y2), (x1, y2-l), color, t)
    cv2.line(frame, (x2, y2), (x2-l, y2), color, t)
    cv2.line(frame, (x2, y2), (x2, y2-l), color, t)

    # ID label above face
    face_label = f"Face #{face_id}"
    font = cv2.FONT_HERSHEY_DUPLEX
    (w, h), _ = cv2.getTextSize(face_label, font, 0.5, 1)
    cv2.rectangle(frame, (x1, y1-h-8), (x1+w+8, y1), color, -1)
    cv2.putText(frame, face_label, (x1+4, y1-4), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Determine position for attributes (right or left of face)
    frame_width = frame.shape[1]
    attrs_x = x2 + 10  # Default: right side
    
    # If too close to right edge, place on left
    if x2 + 250 > frame_width:
        attrs_x = x1 - 250
    
    attrs_y = y1
    font_size = 0.35
    font_thickness = 1
    line_height = 15
    
    # Draw semi-transparent background for attributes
    attr_texts = [
        f"Name: {name}",
        f"Gender: {gender}",
        f"Age: {age_range}",
        f"Emotion: {emotion}",
        f"Mask: {mask_status}",
        f"Lighting: {lighting_quality}"
    ]
    
    bg_height = len(attr_texts) * line_height + 10
    bg_width = 160
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (attrs_x-5, attrs_y-5), (attrs_x+bg_width, attrs_y+bg_height), 
                (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.85, overlay, 0.15, 0, frame)
    cv2.rectangle(frame, (attrs_x-5, attrs_y-5), (attrs_x+bg_width, attrs_y+bg_height), 
                color, 1)
    
    # Draw attribute text
    text_y = attrs_y + 12
    
    cv2.putText(frame, f"Name: {name}", (attrs_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
    text_y += line_height
    
    cv2.putText(frame, f"Gender: {gender}", (attrs_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
    text_y += line_height
    
    cv2.putText(frame, f"Age: {age_range}", (attrs_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), font_thickness, cv2.LINE_AA)
    text_y += line_height
    
    # Emotion with color
    emotion_colors = {
        'Happy': (0, 255, 0), 'Sad': (255, 0, 0), 'Angry': (0, 0, 255),
        'Surprised': (255, 255, 0), 'Neutral': (128, 128, 128),
        'Fear': (128, 0, 128), 'Disgust': (0, 128, 128)
    }
    emotion_color = emotion_colors.get(emotion, (255, 255, 255))
    cv2.putText(frame, f"Emotion: {emotion}", (attrs_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, emotion_color, font_thickness, cv2.LINE_AA)
    text_y += line_height
    
    # Mask status with color
    mask_color = (0, 255, 0) if mask_status == 'No Mask' else (0, 165, 255)
    cv2.putText(frame, f"Mask: {mask_status}", (attrs_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, mask_color, font_thickness, cv2.LINE_AA)
    text_y += line_height
    
    # Lighting with color
    lighting_color = (0, 255, 0) if lighting_quality == 'Good' else (0, 165, 255)
    cv2.putText(frame, f"Lighting: {lighting_quality}", (attrs_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, lighting_color, font_thickness, cv2.LINE_AA)

def draw_detailed_sidebar(frame, results):
    """Deprecated: Now using dynamic face-following attributes. This function is kept for compatibility."""
    pass  # Attributes are now drawn by draw_advanced_ui() next to each face

def main():
    parser = argparse.ArgumentParser(description='AI Smart Face Analytics System')
    parser.add_argument('--source', type=str, default='webcam',
                       help='Source: webcam, image path, or video path')
    parser.add_argument('--output', type=str, help='Output video path (for video sources)')
    parser.add_argument('--csv', type=str, help='CSV file to save results')
    parser.add_argument('--dashboard', action='store_true', help='Launch analytics dashboard')

    args = parser.parse_args()

    global frame_to_analyze, last_results, is_running, frame_count

    try:
        analytics_system = AISmartFaceAnalytics()
    except Exception as e:
        print(f"Error initializing AI components: {e}")
        return

    if args.source == 'webcam':
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not access camera.")
            return
        is_video = False
    elif args.source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"Error: Could not open video file {args.source}")
            return
        is_video = True
    else:
        # Assume it's an image
        process_image(analytics_system, args.source, args.csv)
        return

    # For video/webcam
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))
    else:
        out = None

    # Start background analysis thread
    thread = threading.Thread(target=analysis_thread, args=(analytics_system,), daemon=True)
    thread.start()

    print("\n" + "="*60)
    print(" 🤖 AI SMART FACE ANALYTICS SYSTEM v1.0")
    print("="*60)
    print("Features: Face Detection | Age/Gender | Emotion | Recognition")
    print("         Mask Detection | Lighting Analysis | Real-time Tracking")
    print(f"Source: {args.source}")
    if args.output:
        print(f"Output: {args.output}")
    print("Tracking: Advanced Centroid Multi-Object Tracker")
    print("Analytics: Real-time Statistics & Data Logging")
    print("Press 'Q' to Quit | 'D' for Dashboard\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            if is_video:
                break  # End of video
            else:
                continue  # Webcam error, try again

        analytics_system.fps_counter.update()

        # Pass a copy to the analysis thread
        with analysis_lock:
            if frame_to_analyze is None:
                frame_to_analyze = frame.copy()
            current_results = last_results.copy()

        # Deduplicate results by face_id (prevent same face from appearing multiple times)
        unique_results = {}
        for result in current_results:
            face_id = result['face_id']
            if face_id not in unique_results:
                unique_results[face_id] = result
        current_results = list(unique_results.values())

        # Render advanced UI for each detected face (with dynamic attributes)
        for result in current_results:
            draw_advanced_ui(frame, result)

        # Analytics overlay summary (compact) - showing CURRENT FRAME stats
        frame_analytics = calculate_frame_analytics(current_results)
        draw_analytics_overlay(frame, frame_analytics)

        # Tech overlay with FPS
        fps = analytics_system.fps_counter.get_fps()
        overlay_text = f"AI SMART ANALYTICS | FPS: {fps:.1f} | Faces: {len(current_results)}"
        if is_video:
            overlay_text += " | VIDEO MODE"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, overlay_text, (15, 28),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        if out:
            out.write(frame)

        cv2.imshow("AI Smart Face Analytics System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            is_running = False
            break
        elif key == ord('d') or args.dashboard:
            # Launch dashboard
            launch_dashboard(analytics_system.data_logger)
            args.dashboard = False  # Only launch once

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

    # Save results to CSV
    if args.csv:
        analytics_system.data_logger.save_to_csv(args.csv)
        print(f"Results saved to {args.csv}")

    print("\n[System] AI Smart Face Analytics shutdown complete.")

def launch_dashboard(data_logger):
    """Launch the analytics dashboard"""
    try:
        import streamlit as st
        import subprocess
        # This would launch a separate Streamlit dashboard
        # For now, we'll create a simple matplotlib dashboard
        create_matplotlib_dashboard(data_logger)
    except ImportError:
        print("Streamlit not installed. Creating matplotlib dashboard...")
        create_matplotlib_dashboard(data_logger)

def create_matplotlib_dashboard(data_logger):
    """Create a matplotlib-based analytics dashboard"""
    try:
        import matplotlib.pyplot as plt

        analytics = data_logger.get_analytics()

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('AI Smart Face Analytics Dashboard', fontsize=16)

        # Gender distribution
        if analytics['gender_counts']:
            axes[0,0].bar(analytics['gender_counts'].keys(), analytics['gender_counts'].values())
            axes[0,0].set_title('Gender Distribution')
            axes[0,0].set_ylabel('Count')

        # Age distribution
        if analytics['age_counts']:
            ages = list(analytics['age_counts'].keys())
            counts = list(analytics['age_counts'].values())
            axes[0,1].bar(range(len(ages)), counts)
            axes[0,1].set_xticks(range(len(ages)))
            axes[0,1].set_xticklabels(ages, rotation=45)
            axes[0,1].set_title('Age Distribution')
            axes[0,1].set_ylabel('Count')

        # Emotion distribution
        if analytics['emotion_counts']:
            emotions = list(analytics['emotion_counts'].keys())
            counts = list(analytics['emotion_counts'].values())
            axes[1,0].pie(counts, labels=emotions, autopct='%1.1f%%')
            axes[1,0].set_title('Emotion Distribution')

        # Mask distribution
        if analytics['mask_counts']:
            axes[1,1].bar(analytics['mask_counts'].keys(), analytics['mask_counts'].values())
            axes[1,1].set_title('Mask Detection')
            axes[1,1].set_ylabel('Count')

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available for dashboard")

if __name__ == "__main__":
    main()
