"""
Lighting Analysis Module
=======================

Analyzes face region lighting quality and provides recommendations.
Classifies lighting as Good, Low, or Very Bright with improvement suggestions.

Author: Rishabh Soni
Version: 1.0
"""

import cv2
import numpy as np

class LightingAnalyzer:
    """
    Lighting quality analysis for face regions.
    Evaluates brightness levels and provides lighting condition classification.
    """

    def __init__(self):
        """Initialize the lighting analyzer"""
        # Lighting thresholds (adjustable)
        self.bright_threshold = 200  # Very bright
        self.good_upper_threshold = 180  # Good lighting upper bound
        self.good_lower_threshold = 80   # Good lighting lower bound
        self.low_threshold = 50          # Low lighting

    def analyze_lighting(self, face_img):
        """
        Analyze lighting quality of face region.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            str: Lighting quality classification
        """
        try:
            # Convert to grayscale for brightness analysis
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # Calculate average brightness
            avg_brightness = np.mean(gray)

            # Calculate brightness standard deviation (contrast)
            brightness_std = np.std(gray)

            # Classify lighting conditions
            if avg_brightness > self.bright_threshold:
                return "Very Bright"
            elif avg_brightness > self.good_upper_threshold:
                return "Good Lighting"
            elif avg_brightness > self.good_lower_threshold:
                # Check for uneven lighting (high contrast)
                if brightness_std > 60:
                    return "Uneven Lighting"
                else:
                    return "Good Lighting"
            elif avg_brightness > self.low_threshold:
                return "Low Lighting"
            else:
                return "Very Low Lighting"

        except Exception as e:
            print(f"[LightingAnalyzer] Analysis error: {e}")
            return "Unknown"

    def get_lighting_score(self, face_img):
        """
        Get detailed lighting score and recommendations.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            dict: Lighting analysis results
        """
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            avg_brightness = np.mean(gray)
            brightness_std = np.std(gray)

            # Calculate lighting score (0-100)
            # Optimal brightness range: 100-160
            if 100 <= avg_brightness <= 160:
                score = 100
            elif 80 <= avg_brightness <= 180:
                # Linear interpolation for scores
                if avg_brightness < 100:
                    score = 80 + (avg_brightness - 80) * 0.5
                else:
                    score = 100 - (avg_brightness - 160) * 0.5
            else:
                score = max(0, 100 - abs(avg_brightness - 130) * 0.5)

            # Penalty for uneven lighting
            if brightness_std > 50:
                score *= 0.8

            score = min(100, max(0, score))

            result = {
                'quality': self.analyze_lighting(face_img),
                'score': round(score, 1),
                'brightness': round(avg_brightness, 1),
                'contrast': round(brightness_std, 1),
                'recommendation': self._get_recommendation(avg_brightness, brightness_std)
            }

            return result

        except Exception as e:
            print(f"[LightingAnalyzer] Detailed analysis error: {e}")
            return {
                'quality': 'Unknown',
                'score': 0.0,
                'brightness': 0.0,
                'contrast': 0.0,
                'recommendation': 'Unable to analyze lighting'
            }

    def _get_recommendation(self, brightness, contrast):
        """
        Get lighting improvement recommendation.

        Args:
            brightness: Average brightness value
            contrast: Brightness standard deviation

        Returns:
            str: Recommendation message
        """
        if brightness > self.bright_threshold:
            return "Reduce light intensity or use diffused lighting"
        elif brightness < self.low_threshold:
            return "Increase lighting or move closer to light source"
        elif brightness < self.good_lower_threshold:
            return "Add more light to improve visibility"
        elif brightness > self.good_upper_threshold:
            return "Slightly reduce light intensity"
        elif contrast > 60:
            return "Use more even lighting to reduce shadows"
        else:
            return "Lighting conditions are good"