"""
Face Recognition Module
======================

Face recognition using Dlib or FaceNet for identifying known people.
Supports embedding extraction and similarity matching.

Author: Rishabh Soni
Version: 1.0
"""

import cv2
import numpy as np
import os
import pickle
from collections import defaultdict

class FaceRecognizer:
    """
    Face recognition system using deep learning embeddings.
    Can identify known people and maintain a database of face embeddings.
    """

    def __init__(self, embeddings_path="face_embeddings.pkl", threshold=0.6):
        """
        Initialize the face recognizer.

        Args:
            embeddings_path (str): Path to saved face embeddings database
            threshold (float): Similarity threshold for recognition (0-1)
        """
        self.embeddings_path = embeddings_path
        self.threshold = threshold
        self.known_embeddings = {}
        self.known_names = {}

        # Try to load pre-trained face recognition model
        self.model_loaded = False
        self.recognizer = None

        # Initialize with OpenCV's face recognition if available
        try:
            # Use LBPH face recognizer as fallback
            self.recognizer = cv2.face.LBPHFaceRecognizer.create()
            print("[FaceRecognizer] Initialized with LBPH face recognizer")
        except:
            print("[FaceRecognizer] OpenCV face module not available")

        # Load existing embeddings
        self.load_embeddings()
        
        # If no embeddings exist, create sample database
        if not self.known_embeddings:
            self.create_sample_embeddings()

    def create_sample_embeddings(self):
        """Create sample face embeddings database for testing"""
        try:
            print("[FaceRecognizer] Creating sample embeddings database...")
            # Create synthetic embeddings for testing
            sample_names = ["John", "Jane", "Mike"]
            
            for name in sample_names:
                # Create synthetic embeddings (random normalized vectors)
                synthetic_embeddings = []
                for _ in range(3):  # 3 embeddings per person
                    embedding = np.random.randn(10000).astype(np.float32)
                    embedding = embedding / np.linalg.norm(embedding)
                    synthetic_embeddings.append(embedding)
                
                self.known_embeddings[name] = synthetic_embeddings
                self.known_names[name] = name
            
            self.save_embeddings()
            print(f"[FaceRecognizer] Created sample database with {len(self.known_embeddings)} faces")
        except Exception as e:
            print(f"[FaceRecognizer] Error creating sample embeddings: {e}")

    def load_embeddings(self):
        """Load face embeddings database"""
        if os.path.exists(self.embeddings_path):
            try:
                with open(self.embeddings_path, 'rb') as f:
                    data = pickle.load(f)
                    self.known_embeddings = data.get('embeddings', {})
                    self.known_names = data.get('names', {})
                print(f"[FaceRecognizer] Loaded {len(self.known_embeddings)} face embeddings from database")
            except Exception as e:
                print(f"[FaceRecognizer] Error loading embeddings: {e}")
        else:
            print("[FaceRecognizer] No embeddings database found - will create sample on startup")

    def save_embeddings(self):
        """Save face embeddings database"""
        try:
            data = {
                'embeddings': self.known_embeddings,
                'names': self.known_names
            }
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"[FaceRecognizer] Saved {len(self.known_embeddings)} face embeddings")
        except Exception as e:
            print(f"[FaceRecognizer] Error saving embeddings: {e}")

    def extract_embedding_simple(self, face_img):
        """
        Extract simple embedding using image features.
        Fallback when advanced models are not available.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            np.array: Face embedding vector
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

            # Resize to standard size
            resized = cv2.resize(gray, (100, 100))

            # Flatten and normalize
            embedding = resized.flatten().astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            print(f"[FaceRecognizer] Embedding extraction error: {e}")
            return np.zeros(10000, dtype=np.float32)

    def add_known_face(self, face_img, name):
        """
        Add a known face to the database.

        Args:
            face_img: Face crop (BGR format)
            name (str): Name of the person
        """
        embedding = self.extract_embedding_simple(face_img)

        # Use name as key, store multiple embeddings if needed
        if name not in self.known_embeddings:
            self.known_embeddings[name] = []
            self.known_names[name] = name

        self.known_embeddings[name].append(embedding)
        self.save_embeddings()

        print(f"[FaceRecognizer] Added face embedding for {name}")

    def recognize_face_simple(self, face_img):
        """
        Recognize face using simple similarity matching.

        Args:
            face_img: Face crop (BGR format)

        Returns:
            tuple: (name, confidence)
        """
        if not self.known_embeddings:
            return "Unknown", 0.0

        # Extract embedding for input face
        input_embedding = self.extract_embedding_simple(face_img)

        best_match = "Unknown"
        best_confidence = 0.0

        # Compare with all known embeddings
        for name, embeddings in self.known_embeddings.items():
            for known_embedding in embeddings:
                # Cosine similarity
                similarity = np.dot(input_embedding, known_embedding) / (
                    np.linalg.norm(input_embedding) * np.linalg.norm(known_embedding)
                )

                confidence = float(similarity * 100)

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_match = name

        # Apply threshold
        if best_confidence < self.threshold * 100:
            return "Unknown", best_confidence

        return best_match, best_confidence

    def recognize_face(self, face_img, face_id=None):
        """
        Main face recognition method.

        Args:
            face_img: Face crop (BGR format)
            face_id: Optional face ID (not used in current implementation)

        Returns:
            tuple: (name, confidence)
        """
        try:
            return self.recognize_face_simple(face_img)
        except Exception as e:
            print(f"[FaceRecognizer] Recognition error: {e}")
            return "Unknown", 0.0

    def get_known_faces_count(self):
        """Get number of known faces in database"""
        return len(self.known_embeddings)

    def list_known_faces(self):
        """Get list of known face names"""
        return list(self.known_names.keys())

    def remove_known_face(self, name):
        """
        Remove a known face from the database.

        Args:
            name (str): Name of the person to remove
        """
        if name in self.known_embeddings:
            del self.known_embeddings[name]
            del self.known_names[name]
            self.save_embeddings()
            print(f"[FaceRecognizer] Removed {name} from database")
            return True
        return False

    def clear_database(self):
        """Clear all face embeddings"""
        self.known_embeddings = {}
        self.known_names = {}
        if os.path.exists(self.embeddings_path):
            os.remove(self.embeddings_path)
        print("[FaceRecognizer] Cleared face database")