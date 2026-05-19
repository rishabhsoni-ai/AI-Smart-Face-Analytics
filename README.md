# 🤖 AI Smart Face Analytics System

**Advanced Computer Vision System for Comprehensive Face Analysis**

This project provides a state-of-the-art real-time face analytics system that extends basic age and gender detection with advanced features including emotion recognition, face identification, mask detection, lighting analysis, and interactive analytics dashboards.

## ✨ Features

### Core Analytics
- **Real-time Face Detection** - YuNet ONNX model for accurate face localization with tracking IDs
- **Advanced Age & Gender Prediction** - Modern deep learning models with 14 age ranges and gender classification
- **Emotion Detection** - 7 emotion categories (Happy, Sad, Angry, Neutral, Surprise, Fear, Disgust)
- **Face Recognition** - Identify known individuals with similarity matching
- **Mask Detection** - CNN-based mask wearing classification
- **Lighting Quality Analysis** - Automatic lighting condition assessment with recommendations

### Performance & Intelligence
- **Multi-threaded Architecture** - Background analysis for 25+ FPS real-time processing
- **Temporal Stabilization** - Smart filtering prevents prediction jitter
- **Confidence Scoring** - Every prediction includes accuracy percentage
- **Modular Design** - Clean, extensible codebase with separate analysis modules

### Data & Visualization
- **Comprehensive CSV Logging** - Detailed detection data with timestamps and frame numbers
- **Interactive Analytics Dashboard** - Real-time statistics with matplotlib visualizations
- **Multiple Input Sources** - Webcam, video files, and image processing
- **Video Output** - Save annotated videos with all analytics overlays

## 🚀 Quick Start

### Installation
```bash
# Clone or download the project
cd "AI Smart Face Analytics System"

# Install dependencies
pip install opencv-python numpy scipy matplotlib seaborn pandas

# Download models
python setup_models.py
```

### Usage

#### Webcam Analysis (Real-time)
```bash
python main.py --source webcam
```

#### Video File Processing
```bash
python main.py --source video.mp4 --output analyzed_video.mp4 --csv results.csv
```

#### Image Analysis
```bash
python main.py --source image.jpg --csv results.csv
```

#### Launch Analytics Dashboard
```bash
python main.py --source webcam --dashboard
```

#### Streamlit Deployment
```bash
pip install streamlit
streamlit run streamlit_app.py
```

Then open the local Streamlit URL displayed in your browser and upload an image or video for analysis.

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN.PY (Orchestrator)                    │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │ FaceDetector    AgeGenderPredictor    EmotionDetector │ │
│  │ MaskDetector    FaceRecognizer       LightingAnalyzer │ │
│  └─────────────────────────────────────────────────────────┘ │
│                           │                                   │
│  Analysis Thread ─────────┼─ UI Thread                        │
│  (Background Processing)  │ (Display & Logging)               │
└───────────────────────────┼───────────────────────────────────┘
                            │
                    ┌───────▼───────┐
                    │   UTILS       │
                    │ FPS Counter   │
                    │ Data Logger   │
                    │ Analytics     │
                    └───────────────┘
```

## 🎯 Advanced Features

### Face Recognition System
- Add known faces to database
- Automatic similarity matching
- Persistent embeddings storage

### Emotion Analysis
- FER2013-trained emotion classification
- Real-time mood detection
- Confidence-based filtering

### Mask Detection
- Medical mask recognition
- Safety compliance monitoring
- Statistical reporting

### Lighting Intelligence
- Automatic brightness analysis
- Quality classification (Good/Low/Very Bright)
- Improvement recommendations

### Analytics Dashboard
- Real-time statistics overlay
- Comprehensive matplotlib visualizations
- Gender, age, emotion distributions
- Confidence trend analysis

## 📈 Performance Specifications

- **Real-time Processing**: 25+ FPS on modern hardware
- **Face Tracking**: Centroid-based with ID stability
- **Memory Efficient**: Streaming processing, no full video loading
- **Multi-threaded**: UI and analysis run concurrently
- **Optimized Pipelines**: Face crops processed efficiently

## 📋 CSV Output Format

```csv
timestamp,frame_number,face_id,name,gender,age_range,emotion,mask_status,lighting_quality,recognition_conf,age_gender_conf,emotion_conf,mask_conf
2024-01-15 10:30:45.123,150,1,John Doe,Male,25-29,Happy,No Mask,Good Lighting,87.5,92.3,89.1,94.2
```

## 🎨 Visual Output

Bounding boxes display comprehensive information:
```
#1 | John Doe | Male | 25-29 | Happy | Mask: No (94%)
Gender: Male (92%) | Age: 25-29
Emotion: Happy (89%)
Lighting: Good Lighting
```

## 🔧 Configuration

### Model Configuration
- Face Detection: YuNet ONNX (confidence: 0.6, NMS: 0.3)
- Age/Gender: Vision Transformer ONNX with Caffe fallback
- Tracking: Centroid-based (max disappear: 30 frames, distance: 80px)
- Stabilization: 15-frame history with majority voting

### Performance Tuning
- Adjust `input_size` in FaceDetector for speed vs accuracy trade-off
- Modify `history_size` in predictors for stabilization sensitivity
- Change confidence thresholds for different environments

## 🛠️ Development

### Project Structure
```
ai-smart-face-analytics/
├── main.py                 # Main orchestrator and UI
├── detector.py            # Face detection and tracking
├── age_gender_model.py    # Age/gender prediction
├── emotion_model.py       # Emotion detection
├── face_recognition.py    # Face identification
├── mask_detection.py      # Mask classification
├── lighting_analysis.py   # Lighting quality analysis
├── utils.py              # Helper functions and classes
├── dashboard.py          # Analytics visualization
├── setup_models.py       # Model download utility
└── models/               # Pre-trained model files
```

### Adding New Features
1. Create new module in project root
2. Import in `AISmartFaceAnalytics` class
3. Add to analysis pipeline in `analyze_face()`
4. Update UI rendering in `draw_advanced_ui()`
5. Add to CSV logging in `DataLogger`

## 📊 Analytics Dashboard

The system includes a comprehensive analytics dashboard showing:

- **Gender Distribution**: Male/female ratios with percentages
- **Age Distribution**: Population age range breakdown
- **Emotion Analysis**: Mood distribution across detections
- **Mask Compliance**: Wearing rate statistics
- **Lighting Quality**: Environmental condition analysis
- **Confidence Trends**: Prediction accuracy over time
- **Detection Timeline**: Face counts across video frames

## 🔒 Privacy & Ethics

- **Local Processing**: All analysis happens on-device
- **No Data Transmission**: No cloud uploads or external APIs
- **Face Storage**: Optional embeddings storage with user consent
- **Transparent Logging**: Clear data collection with timestamps

## 📝 Requirements

- **Python**: 3.7+
- **OpenCV**: 4.5+ with DNN module
- **NumPy**: 1.19+
- **SciPy**: 1.5+
- **Matplotlib**: 3.3+ (for dashboard)
- **Pandas**: 1.2+ (for data analysis)
- **Seaborn**: 0.11+ (for visualizations)

## 🚀 Future Enhancements

- **Streamlit Web Dashboard**: Interactive web-based analytics
- **Advanced Face Recognition**: Dlib/FaceNet integration
- **Multi-Camera Support**: Synchronized multi-view analysis
- **Action Recognition**: Behavioral pattern detection
- **Demographic Insights**: Population statistics and trends
- **API Integration**: RESTful endpoints for external systems

## 📄 License

This project is open-source. Please cite appropriately for academic use.

## 🤝 Contributing

Contributions welcome! Please submit issues and pull requests for enhancements.

## 📬 Contact

Have questions or feedback? Let's connect!

**Author:** Rishabh Soni  
**GitHub:** [Rishabh Soni](https://github.com/rishabhsoni-ai)  
**Live Demo:** [https://ai-smart-face-analytics-123.streamlit.app/](https://ai-smart-face-analytics-123.streamlit.app/)

---

**Built with OpenCV, Deep Learning, and Computer Vision expertise**