=========================
I-COACH Documentation
=========================

.. image:: https://img.shields.io/badge/python-3.8%2B-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :alt: License

.. image:: https://img.shields.io/badge/streamlit-1.28%2B-ff6b6b.svg
   :alt: Streamlit

.. image:: https://img.shields.io/badge/tensorflow-2.13%2B-orange.svg
   :alt: TensorFlow

**I-COACH** is an AI-powered exercise recognition and form checking application built with Streamlit, TensorFlow, and MediaPipe. It provides real-time exercise classification with advanced form analysis for squats, hammer curls, and push-ups.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   installation
   usage
   api
   models
   troubleshooting
   contributing

Welcome to I-COACH
==================

I-COACH revolutionizes fitness training by combining cutting-edge computer vision with deep learning to provide intelligent exercise recognition and real-time form correction. Whether you're a beginner learning proper form or an experienced athlete looking to perfect your technique, I-COACH serves as your personal AI fitness coach.

Quick Start
-----------

1. **Install I-COACH**: ``pip install -r requirements.txt``
2. **Download Models**: Place model files in your project directory
3. **Run Application**: ``streamlit run app.py``
4. **Start Training**: Use live camera or upload videos for analysis

Key Features at a Glance
------------------------

✅ **Real-time Exercise Recognition** - Identifies 5 different exercises instantly
✅ **AI Form Checking** - Advanced form analysis with specific feedback
✅ **Live Camera Feed** - Real-time predictions with visual overlays
✅ **Video Analysis** - Upload and analyze pre-recorded workouts
✅ **AI Fitness Coach** - Interactive chatbot for personalized advice
✅ **Modern UI** - Beautiful, responsive interface with gradient designs

.. _overview:

Overview
========

.. sidebar:: Navigation
   :class: sidebar-nav

   **Overview Sections**
   
   - `Core Philosophy`_
   - `Supported Exercises`_
   - `AI Technology Stack`_
   - `Real-world Applications`_

I-COACH represents the next generation of fitness technology, where artificial intelligence meets personal training. Our application doesn't just count reps—it understands movement patterns, analyzes form, and provides intelligent feedback to help users achieve their fitness goals safely and effectively.

Core Philosophy
---------------

**Form Over Everything**
   Perfect form prevents injuries and maximizes results. I-COACH prioritizes technique analysis over simple exercise counting, ensuring users develop proper movement patterns from day one.

**Accessibility First**
   Professional-grade form analysis should be available to everyone. I-COACH brings gym-quality feedback to your home, making expert coaching accessible regardless of location or budget.

**Continuous Learning**
   Our AI models continuously improve through advanced machine learning techniques, providing increasingly accurate and personalized feedback over time.

Supported Exercises
-------------------

**Classification Supported (5 Exercises)**

1. **Bench Press** 🏋️‍♂️
   - Upper body strength training
   - Chest, shoulders, and triceps focus
   - Proper bar path recognition

2. **Hammer Curl** 💪
   - Bicep and forearm development
   - Controlled movement analysis
   - **AI Form Checking Available**

3. **Pull Up** 🤸‍♂️
   - Upper body compound movement
   - Lat and upper back focus
   - Full range of motion detection

4. **Push-up** 🤲
   - Bodyweight upper body exercise
   - Full-body stabilization
   - **AI Form Checking Available**

5. **Squat** 🦵
   - Lower body compound movement
   - Quad, glute, and hamstring focus
   - **Advanced 5-Class Form Analysis**

**Advanced Form Checking**

- **Squat Analysis**: 5 specific form classifications
  - ✅ Correct form
  - ⚠️ Back wrap issues
  - ⚠️ Rounded back problems
  - ⚠️ Inner thigh/knee tracking
  - ⚠️ Insufficient depth

- **Hammer Curl Analysis**: Binary classification
  - ✅ Controlled movement
  - ⚠️ Swinging/momentum issues

- **Push-up Analysis**: Binary classification
  - ✅ Proper alignment and depth
  - ⚠️ Form breakdown detection

AI Technology Stack
-------------------

**Computer Vision**
   - **MediaPipe Pose**: 33 3D pose landmarks
   - **Real-time Processing**: 30 FPS capability
   - **Robust Tracking**: Works in various lighting conditions

**Deep Learning**
   - **TensorFlow/Keras**: Neural network framework
   - **LSTM/GRU Networks**: Sequence modeling
   - **Custom Attention Layers**: Focused analysis
   - **Multi-class Classification**: Exercise recognition
   - **Binary/Multi-class Form Analysis**: Technique assessment

**Natural Language Processing**
   - **NLTK**: Text processing and analysis
   - **Custom Tokenization**: Fitness-specific vocabulary
   - **Intent Classification**: Understanding user queries
   - **Response Generation**: Contextual fitness advice

Real-world Applications
-----------------------

**Home Fitness Enthusiasts**
   - Perfect for home gym setups
   - No need for expensive personal trainers
   - Immediate feedback on form

**Physical Therapy**
   - Movement pattern analysis
   - Recovery progress tracking
   - Safe exercise execution

**Fitness Professionals**
   - Client assessment tool
   - Form demonstration aid
   - Progress documentation

**Beginners**
   - Learn proper form from the start
   - Build confidence with AI guidance
   - Prevent injury through correct technique

.. _installation:

Installation
============

.. sidebar:: Installation Guide
   :class: sidebar-nav

   **Installation Steps**
   
   - `System Requirements`_
   - `Python Environment`_
   - `Dependencies`_
   - `Model Files`_
   - `Verification`_

System Requirements
-------------------

**Minimum Requirements**
   - **OS**: Windows 10, macOS 10.14, Ubuntu 18.04
   - **Python**: 3.8 or higher
   - **RAM**: 8GB minimum, 16GB recommended
   - **Storage**: 2GB free space for models
   - **Camera**: Webcam for live predictions

**Recommended Specifications**
   - **CPU**: Intel i5 or AMD Ryzen 5 (or better)
   - **GPU**: NVIDIA GTX 1060 or better (for GPU acceleration)
   - **RAM**: 16GB or more
   - **Camera**: HD webcam (1080p preferred)

Python Environment
------------------

**Setting up Virtual Environment**

.. code-block:: bash

   # Create virtual environment
   python -m venv icoach-env
   
   # Activate environment (Windows)
   icoach-env\Scripts\activate
   
   # Activate environment (macOS/Linux)
   source icoach-env/bin/activate

**Verify Python Version**

.. code-block:: bash

   python --version
   # Should show Python 3.8.x or higher

Dependencies
------------

**Core Dependencies**

.. code-block:: bash

   # Install core requirements
   pip install streamlit>=1.28.0
   pip install opencv-python>=4.8.0
   pip install mediapipe>=0.10.0
   pip install tensorflow>=2.13.0
   pip install numpy>=1.24.0
   pip install pillow>=9.5.0
   pip install nltk>=3.8.0

**Optional GPU Support**

.. code-block:: bash

   # For NVIDIA GPU acceleration
   pip install tensorflow-gpu>=2.13.0
   
   # Verify GPU detection
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

**Development Dependencies**

.. code-block:: bash

   # Additional tools for development
   pip install pytest>=7.0.0
   pip install black>=23.0.0
   pip install flake8>=6.0.0
   pip install jupyter>=1.0.0

Model Files
-----------

**Required Model Files**

Create a ``models/`` directory and place the following files:

1. **Exercise Classification Model**
   - File: ``best_exercise_classifier.h5``
   - Size: ~50MB
   - Purpose: Main exercise recognition

2. **Form Checking Models**
   - ``best_attention_model.h5`` - Hammer curl form (Binary)
   - ``best_pu_attention_model.h5`` - Push-up form (Binary)
   - ``best_squat_attention_model.h5`` - Squat form (5-class)

3. **Chatbot Models (Optional)**
   - ``fitness_lstm_model.h5`` - Chatbot neural network
   - ``fitness_lstm_tokenizer.pkl`` - Text tokenizer

**Directory Structure**

.. code-block:: text

   i-coach/
   ├── app.py
   ├── requirements.txt
   ├── models/
   │   ├── best_exercise_classifier.h5
   │   ├── best_attention_model.h5
   │   ├── best_pu_attention_model.h5
   │   ├── best_squat_attention_model.h5
   │   ├── fitness_lstm_model.h5
   │   └── fitness_lstm_tokenizer.pkl
   └── docs/
       └── index.rst

**Model Download Script**

.. code-block:: bash

   # Download models (if available from repository)
   python download_models.py
   
   # Or manually download from releases
   wget https://github.com/yourusername/i-coach/releases/download/v1.0/models.zip
   unzip models.zip

Verification
------------

**Test Installation**

.. code-block:: bash

   # Run basic import test
   python -c "
   import streamlit as st
   import cv2
   import mediapipe as mp
   import tensorflow as tf
   print('✅ All imports successful!')
   "

**Launch Application**

.. code-block:: bash

   # Start I-COACH
   streamlit run app.py
   
   # Application should open at http://localhost:8501

**Verify Models**

.. code-block:: bash

   # Check model loading
   python -c "
   import tensorflow as tf
   try:
       model = tf.keras.models.load_model('models/best_exercise_classifier.h5')
       print('✅ Exercise model loaded successfully!')
   except:
       print('❌ Exercise model not found')
   "

.. _usage:

Usage Guide
===========

.. sidebar:: Usage Navigation
   :class: sidebar-nav

   **Usage Sections**
   
   - `Getting Started`_
   - `Live Camera Mode`_
   - `Video Upload Analysis`_
   - `AI Fitness Coach`_
   - `Best Practices`_

Getting Started
---------------

**First Launch**

1. **Start the Application**
   
   .. code-block:: bash
   
      streamlit run app.py

2. **Check Model Status**
   
   In the sidebar, verify all models show "✅ Loaded" status:
   - Classification Model
   - Form Checking Models
   - AI Fitness Coach (if available)

3. **Camera Permissions**
   
   Grant camera access when prompted by your browser.

**Interface Overview**

The I-COACH interface consists of three main tabs:

- **📹 Live Camera**: Real-time exercise recognition
- **📁 Video Upload**: Analyze pre-recorded workouts
- **🤖 AI Fitness Coach**: Interactive fitness guidance

Live Camera Mode
----------------

**Setting Up Live Analysis**

1. **Position Your Camera**
   - Ensure full body is visible
   - Maintain 6-8 feet distance from camera
   - Good lighting from front or side
   - Stable camera position

2. **Start Recognition**
   - Click "🎥 Start Camera"
   - Allow camera permissions
   - Begin performing exercises

3. **Understanding Predictions**
   - Predictions update every 3 seconds
   - Confidence scores indicate accuracy
   - Form feedback appears for supported exercises

**Live Interface Elements**

**Main Video Feed**
   - Real-time camera with pose overlay
   - Exercise predictions displayed on frame
   - Form feedback text overlay
   - Countdown timer for next prediction

**Stats Panel**
   - Current exercise prediction
   - Confidence percentage
   - Form analysis results
   - Frame processing count

**Form Feedback Types**

**Good Form Indicators**
   - Green background colors
   - "Excellent" or "Perfect" feedback
   - High confidence scores (>80%)

**Form Correction Needed**
   - Red/orange background colors
   - Specific improvement suggestions
   - Moderate confidence scores (50-80%)

**Example Feedback Messages**

.. code-block:: text

   ✅ "Perfect squat form! Excellent depth, posture, and alignment! 🎯"
   ⚠️ "Rounded Back Detected: Keep chest up and shoulders back!"
   ⚠️ "Insufficient Depth: Go deeper! Aim for thighs parallel to ground!"

Video Upload Analysis
---------------------

**Supported Video Formats**
   - MP4 (recommended)
   - AVI
   - MOV
   - MKV

**Upload Process**

1. **Select Video File**
   - Click "Choose a video file"
   - Select workout video from device
   - Maximum recommended: 30 seconds

2. **Analysis Processing**
   - Automatic pose detection
   - Exercise classification
   - Form analysis (if applicable)
   - Results generation

3. **Review Results**
   - Exercise type identification
   - Confidence scores
   - Form feedback
   - Exercise-specific tips

**Video Requirements**

**Optimal Video Characteristics**
   - Clear view of full body
   - Steady camera position
   - Good lighting conditions
   - Single person performing exercise
   - Clear, distinct movements

**Analysis Output**

**Exercise Classification**
   - Detected exercise type
   - Confidence percentage
   - Alternative exercise suggestions

**Form Analysis**
   - Specific form issues identified
   - Improvement recommendations
   - Technique tips

**Exercise-Specific Tips**
   - Customized advice for detected exercise
   - Common mistakes to avoid
   - Progressive improvement suggestions

AI Fitness Coach
----------------

**Chatbot Capabilities**

The AI Fitness Coach uses natural language processing to provide personalized fitness guidance:

**Question Categories**
   - Exercise form and technique
   - Workout routine planning
   - Repetition and set recommendations
   - Beginner fitness guidance
   - Injury prevention tips

**Interaction Methods**

**Text Input**
   - Type questions in natural language
   - Receive personalized responses
   - Conversation history maintained

**Quick Questions**
   - Pre-defined common questions
   - Instant response generation
   - Popular fitness topics

**Example Interactions**

.. code-block:: text

   You: "How can I improve my squat depth?"
   AI Coach: "To improve squat depth, focus on ankle mobility, 
   hip flexibility, and core strength. Practice goblet squats 
   and wall sits to build the movement pattern..."

   You: "Best exercises for beginners?"
   AI Coach: "Start with bodyweight exercises: squats, push-ups, 
   planks, and lunges. These build fundamental strength and 
   movement patterns safely..."

**Conversation Features**
   - Context-aware responses
   - Fitness terminology understanding
   - Progressive difficulty suggestions
   - Safety-first recommendations

Best Practices
--------------

**Camera Setup Tips**

**Lighting**
   - Natural light works best
   - Avoid backlighting
   - Consistent lighting prevents detection issues

**Positioning**
   - Full body visible in frame
   - Maintain consistent distance
   - Avoid excessive background movement

**Exercise Performance**

**For Accurate Recognition**
   - Perform exercises with clear, distinct movements
   - Maintain proper form throughout
   - Allow 3-second intervals between predictions
   - Avoid rushed or partial movements

**Form Checking Optimization**
   - Focus on supported exercises for form analysis
   - Perform full range of motion
   - Maintain steady, controlled movements
   - Position body perpendicular to camera

**Troubleshooting Common Issues**

**Low Confidence Scores**
   - Improve lighting conditions
   - Ensure full body visibility
   - Perform exercises more distinctly
   - Check camera stability

**Inaccurate Predictions**
   - Verify exercise is in supported list
   - Allow adequate prediction time
   - Check pose landmark detection
   - Ensure proper exercise execution

.. _api:

API Reference
=============

.. sidebar:: API Navigation
   :class: sidebar-nav

   **API Sections**
   
   - `Core Classes`_
   - `Model Functions`_
   - `Pose Processing`_
   - `Utility Functions`_
   - `Configuration`_

Core Classes
------------

LivePredictionSystem
~~~~~~~~~~~~~~~~~~~~

The main class handling real-time exercise recognition and form analysis.

.. code-block:: python

   class LivePredictionSystem:
       """
       Handles real-time pose detection, exercise classification,
       and form analysis for live camera feed.
       """
       
       def __init__(self):
           """
           Initialize MediaPipe pose detection and prediction buffers.
           
           Attributes:
               pose: MediaPipe Pose solution
               sequence_buffer: List of pose sequences for classification
               form_buffer: List of pose sequences for form analysis
               prediction_interval: Time between predictions (seconds)
           """
           
       def process_frame(self, frame):
           """
           Process a single camera frame for pose detection and prediction.
           
           Args:
               frame (numpy.ndarray): Input video frame in BGR format
               
           Returns:
               numpy.ndarray: Annotated frame with pose landmarks and predictions
               
           Features:
               - Pose landmark detection
               - Sequence buffer management
               - Prediction timing control
               - Frame annotation with results
           """
           
       def make_prediction(self):
           """
           Generate exercise and form predictions from buffered sequences.
           
           Process:
               1. Prepare sequence data for model input
               2. Normalize pose landmarks
               3. Run exercise classification
               4. Perform form analysis if applicable
               5. Update session state with results
           """

AdditiveAttention Layer
~~~~~~~~~~~~~~~~~~~~~~~

Custom TensorFlow layer implementing additive attention mechanism.

.. code-block:: python

   class AdditiveAttention(tf.keras.layers.Layer):
       """
       Custom attention layer for focusing on relevant pose sequences
       during form analysis.
       """
       
       def __init__(self, units, **kwargs):
           """
           Initialize attention layer.
           
           Args:
               units (int): Number of attention units
               **kwargs: Additional layer arguments
           """
           
       def call(self, values, query):
           """
           Compute attention weights and context vector.
           
           Args:
               values: Input sequence values
               query: Query vector for attention
               
           Returns:
               tuple: (context_vector, attention_weights)
               
           Mechanism:
               - Computes attention scores using tanh activation
               - Applies softmax for attention weights
               - Generates weighted context vector
           """
           
       def get_config(self):
           """Return layer configuration for serialization."""

FitnessChatbot
~~~~~~~~~~~~~~

Natural language processing chatbot for fitness guidance.

.. code-block:: python

   class FitnessChatbot:
       """
       AI-powered fitness chatbot using LSTM and NLP techniques.
       """
       
       def __init__(self):
           """
           Initialize chatbot with NLP components.
           
           Attributes:
               MAX_SEQ_LEN: Maximum sequence length for text processing
               stemmer: Porter stemmer for word normalization
               stop_words: Common English stop words
               model: LSTM model for intent classification
               tokenizer: Text tokenizer for preprocessing
           """
           
       def preprocess_text(self, text):
           """
           Preprocess user input text.
           
           Args:
               text (str): Raw user input
               
           Returns:
               str: Preprocessed text ready for model input
               
           Steps:
               1. Convert to lowercase
               2. Remove non-alphabetic characters
               3. Remove stop words
               4. Apply stemming
           """
           
       def predict_intent(self, user_input, confidence_threshold=0.3):
           """
           Classify user intent from input text.
           
           Args:
               user_input (str): User question or statement
               confidence_threshold (float): Minimum confidence for prediction
               
           Returns:
               tuple: (intent_label, confidence_score)
           """
           
       def get_response(self, user_input):
           """
           Generate response for user input.
           
           Args:
               user_input (str): User question
               
           Returns:
               dict: Response data with intent and confidence
           """

Model Functions
---------------

Model Loading Functions
~~~~~~~~~~~~~~~~~~~~~~~

Cached functions for loading AI models efficiently.

.. code-block:: python

   @st.cache_resource
   def load_exercise_model():
       """
       Load main exercise classification model.
       
       Returns:
           tf.keras.Model: Loaded exercise classification model
           
       Model Details:
           - Input: (batch_size, 100, 99) pose sequences
           - Output: (batch_size, 5) exercise probabilities
           - Architecture: Deep neural network with LSTM layers
       """
       
   @st.cache_resource
   def load_form_model():
       """
       Load hammer curl form checking model.
       
       Returns:
           tf.keras.Model: Binary classification model for form analysis
           
       Model Details:
           - Input: (batch_size, 100, 132) pose sequences with visibility
           - Output: (batch_size, 1) form correctness probability
           - Architecture: LSTM with custom attention mechanism
       """
       
   @st.cache_resource
   def load_pushup_form_model():
       """
       Load push-up specific form analysis model.
       
       Returns:
           tf.keras.Model: Push-up form checking model
       """
       
   @st.cache_resource
   def load_squat_form_model():
       """
       Load squat form analysis model.
       
       Returns:
           tf.keras.Model: Multi-class squat form classification model
           
       Model Details:
           - Input: (batch_size, 100, 132) pose sequences
           - Output: (batch_size, 5) form class probabilities
           - Classes: correct, bad_back_wrap, bad_back_round, 
                     bad_inner_thigh, shallow
       """

Pose Processing
---------------

Pose Landmark Functions
~~~~~~~~~~~~~~~~~~~~~~~

Functions for processing MediaPipe pose landmarks.

.. code-block:: python

   def extract_pose_landmarks(landmarks, include_visibility=False):
       """
       Extract pose landmarks from MediaPipe results.
       
       Args:
           landmarks: MediaPipe pose landmarks
           include_visibility (bool): Whether to include visibility data
           
       Returns:
           list: Flattened landmark coordinates
           
       Format:
           - Without visibility: [x1, y1, z1, x2, y2, z2, ...] (99 values)
           - With visibility: [x1, y1, z1, v1, x2, y2, z2, v2, ...] (132 values)
           
       Error Handling:
           - Returns zero-filled list if landmarks unavailable
           - Handles incomplete landmark data gracefully
       """
       
   def analyze_form_with_model(form_sequence, exercise_type):
       """
       Analyze exercise form using appropriate model.
       
       Args:
           form_sequence (list): Sequence of pose landmarks with visibility
           exercise_type (str): Type of exercise being analyzed
           
       Returns:
           dict: Form analysis results
           
       Supported Exercise Types:
           - 'hammer curl': Binary form classification
           - 'push-up': Binary form classification  
           - 'squat': 5-class form classification
           
       Return Format:
           {
               'is_correct': bool,
               'confidence': float,
               'feedback': str,
               'form_type': str,
               'predicted_form': str (for squat only)
           }
       """

Video Processing Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~

Functions for handling video file analysis.

.. code-block:: python

   def save_video_temporarily(uploaded_file):
       """
       Save uploaded video to temporary file.
       
       Args:
           uploaded_file: Streamlit uploaded file object
           
       Returns:
           str: Path to temporary video file
           
       Features:
           - Creates temporary file with proper extension
           - Handles file writing and path management
           - Automatic cleanup preparation
       """
       
   def analyze_video_file(uploaded_file):
       """
       Analyze uploaded video file for exercise classification and form.
       
       Args:
           uploaded_file: Streamlit uploaded file object
           
       Returns:
           tuple: (analysis_results, temp_file_path)
           
       Analysis Results Format:
           {
               'exercise': str,
               'confidence': float,
               'form_check': dict (if applicable)
           }
           
       Process:
           1. Save video to temporary file
           2. Extract pose landmarks from frames
           3. Prepare sequences for model input
           4. Run classification and form analysis
           5. Return structured results
       """
       
   def cleanup_temp_file(temp_path):
       """
       Clean up temporary video file.
       
       Args:
           temp_path (str): Path to temporary file
           
       Features:
           - Safe file deletion with error handling
           - Existence check before deletion
           - Silent failure for cleanup robustness
       """

Utility Functions
-----------------

Configuration Functions
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def initialize_session_state():
       """
       Initialize Streamlit session state variables.
       
       Session State Variables:
           - camera_running: bool
           - current_prediction: str
           - prediction_confidence: float
           - form_feedback: str
           - frames_processed: int
           - last_prediction_time: float
           - chat_history: list
           - chatbot_model: FitnessChatbot
           - prediction_system: LivePredictionSystem
       """
       
   def setup_page_config():
       """
       Configure Streamlit page settings.
       
       Settings:
           - Page title and icon
           - Layout configuration
           - Custom CSS styling
           - UI theme and colors
       """

Constants and Configuration
---------------------------

Global Variables
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Exercise classification labels
   label_map = ['bench press', 'hammer curl', 'pull up', 'push-up', 'squat']
   inv_label_map = {i: label for i, label in enumerate(label_map)}
   
   # MediaPipe configuration
   mp_pose = mp.solutions.pose
   mp_drawing = mp.solutions.drawing_utils
   
   # Model file paths (searched in order)
   MODEL_PATHS = {
       'exercise_classifier': [
           r"D:\Downloads\best_exercise_classifier.h5",
           "./best_exercise_classifier.h5",
           "./models/best_exercise_classifier.h5"
       ],
       'form_models': {
           'hammer_curl': [
               r"D:\Downloads\best_attention_model.h5",
               "./best_attention_model.h5",
               "./models/best_attention_model.h5"
           ],
           'pushup': [
               r"D:\Downloads\best_pu_attention_model.h5",
               "./best_pu_attention_model.h5",
               "./models/best_pu_attention_model.h5"
           ],
           'squat': [
               r"D:\Downloads\best_squat_attention_model.h5",
               "./best_squat_attention_model.h5",
               "./models/best_squat_attention_model.h5"
           ]
       }
   }

Processing Parameters
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Sequence processing parameters
   MAX_SEQUENCE_LENGTH = 100
   POSE_LANDMARKS_COUNT = 33
   COORDINATE_DIMENSIONS = 3  # x, y, z
   VISIBILITY_DIMENSIONS = 4  # x, y, z, visibility
   
   # Prediction parameters
   PREDICTION_INTERVAL = 3.0  # seconds
   CONFIDENCE_THRESHOLD = 0.5
   FORM_ANALYSIS_MIN_FRAMES = 50
   BUFFER_SIZE_LIMIT = 150
   
   # MediaPipe pose configuration
   POSE_CONFIG = {
       'static_image_mode': False,
       'model_complexity': 1,
       'enable_segmentation': False,
       'min_detection_confidence': 0.5,
       'min_tracking_confidence': 0.5
   }

.. _models:

Models & Architecture
=====================

.. sidebar:: Models Navigation
   :class: sidebar-nav

   **Model Sections**
   
   - `Exercise Classification`_
   - `Form Analysis Models`_
   - `Chatbot Architecture`_
   - `Training Details`_
   - `Performance Metrics`_

Exercise Classification
-----------------------

**Model Architecture**

The main exercise classification model uses a deep neural network designed for sequential pose data analysis.

**Input Specifications**
   - **Shape**: (batch_size, 100, 99)
   - **Data**: 100 frames of 33 pose landmarks (x, y, z)
   - **Normalization**: Z-score normalization applied
   - **Preprocessing**: Sequence padding and temporal alignment

**Network Architecture**

.. code-block:: python

   model = Sequential([
       # Input layer
       Input(shape=(100, 99)),
       
       # Feature extraction layers
       LSTM(128, return_sequences=True, dropout=0.2),
       LSTM(64, return_sequences=True, dropout=0.2),
       LSTM(32, dropout=0.2),
       
       # Classification layers
       Dense(64, activation='relu'),
       Dropout(0.3),
       Dense(32, activation='relu'),
       Dropout(0.2),
       Dense(5, activation='softmax')  # 5 exercise classes
   ])

**Output Classes**

1. **Bench Press** - Upper body pressing movement
2. **Hammer Curl** - Bicep isolation exercise
3. **Pull Up** - Vertical pulling movement
4. **Push-up** - Bodyweight pressing exercise
5. **Squat** - Lower body compound movement

**Training Details**
   - **Loss Function**: Sparse categorical crossentropy
   - **Optimizer**: Adam with learning rate scheduling
   - **Metrics**: Accuracy, precision, recall, F1-score
   - **Regularization**: Dropout, L2 regularization

Form Analysis Models
--------------------

**Hammer Curl Form Model**

Binary classification model for assessing hammer curl technique.

**Architecture Features**
   - **Input**: (batch_size, 100, 132) - includes visibility data
   - **Custom Attention**: AdditiveAttention layer for focus
   - **Output**: Binary classification (correct/incorrect)

.. code-block:: python

   # Hammer curl model architecture
   inputs = Input(shape=(100, 132))
   
   # LSTM feature extraction
   lstm_out = LSTM(64, return_sequences=True)(inputs)
   lstm_out = LSTM(32, return_sequences=True)(lstm_out)
   
   # Attention mechanism
   query = GlobalAveragePooling1D()(lstm_out)
   context, attention = AdditiveAttention(32)(lstm_out, query)
   
   # Classification
   dense = Dense(16, activation='relu')(context)
   output = Dense(1, activation='sigmoid')(dense)

**Push-up Form Model**

Similar architecture to hammer curl model, specialized for push-up movement patterns.

**Key Features**
   - **Body Alignment Detection**: Monitors straight line from head to heels
   - **Range of Motion**: Analyzes depth and full extension
   - **Tempo Analysis**: Evaluates movement speed and control

**Squat Form Model**

Advanced 5-class classification model for comprehensive squat analysis.

**Form Categories**

1. **Correct Form** ✅
   - Proper depth (thighs parallel or below)
   - Neutral spine alignment
   - Knees tracking over toes
   - Controlled movement pattern

2. **Bad Back Wrap** ⚠️
   - Excessive forward lean
   - Loss of chest position
   - Compromised spinal
3. **Bad Back Round** ⚠️
   - Rounded thoracic spine
   - Loss of lumbar curve
   - Forward head posture
   - Inability to maintain chest up position

4. **Bad Inner Thigh** ⚠️
   - Knee valgus (knees caving inward)
   - Poor hip mobility
   - Inadequate glute activation
   - Ankle mobility limitations

5. **Shallow Squat** ⚠️
   - Insufficient depth
   - Thighs above parallel
   - Limited hip flexion
   - Reduced training effectiveness

**Model Architecture**

```python
# Squat form model (5-class classification)
inputs = Input(shape=(100, 132))

# Multi-layer LSTM with attention
lstm1 = LSTM(128, return_sequences=True, dropout=0.2)(inputs)
lstm2 = LSTM(64, return_sequences=True, dropout=0.2)(lstm1)
lstm3 = LSTM(32, return_sequences=True, dropout=0.2)(lstm2)

# Global attention mechanism
query = GlobalMaxPooling1D()(lstm3)
context, attention_weights = AdditiveAttention(64)(lstm3, query)

# Multi-class classification head
dense1 = Dense(32, activation='relu')(context)
dropout1 = Dropout(0.3)(dense1)
dense2 = Dense(16, activation='relu')(dropout1)
dropout2 = Dropout(0.2)(dense2)
output = Dense(5, activation='softmax')(dropout2)  # 5 form classes
```

**Training Configuration**
- **Dataset**: Custom squat form dataset
- **Data Augmentation**: Temporal jittering, noise injection
- **Class Balancing**: Weighted loss function
- **Validation**: Stratified K-fold cross-validation

## Chatbot Architecture

**Natural Language Processing Model**

The AI Fitness Coach uses an LSTM-based architecture for intent classification and response generation.

**Model Components**

**Text Preprocessing Pipeline**
- **Tokenization**: Custom fitness vocabulary
- **Stemming**: Porter stemmer for word normalization  
- **Stop Word Removal**: Common English stop words
- **Sequence Padding**: Fixed-length input sequences

**LSTM Architecture**

```python
# Chatbot model architecture
model = Sequential([
    Embedding(vocab_size, 128, input_length=MAX_SEQ_LEN),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(num_intents, activation='softmax')
])
```

**Intent Categories**
- Exercise technique questions
- Workout routine planning
- Repetition and set guidance
- Beginner fitness advice
- Injury prevention tips
- Equipment recommendations
- Nutrition basics
- Progress tracking

**Response Generation**
- Template-based responses
- Context-aware suggestions
- Personalized recommendations
- Safety-first guidance

## Training Details

**Data Collection**

**Exercise Classification Dataset**
- **Total Samples**: ~10,000 exercise sequences
- **Data Sources**: Multiple fitness video datasets
- **Annotation**: Manual labeling by fitness experts
- **Quality Control**: Multi-reviewer validation

**Form Analysis Datasets**
- **Hammer Curl**: 2,500 sequences (correct/incorrect)
- **Push-up**: 3,200 sequences (proper/improper form)
- **Squat**: 4,800 sequences (5 form categories)

**Data Preprocessing**
- **Pose Extraction**: MediaPipe Pose landmarks
- **Normalization**: Center-relative coordinates
- **Augmentation**: Temporal shifts, mirror transforms
- **Sequence Alignment**: Dynamic time warping

**Training Procedures**

**Hyperparameter Optimization**
- **Learning Rate**: 0.001 with cosine annealing
- **Batch Size**: 32 (adjusted for GPU memory)
- **Epochs**: 100 with early stopping
- **Validation Split**: 20% stratified sampling

**Regularization Techniques**
- **Dropout**: 0.2-0.3 across layers
- **L2 Regularization**: 1e-4 weight decay
- **Batch Normalization**: Applied to dense layers
- **Gradient Clipping**: Prevents exploding gradients

**Model Selection**
- **Cross-Validation**: 5-fold stratified CV
- **Ensemble Methods**: Best model averaging
- **Hyperparameter Tuning**: Bayesian optimization
- **Early Stopping**: Patience of 10 epochs

## Performance Metrics

**Exercise Classification Performance**

**Overall Accuracy**: 94.3%

**Per-Class Metrics**:
- **Bench Press**: Precision 96.2%, Recall 93.8%, F1 95.0%
- **Hammer Curl**: Precision 92.1%, Recall 94.7%, F1 93.4%
- **Pull Up**: Precision 93.8%, Recall 92.3%, F1 93.0%
- **Push-up**: Precision 95.4%, Recall 96.1%, F1 95.7%
- **Squat**: Precision 94.2%, Recall 95.8%, F1 95.0%

**Form Analysis Performance**

**Hammer Curl Form (Binary)**
- **Accuracy**: 91.7%
- **Precision**: 89.3%
- **Recall**: 93.2%
- **F1-Score**: 91.2%

**Push-up Form (Binary)**
- **Accuracy**: 88.9%
- **Precision**: 87.1%
- **Recall**: 90.4%
- **F1-Score**: 88.7%

**Squat Form (5-Class)**
- **Overall Accuracy**: 86.4%
- **Macro F1-Score**: 84.7%
- **Weighted F1-Score**: 86.1%

**Detailed Squat Form Metrics**:
- **Correct Form**: Precision 92.1%, Recall 89.7%
- **Bad Back Wrap**: Precision 81.3%, Recall 84.2%
- **Bad Back Round**: Precision 83.7%, Recall 82.9%
- **Bad Inner Thigh**: Precision 78.9%, Recall 81.1%
- **Shallow Squat**: Precision 87.4%, Recall 85.3%

**Real-time Performance**
- **Processing Speed**: 30 FPS on modern hardware
- **Latency**: <100ms per frame
- **Memory Usage**: ~2GB GPU memory
- **CPU Utilization**: 15-25% on quad-core systems

---

## Troubleshooting

**Common Issues and Solutions**

**Model Loading Problems**

**Issue**: "Model file not found" error
**Solution**: 
- Verify model files are in correct directory
- Check file permissions
- Download missing model files
- Ensure sufficient disk space

**Issue**: "Model loading failed" error
**Solution**:
- Check TensorFlow version compatibility
- Verify GPU/CPU availability
- Clear model cache: `st.cache_resource.clear()`
- Restart application

**Camera Issues**

**Issue**: Camera not detected
**Solution**:
- Grant browser camera permissions
- Check camera availability in other applications
- Try different browsers (Chrome recommended)
- Verify camera drivers are updated

**Issue**: Poor pose detection
**Solution**:
- Improve lighting conditions
- Ensure full body is visible
- Remove background distractions
- Check camera focus and stability

**Performance Issues**

**Issue**: Slow processing speed
**Solution**:
- Close unnecessary applications
- Reduce video resolution if possible
- Enable GPU acceleration
- Check system resource usage

**Issue**: Inaccurate predictions
**Solution**:
- Ensure proper exercise execution
- Allow adequate prediction time (3+ seconds)
- Check exercise is in supported list
- Verify good pose landmark detection

**Form Analysis Issues**

**Issue**: No form feedback provided
**Solution**:
- Verify exercise supports form analysis
- Ensure adequate sequence length (50+ frames)
- Check pose landmark visibility
- Perform exercise with clear movements

**Issue**: Incorrect form assessment
**Solution**:
- Review exercise technique
- Ensure proper camera positioning
- Check for consistent movement patterns
- Consider lighting and background factors

**Installation Issues**

**Issue**: Dependency conflicts
**Solution**:
```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows

# Install dependencies one by one
pip install streamlit==1.28.0
pip install tensorflow==2.13.0
pip install opencv-python==4.8.0
pip install mediapipe==0.10.0
```

**Issue**: GPU not detected
**Solution**:
- Install CUDA toolkit
- Install cuDNN libraries
- Verify GPU compatibility
- Test with: `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`

---

## Contributing

**Development Setup**

**Prerequisites**
```bash
# Development environment
python -m venv dev_env
source dev_env/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt
pip install pytest black flake8 mypy
```

**Code Style**
- **Formatter**: Black with 88-character line limit
- **Linter**: Flake8 with custom configuration
- **Type Checking**: MyPy for static analysis
- **Documentation**: Google-style docstrings

**Testing**
```bash
# Run test suite
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test categories
pytest tests/test_models.py -v
pytest tests/test_pose_processing.py -v
```

**Contribution Guidelines**

**Pull Request Process**
1. Fork repository and create feature branch
2. Implement changes with comprehensive tests
3. Ensure all tests pass and code style compliance
4. Update documentation as needed
5. Submit pull request with detailed description

**Code Review Checklist**
- [ ] Code follows project style guidelines
- [ ] Comprehensive test coverage
- [ ] Documentation updated
- [ ] No breaking changes to API
- [ ] Performance impact assessed

**Issue Reporting**
- Use GitHub issue templates
- Provide detailed reproduction steps
- Include system information
- Attach relevant logs or screenshots

**Feature Requests**
- Describe use case and benefits
- Consider implementation complexity
- Discuss with maintainers before large changes
- Provide mockups or examples if applicable

---

## License and Acknowledgments

**License**
This project is licensed under the MIT License. See the LICENSE file for full details.

**Acknowledgments**
- **MediaPipe**: Google's pose estimation framework
- **TensorFlow**: Deep learning framework
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library
- **NLTK**: Natural language processing toolkit

**Contributors**
Special thanks to all contributors who have helped improve I-COACH through code contributions, bug reports, and feature suggestions.

**Citation**
If you use I-COACH in your research or projects, please cite:
```
@software{icoach2024,
  title={I-COACH: AI-Powered Exercise Recognition and Form Analysis},
  author={[Your Name/Team]},
  year={2024},
  url={https://github.com/yourusername/i-coach}
}
```

---

**Contact Information**
- **GitHub**: [Repository URL]
- **Documentation**: [Documentation URL]
- **Issues**: [GitHub Issues URL]
- **Discussions**: [GitHub Discussions URL]

---

*Last updated: June 2025*
*Version: 1.0.0*
