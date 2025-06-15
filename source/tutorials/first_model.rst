Understanding I-Coach Models
============================

I-Coach uses multiple specialized AI models to provide accurate exercise recognition and form analysis. This tutorial explains how each model works and how they collaborate to deliver intelligent fitness feedback.

Model Architecture Overview
----------------------------

I-Coach implements a multi-model architecture with four main components:

1. **Exercise Classification Model** - Identifies the type of exercise
2. **Form Checking Models** - Analyzes exercise form quality  
3. **Chatbot Model** - Provides fitness guidance and answers questions
4. **Pose Estimation** - Extracts body landmarks using MediaPipe

Exercise Classification Model
-----------------------------

**Architecture**: LSTM (Long Short-Term Memory) Neural Network

**Input**: Sequence of 99-dimensional pose landmarks over time
**Output**: Classification among 5 exercise types

**How it Works**:

1. **Pose Extraction**: MediaPipe extracts 33 body landmarks (x, y, z coordinates)
2. **Sequence Building**: Creates temporal sequences of 100 frames
3. **Normalization**: Standardizes the data for consistent predictions
4. **LSTM Processing**: Analyzes movement patterns over time
5. **Classification**: Outputs exercise type with confidence score

.. code-block:: python

   # Example of pose landmark extraction
   def extract_pose_landmarks(landmarks):
       keypoints = []
       for lm in landmarks:
           keypoints.extend([lm.x, lm.y, lm.z])
       return keypoints  # Returns 99-dimensional vector

**Supported Exercises**:
- Bench Press
- Hammer Curl  
- Pull Up
- Push-up
- Squat

Form Checking Models
--------------------

I-Coach uses specialized attention-based models for form analysis:

**Hammer Curl Form Model**
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Type**: LSTM with Attention mechanism
- **Purpose**: Analyzes controlled movement and form quality
- **Output**: Binary classification (Good/Poor form)

**Push-up Form Model**
~~~~~~~~~~~~~~~~~~~~~~

- **Type**: LSTM with Attention mechanism  
- **Purpose**: Monitors body alignment and range of motion
- **Output**: Binary classification (Good/Poor form)

**Squat Form Model**
~~~~~~~~~~~~~~~~~~~~

- **Type**: Multi-class LSTM with Attention
- **Purpose**: Detailed squat form analysis
- **Output**: 5-class classification:
  - ``correct`` - Perfect form
  - ``bad_back_wrap`` - Forward lean issues
  - ``bad_back_round`` - Rounded back posture
  - ``bad_inner_thigh`` - Knee alignment problems  
  - ``shallow`` - Insufficient depth

**Attention Mechanism**:

.. code-block:: python

   class AdditiveAttention(Layer):
       def __init__(self, units):
           super(AdditiveAttention, self).__init__()
           self.units = units
           self.W1 = Dense(units)
           self.W2 = Dense(units)  
           self.V = Dense(1)
       
       def call(self, values, query):
           # Compute attention weights
           score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query)))
           attention_weights = tf.nn.softmax(score, axis=1)
           # Apply attention to create context vector
           context_vector = attention_weights * values
           return tf.reduce_sum(context_vector, axis=1)

Chatbot Model
-------------

**Architecture**: LSTM-based sequence-to-sequence model with intent classification

**Features**:
- Natural language understanding
- Intent recognition for fitness-related queries
- Context-aware responses
- Preprocessing with NLTK (stemming, stopword removal)

**Training Data**: Fitness-specific intents and responses covering:
- Exercise form guidance
- Workout planning
- Beginner advice
- Equipment recommendations
- Injury prevention

Data Flow Pipeline
------------------

1. **Video Input** → Camera feed or uploaded video
2. **Pose Detection** → MediaPipe extracts body landmarks  
3. **Sequence Building** → Temporal sequences for LSTM processing
4. **Exercise Classification** → Identify exercise type
5. **Form Analysis** → Specialized model analyzes technique
6. **Result Display** → Visual feedback with recommendations

Model Performance
-----------------

**Exercise Classification**:
- Accuracy: >90% on test data
- Real-time inference: ~30ms per frame
- Robust to different body types and camera angles

**Form Checking**:
- Squat form: 5-class accuracy >85%
- Binary form models: >90% accuracy
- Attention mechanism improves interpretability

**Chatbot**:
- Intent classification accuracy: >88%
- Response relevance: High for fitness domain
- Supports conversational context

Using the Models
----------------

**Real-time Prediction**:

.. code-block:: python

   # Initialize prediction system
   prediction_system = LivePredictionSystem()
   
   # Process video frame
   processed_frame = prediction_system.process_frame(frame)
   
   # Get current prediction
   exercise = st.session_state.current_prediction
   confidence = st.session_state.prediction_confidence

**Video Analysis**:

.. code-block:: python

   # Analyze uploaded video
   result, temp_path = analyze_video_file(uploaded_file)
   
   # Access results
   exercise_type = result['exercise']
   confidence = result['confidence'] 
   form_feedback = result.get('form_check', None)

Model Customization
-------------------

**Training Your Own Models**:

1. Collect exercise video data
2. Extract pose landmarks using MediaPipe
3. Label data for exercise types and form quality
4. Train LSTM models using TensorFlow/Keras
5. Implement attention mechanisms for form analysis
6. Validate on test data and tune hyperparameters

**Transfer Learning**:
- Start with pre-trained I-Coach models
- Fine-tune on your specific exercise data
- Adapt to new exercise types or form criteria

**Model Optimization**:
- Quantization for faster inference
- Model pruning to reduce size
- Hardware-specific optimizations (GPU/CPU)

Troubleshooting Models
----------------------

**Low Accuracy Issues**:
- Check camera positioning and lighting
- Ensure full body visibility in frame
- Verify model files are loaded correctly
- Consider retraining with domain-specific data

**Performance Issues**:
- Reduce sequence length for faster inference
- Use GPU acceleration if available
- Optimize video resolution and frame rate
- Consider model quantization

**Form Checking Limitations**:
- Models trained on specific exercise variations
- May not generalize to all body types
- Requires clear view of key body parts
- Best results with controlled environment

Next Steps
----------

- Experiment with different exercises and camera angles
- Try the video upload feature for detailed analysis  
- Explore the AI chatbot for personalized guidance
- Consider contributing training data for model improvements
