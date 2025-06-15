Examples
========

Practical examples and use cases for I-Coach.

.. toctree::
   :maxdepth: 2

Common Use Cases
----------------

**Home Workout Setup**
- Setting up your camera for optimal recognition
- Creating a dedicated workout space
- Troubleshooting common issues

**Gym Integration**  
- Using I-Coach alongside traditional training
- Comparing AI feedback with trainer guidance
- Tracking progress over time

**Rehabilitation Applications**
- Monitoring exercise form during recovery
- Ensuring safe movement patterns
- Progress tracking for physical therapy

Code Examples
-------------

**Basic Usage**:

.. code-block:: python

   import streamlit as st
   
   # Launch I-Coach
   streamlit run app.py

**Custom Analysis**:

.. code-block:: python

   # Analyze specific video file
   result, temp_path = analyze_video_file(video_file)
   
   if result:
       exercise = result['exercise']
       confidence = result['confidence']
       print(f"Detected: {exercise} ({confidence:.1%})")

Sample Videos
-------------

Download these sample videos to test I-Coach:

- `sample_squat.mp4` - Perfect squat form demonstration
- `sample_pushup.mp4` - Push-up with form corrections  
- `sample_curl.mp4` - Proper hammer curl technique

Integration Examples
--------------------

**Fitness App Integration**
- REST API endpoints for exercise recognition
- Mobile app integration patterns
- Cloud deployment strategies

**Wearable Device Integration**
- Combining with heart rate monitors
- Activity tracking synchronization
- Multi-modal fitness analysis
