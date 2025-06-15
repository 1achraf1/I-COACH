Quick Start Guide
=================

Get started with I-Coach in just a few minutes!

Launch the Application
----------------------

.. code-block:: bash

   streamlit run app.py

The application will open in your web browser at ``http://localhost:8501``

Interface Overview
------------------

I-Coach features three main tabs:

📹 **Live Camera**
   Real-time exercise recognition using your webcam

📁 **Video Upload** 
   Analyze pre-recorded workout videos

🤖 **AI Fitness Coach**
   Interactive chatbot for fitness guidance

Live Exercise Recognition
-------------------------

1. **Start Camera**
   
   Click the "🎥 Start Camera" button to begin live recognition

2. **Position Yourself**
   
   Stand in front of the camera with your full body visible

3. **Start Exercising**
   
   Begin performing one of the supported exercises:
   - Bench Press
   - Hammer Curl
   - Pull Up
   - Push-up
   - Squat

4. **Get Feedback**
   
   - Exercise predictions appear every 3 seconds
   - Form feedback is provided for hammer curls, push-ups, and squats
   - Real-time stats are displayed in the sidebar

Video Analysis
--------------

1. **Upload Video**
   
   Use the file uploader to select an MP4, AVI, MOV, or MKV file

2. **Wait for Analysis**
   
   The system will process your video and extract pose landmarks

3. **View Results**
   
   - Exercise classification with confidence score
   - Form analysis (if applicable)
   - Exercise-specific tips and recommendations

AI Fitness Coach
----------------

1. **Ask Questions**
   
   Type your fitness-related questions in the chat input

2. **Get Expert Advice**
   
   The AI coach provides personalized responses based on:
   - Exercise form guidance
   - Workout routines  
   - Beginner tips
   - Rep and set recommendations

3. **Quick Questions**
   
   Use the pre-defined buttons for common questions

Supported Exercises
-------------------

**With AI Form Checking:**

🤖 **Squats**
   - Detects 5 specific form issues
   - Back position analysis
   - Knee alignment checking
   - Depth assessment

🤖 **Hammer Curls**
   - Movement control analysis
   - Range of motion checking
   - Form quality assessment

🤖 **Push-ups**
   - Body alignment monitoring
   - Range of motion analysis
   - Form correctness evaluation

**Classification Only:**

- **Bench Press**: Exercise recognition and tips
- **Pull-ups**: Exercise recognition and guidance

Tips for Best Results
---------------------

**Camera Setup**
- Ensure good lighting
- Position camera to capture full body
- Maintain stable camera position
- Wear contrasting clothing against background

**Exercise Performance**
- Perform exercises with clear, deliberate movements
- Maintain consistent form throughout
- Allow the system 3-5 seconds to recognize the exercise
- Follow the form feedback for optimal results

**Video Upload**
- Use high-quality videos (720p or higher recommended)
- Ensure the exerciser is clearly visible
- Keep video length reasonable (under 2 minutes for faster processing)
- Good lighting and clear background improve accuracy

Next Steps
----------

- Explore the :doc:`tutorials/index` for detailed guides
- Check out :doc:`examples/index` for common use cases  
- Read the :doc:`user_guide/index` for advanced features
- Review :doc:`development/index` if you want to contribute
