Installation Guide
==================

System Requirements
-------------------

**Hardware Requirements**
- Webcam (for live exercise recognition)
- Minimum 4GB RAM
- GPU recommended for faster inference (optional)

**Software Requirements**
- Python 3.7 or higher
- Operating System: Windows, macOS, or Linux

Dependencies
------------

Core Libraries
~~~~~~~~~~~~~~

.. code-block:: text

   streamlit>=1.28.0
   opencv-python>=4.8.0
   mediapipe>=0.10.0
   tensorflow>=2.10.0
   numpy>=1.21.0
   pandas>=1.3.0
   Pillow>=8.3.0

AI & ML Libraries
~~~~~~~~~~~~~~~~~

.. code-block:: text

   nltk>=3.8
   scikit-learn>=1.0.0

Installation Steps
------------------

1. **Clone the Repository**

.. code-block:: bash

   git clone https://github.com/yourusername/i-coach.git
   cd i-coach

2. **Create Virtual Environment** (Recommended)

.. code-block:: bash

   python -m venv i-coach-env
   
   # On Windows
   i-coach-env\Scripts\activate
   
   # On macOS/Linux
   source i-coach-env/bin/activate

3. **Install Dependencies**

.. code-block:: bash

   pip install -r requirements.txt

4. **Download NLTK Data**

.. code-block:: python

   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')

5. **Download Pre-trained Models**

Download the following model files and place them in the project directory:

- ``best_exercise_classifier.h5`` - Main exercise classification model
- ``best_attention_model.h5`` - Hammer curl form checker
- ``best_pu_attention_model.h5`` - Push-up form checker  
- ``best_squat_attention_model.h5`` - Squat form checker
- ``fitness_chatbot_model.h5`` - AI fitness coach model
- ``chatbot_tokenizer.pkl`` - Tokenizer for chatbot

Model Download Links
~~~~~~~~~~~~~~~~~~~~

.. note::
   Contact the development team for access to pre-trained models or train your own using the provided training scripts.

Verification
------------

Test your installation:

.. code-block:: bash

   streamlit run app.py

You should see the I-Coach web interface at ``http://localhost:8501``

Troubleshooting
---------------

**Camera Access Issues**
- Ensure camera permissions are granted
- Check if other applications are using the camera
- Try different camera indices if multiple cameras are available

**Model Loading Errors**
- Verify model files are in the correct directory
- Check file permissions
- Ensure sufficient disk space and memory

**Package Conflicts**
- Use a virtual environment to avoid conflicts
- Update pip: ``pip install --upgrade pip``
- Clear pip cache: ``pip cache purge``
