.. AI_Fitness_Tracker documentation master file, created by
   sphinx-quickstart on Wed May 21 08:48:32 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AI workout Tracker documentation
================================
Welcome to the documentation of AI workout Tracker, this document gives a detailed description
of the methodologies, tools, and results for the project.
Table of contents:
------------------

- Introduction
- Modelisation
- Installation

Introduction
------------

**AI Workout Tracker (AIWT)** is an AI-powered tool designed to assess exercise form. Whether you're a beginner or an experienced fitness
enthusiast, maintaining the correct form during exercises is critical for preventing injuries and maximizing workout efficiency. AIWT leverages computer vision and
machine learning techniques to analyze your posture, movement, and execution of exercises in real-time.

This documentation will walk you through the system architecture, how to set up AIWT, and how to use it for form checking. It also includes guidelines for contributing 
to the project.

Modelisation
------------

The AI Workout Tracker (AIWT) model follows a two-step process for exercise form evaluation. First, it uses a classification model to identify which exercise the 
user is performing by analyzing key body landmarks through pose estimation. Once the exercise is classified (e.g., squat, push-up, or lunge), the system then 
directs the data to the corresponding form-checking model, which uses a Long Short-Term Memory (LSTM) network to analyze the temporal sequence of movements. 
Instead of relying solely on joint angles, the LSTM model tracks the evolution of the user's posture and movement over time, allowing it to assess the form dynamically.
For example, in a squat, the LSTM evaluates how the user's body moves throughout the exercise, checking for correct knee alignment and hip depth. This approach ensures 
that the system provides accurate, real-time feedback tailored to the specific exercise being performed.
.. toctree::
   :maxdepth: 2
   :caption: Contents:

