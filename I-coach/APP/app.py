import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import tempfile
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import time
import threading
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, GRU
from tensorflow.keras.utils import to_categorical
import warnings
import json
import pickle
import re

warnings.filterwarnings("ignore")
# Suppress warnings and TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

response = {
    'pushups_beginner': "For beginner push-ups: Start on your knees (modified push-up). Keep your body straight, hands shoulder-width apart. Lower chest to ground, push back up. Aim for 3 sets of 5-10 reps. Focus on form over quantity!",

    'pushups_progression': "To progress push-ups: Master regular push-ups first (3x15). Then try: incline push-ups → regular → decline → diamond push-ups → one-arm push-ups. Increase reps gradually before moving to harder variations.",

    'pushups_mistakes': "Common push-up mistakes: Sagging hips, flaring elbows too wide, incomplete range of motion, holding breath. Keep core tight, elbows at 45°, chest touches ground, breathe throughout movement.",

    'pullups_beginner': "Beginner pull-ups: Start with dead hangs (30s), then negative pull-ups (jump up, lower slowly). Use resistance bands or assisted pull-up machine. Practice scapular pulls. Be patient - pull-ups take time to develop!",

    'pullups_assistance': "Assisted pull-up options: Resistance bands around knees, assisted pull-up machine, partner holding feet, or box-assisted pull-ups. Gradually reduce assistance as you get stronger.",

    'pullups_grip': "Pull-up grips: Wide grip (lats focus), shoulder-width (balanced), chin-ups (underhand, biceps focus). Start with shoulder-width overhand grip for best beginner results.",

    'hammer_curl_form': "Hammer curl technique: Stand straight, dumbbells at sides with neutral grip (palms facing each other). Curl up keeping elbows stationary, squeeze at top, lower slowly. Don't swing or use momentum!",

    'hammer_curl_weight': "Hammer curl weight selection: Start with 10-15 lbs for beginners. You should complete 8-12 reps with good form, struggling on last 2-3 reps. Increase weight when you can do 15 reps easily.",

    'bench_press_beginner': "Beginner bench press: Lie flat, feet on floor, grip bar slightly wider than shoulders. Lower bar to chest with control, pause briefly, press up smoothly. Start with empty barbell (45 lbs) to learn form. Always use a spotter!",

    'bench_press_safety': "Bench press safety: ALWAYS use a spotter or safety bars. Don't lift alone. Warm up properly. Use proper form over heavy weight. Know your limits. Have a clear escape plan if you fail a rep.",

    'bench_press_grip': "Bench press grip: Hands slightly wider than shoulders, full grip (not thumbless), wrists straight. Grip too wide reduces range of motion, too narrow stresses wrists. Find your comfortable, strong position.",

    'squat_beginner': "Beginner squat: Stand with feet shoulder-width apart, toes slightly out. Keep chest up, core tight. Sit back like sitting in chair, knees track over toes. Go down until thighs parallel to floor, drive through heels to stand.",

    'squat_depth': "Squat depth: Aim for thighs parallel to floor (90°) minimum. Deeper is better if you have mobility. Stop where you can maintain good form. Improve flexibility gradually to increase depth safely.",

    'squat_knee_issues': "Knee-friendly squats: Ensure knees track over toes, don't cave inward. Start with bodyweight, focus on proper form. Consider box squats or wall squats. If pain persists, consult a physical therapist.",

    'workout_frequency': "Workout frequency: Beginners: 3x per week with rest days between. More advanced: 4-5x per week. Listen to your body. Quality over quantity - better to do 3 good workouts than 6 poor ones.",

    'rest_recovery': "Rest and recovery: Allow 48-72 hours between training same muscle groups. Get 7-9 hours sleep. Stay hydrated. Light activity on rest days is fine. Your muscles grow during rest, not just during workouts!",

    'sets_reps': "Sets and reps: Beginners: 2-3 sets of 8-12 reps for most exercises. Strength: 3-5 sets of 1-5 reps. Endurance: 2-3 sets of 15+ reps. Start conservative and progress gradually.",

    'warm_up': "Warm-up routine: 5-10 minutes light cardio, then dynamic stretches and movement prep. For upper body: arm circles, shoulder rolls. For lower body: leg swings, bodyweight squats. Prepare your body for work ahead!",

    'motivation': "Stay motivated: Set realistic goals, track progress, find a workout buddy, vary your routine, celebrate small wins. Remember why you started. Progress isn't always linear - stay consistent and trust the process!",

    'general_advice': "General fitness advice: Start slowly and focus on form. Be consistent rather than perfect. Progressive overload is key. Listen to your body. Combine exercise with proper nutrition and adequate rest for best results."
}

intents_data = {
    'pushups_beginner': [
        'how to do push ups for beginners', 'pushup tutorial beginner', 'start doing pushups',
        'pushup form beginner', 'learn pushups basics', 'pushup technique new', 'beginner pushup guide'
    ],
    'pushups_progression': [
        'pushup progression', 'advance pushups', 'harder pushups', 'pushup variations',
        'improve pushups', 'pushup next level', 'pushup difficulty increase'
    ],
    'pushups_mistakes': [
        'pushup mistakes', 'wrong pushup form', 'pushup errors', 'bad pushup technique',
        'pushup form check', 'common pushup problems', 'pushup corrections'
    ],
    'pullups_beginner': [
        'pullup for beginners', 'how to start pullups', 'pullup tutorial beginner', 'learn pullups',
        'pullup basics', 'first pullup', 'pullup assistance beginner'
    ],
    'pullups_assistance': [
        'assisted pullups', 'pullup bands', 'pullup machine', 'negative pullups',
        'pullup help', 'pullup support', 'easier pullups'
    ],
    'pullups_grip': [
        'pullup grip', 'pullup hand position', 'wide grip pullups', 'chin up vs pullup',
        'pullup grip width', 'pullup grip types', 'pullup hand placement'
    ],
    'hammer_curl_form': [
        'hammer curl form', 'hammer curl technique', 'how to hammer curl', 'hammer curl beginner',
        'hammer curl proper form', 'hammer curl tutorial', 'hammer curl basics'
    ],
    'hammer_curl_weight': [
        'hammer curl weight', 'hammer curl dumbbell size', 'hammer curl weight selection',
        'how heavy hammer curl', 'hammer curl starting weight', 'hammer curl weight beginner'
    ],
    'bench_press_beginner': [
        'bench press beginner', 'how to bench press', 'bench press tutorial', 'bench press form',
        'bench press technique', 'bench press basics', 'learn bench press'
    ],
    'bench_press_safety': [
        'bench press safety', 'bench press spotter', 'bench press accidents', 'safe bench press',
        'bench press precautions', 'bench press injury prevention', 'bench press risks'
    ],
    'bench_press_grip': [
        'bench press grip', 'bench press hand position', 'bench press grip width',
        'bench press bar grip', 'bench press hand placement', 'bench press grip technique'
    ],
    'squat_beginner': [
        'squat for beginners', 'how to squat', 'squat tutorial', 'squat form beginner',
        'basic squat technique', 'learn squats', 'squat basics'
    ],
    'squat_depth': [
        'squat depth', 'how low squat', 'squat range motion', 'deep squats',
        'squat bottom position', 'squat depth proper', 'full squat'
    ],
    'squat_knee_issues': [
        'squat knee pain', 'squat knee problems', 'squats hurt knees', 'knee safe squats',
        'squat knee injury', 'squat knee protection', 'squats bad knees'
    ],
    'workout_frequency': [
        'how often workout', 'workout frequency', 'training frequency', 'workout schedule',
        'how many times week workout', 'workout routine frequency', 'training schedule'
    ],
    'rest_recovery': [
        'rest between workouts', 'recovery time', 'muscle recovery', 'rest days',
        'workout recovery', 'how long rest', 'recovery period'
    ],
    'sets_reps': [
        'how many sets reps', 'sets and reps', 'rep ranges', 'set rep scheme',
        'how many reps', 'rep count', 'set count'
    ],
    'warm_up': [
        'warm up exercises', 'pre workout warmup', 'how to warm up', 'warmup routine',
        'warm up before workout', 'workout preparation', 'exercise warmup'
    ],
    'motivation': [
        'workout motivation', 'fitness motivation', 'stay motivated', 'exercise motivation',
        'motivation to workout', 'fitness inspiration', 'workout encouragement'
    ],
    'general_advice': [
        'fitness advice', 'workout tips', 'exercise advice', 'fitness tips',
        'general fitness', 'workout guidance', 'exercise help'
    ]
}
# Update the response dictionary
response.update({
    # Push-ups
    'pushups_daily': "Daily push-ups: It's okay to do push-ups daily if you're doing low-moderate volume. For high intensity, allow rest days. Listen to your body - if you feel sore or weak, take a day off.",
    'pushups_muscle_groups': "Push-ups work: Chest (pectorals), shoulders (anterior deltoids), triceps, and core. Secondary muscles include serratus anterior and upper back for stabilization. It's a compound movement!",
    'pushups_hand_placement': "Push-up hand positions: Standard (shoulder-width), wide grip (more chest), narrow/diamond (more triceps), staggered (uneven strength). Experiment to find what feels strongest for you.",
    'pushups_breathing': "Push-up breathing: Inhale on the way down, exhale on the way up. Don't hold your breath! Proper breathing helps maintain core stability and provides oxygen to working muscles.",
    'pushups_plateau': "Push-up plateau solutions: Increase reps per set, add more sets, try harder variations, add weight with a backpack, or incorporate tempo changes (slow negatives). Mix up your routine!",

    # Pull-ups
    'pullups_progression': "Pull-up progression: Dead hang → scapular pulls → negative pull-ups → assisted pull-ups → partial reps → full pull-ups → weighted pull-ups. Master each step before advancing.",
    'pullups_muscle_groups': "Pull-ups target: Latissimus dorsi (main back muscle), rhomboids, middle traps, rear delts, biceps, and forearms. It's one of the best upper body compound exercises!",
    'pullups_frequency': "Pull-up frequency: 2-3 times per week for beginners, allowing rest between sessions. Advanced users can train more frequently with varied grips and intensities.",
    'pullups_common_issues': "Pull-up problems: Kipping (swinging), partial range of motion, gripping too tight, not engaging lats first. Focus on controlled movement, full range, and proper muscle activation.",
    'pullups_alternatives': "Pull-up alternatives: Lat pulldowns, inverted rows, resistance band pull-aparts, or TRX rows. These help build the strength needed for eventual pull-ups.",

    # Hammer Curls
    'hammer_curl_vs_bicep': "Hammer curls vs bicep curls: Hammer curls target brachialis and brachioradialis more, giving arm thickness. Bicep curls focus on bicep peak. Include both for complete arm development.",
    'hammer_curl_tempo': "Hammer curl tempo: 2 seconds up, 1 second squeeze, 3 seconds down. Controlled movement maximizes muscle tension and growth. Avoid bouncing or using momentum.",
    'hammer_curl_mistakes': "Hammer curl errors: Swinging weights, moving elbows, partial range of motion, gripping too tight. Keep elbows fixed, full range, and maintain wrist alignment throughout.",
    'hammer_curl_variations': "Hammer curl variations: Standing, seated, alternating arms, both arms together, cable hammer curls, or hammer curls with different grips. Vary to prevent plateaus.",

    # Bench Press
    'bench_press_arch': "Bench press arch: Slight natural arch is okay and safe. Keeps shoulders stable and reduces injury risk. Don't over-arch - your butt should stay on the bench. Focus on squeezing shoulder blades together.",
    'bench_press_breathing': "Bench press breathing: Take deep breath at top, hold during descent and press, exhale at top or halfway up. This creates core stability and helps with heavier weights.",
    'bench_press_muscle_groups': "Bench press works: Chest (pectorals), front deltoids, and triceps as primary movers. Secondary muscles include core, lats, and leg muscles for stability.",
    'bench_press_plateau': "Bench press plateau: Vary rep ranges, try different grips, add pause reps, incorporate dumbbell variations, or focus on weak points. Sometimes deload and rebuild strength.",
    'bench_press_frequency': "Bench press frequency: 2-3 times per week for most people. Allow 48-72 hours recovery between sessions. You can vary intensity - heavy, medium, light days.",

    # Squats
    'squat_foot_position': "Squat foot position: Shoulder-width apart, toes slightly pointed out (15-30°). Find your natural stance - everyone's hip anatomy is different. Your knees should track over your toes.",
    'squat_breathing': "Squat breathing: Deep breath at top, hold during descent and ascent, exhale at top. This braces your core and provides stability for heavier weights.",
    'squat_muscle_groups': "Squats work: Quadriceps, glutes, hamstrings, calves, and core. It's a full-body exercise that also engages upper back and shoulders for stability.",
    'squat_variations': "Squat variations: Bodyweight, goblet squats, front squats, back squats, Bulgarian split squats, jump squats. Progress from bodyweight to weighted versions gradually.",
    'squat_common_mistakes': "Squat mistakes: Knees caving in, forward lean, not going deep enough, weight on toes, looking up. Focus on knees out, chest up, weight on heels, neutral spine.",
    'squat_mobility': "Squat mobility: Hip flexors, ankles, and thoracic spine affect squat depth. Stretch regularly, do mobility work, consider heel elevation if ankle mobility is limited.",

    # General Training
    'progressive_overload': "Progressive overload: Gradually increase weight, reps, or sets over time. This forces adaptation and growth. Track your workouts to ensure you're progressing. Small increases are better than big jumps.",
    'workout_structure': "Workout structure: Warm-up → compound exercises → isolation exercises → cool-down. Start with big movements when you're fresh, finish with smaller accessory work.",
    'form_vs_weight': "Form vs weight: Perfect form with lighter weight beats sloppy form with heavy weight. Master the movement first, then add weight. Your joints and muscles will thank you long-term.",
})

# Update the intents_data dictionary
intents_data.update({
    # Push-ups
    'pushups_daily': [
        'daily pushups', 'pushups every day', 'pushup daily routine', 'daily pushup challenge',
        'pushups everyday', 'can i do pushups daily', 'daily pushup workout', 'pushup daily training'
    ],
    'pushups_muscle_groups': [
        'pushup muscles worked', 'what muscles do pushups work', 'pushup muscle groups',
        'pushup muscle activation', 'muscles used in pushups', 'pushup muscle benefits',
        'pushup muscle development', 'pushup muscle targeting'
    ],
    'pushups_hand_placement': [
        'pushup hand position', 'pushup hand placement', 'pushup grip', 'pushup hand width',
        'where to put hands pushup', 'pushup hand spacing', 'pushup hand alignment',
        'pushup hand variations', 'pushup grip width'
    ],
    'pushups_breathing': [
        'pushup breathing', 'how to breathe doing pushups', 'pushup breathing technique',
        'breathing during pushups', 'pushup breath control', 'pushup breathing pattern',
        'proper breathing pushups', 'pushup respiratory technique'
    ],
    'pushups_plateau': [
        'pushup plateau', 'stuck at pushups', 'cant improve pushups', 'pushup progress stuck',
        'pushup plateau breakthrough', 'pushup stagnation', 'pushup improvement plateau',
        'overcome pushup plateau', 'pushup progress stalled'
    ],

    # Pull-ups
    'pullups_progression': [
        'pullup progression', 'pullup advancement', 'pullup training progression', 'pullup improvement',
        'pullup workout progression', 'get better at pullups', 'pullup skill development',
        'pullup progression plan', 'pullup training program', 'pullup development stages'
    ],
    'pullups_muscle_groups': [
        'pullup muscles worked', 'what muscles do pullups work', 'pullup muscle groups',
        'pullup muscle activation', 'muscles used in pullups', 'pullup muscle benefits',
        'pullup muscle development', 'pullup muscle targeting', 'pullup muscle engagement'
    ],
    'pullups_frequency': [
        'pullup frequency', 'how often pullups', 'pullup training frequency', 'pullup schedule',
        'pullup workout frequency', 'pullup training schedule', 'how many times pullups',
        'pullup routine frequency', 'pullup training days'
    ],
    'pullups_common_issues': [
        'pullup problems', 'pullup issues', 'pullup difficulties', 'pullup challenges',
        'pullup common mistakes', 'pullup form problems', 'pullup technique issues',
        'pullup execution problems', 'pullup performance issues'
    ],
    'pullups_alternatives': [
        'pullup alternatives', 'pullup substitutes', 'exercises instead of pullups',
        'pullup replacement exercises', 'pullup alternative movements', 'pullup substitutions',
        'exercises like pullups', 'pullup equivalent exercises'
    ],

    # Hammer Curls
    'hammer_curl_vs_bicep': [
        'hammer curl vs bicep curl', 'hammer curl vs regular curl', 'hammer curl difference',
        'hammer curl vs traditional curl', 'hammer curl comparison', 'hammer curl benefits vs bicep curl',
        'hammer curl vs standard curl', 'hammer curl vs normal curl'
    ],
    'hammer_curl_tempo': [
        'hammer curl tempo', 'hammer curl speed', 'hammer curl timing', 'hammer curl pace',
        'hammer curl rhythm', 'hammer curl cadence', 'hammer curl rep speed', 'hammer curl control'
    ],
    'hammer_curl_mistakes': [
        'hammer curl mistakes', 'hammer curl errors', 'hammer curl form problems',
        'hammer curl technique issues', 'hammer curl common mistakes', 'hammer curl wrong form',
        'hammer curl execution errors', 'hammer curl performance mistakes'
    ],
    'hammer_curl_variations': [
        'hammer curl variations', 'hammer curl types', 'hammer curl alternatives',
        'different hammer curls', 'hammer curl exercise variations', 'hammer curl modifications',
        'hammer curl workout variations', 'hammer curl movement variations'
    ],

    # Bench Press
    'bench_press_arch': [
        'bench press arch', 'bench press back arch', 'bench press spine position',
        'bench press body position', 'bench press setup', 'bench press posture',
        'bench press back position', 'bench press arch technique'
    ],
    'bench_press_breathing': [
        'bench press breathing', 'how to breathe bench press', 'bench press breathing technique',
        'breathing during bench press', 'bench press breath control', 'bench press breathing pattern',
        'proper breathing bench press', 'bench press respiratory technique'
    ],
    'bench_press_muscle_groups': [
        'bench press muscles worked', 'what muscles does bench press work', 'bench press muscle groups',
        'bench press muscle activation', 'muscles used in bench press', 'bench press muscle benefits',
        'bench press muscle development', 'bench press muscle targeting'
    ],
    'bench_press_plateau': [
        'bench press plateau', 'stuck at bench press', 'cant improve bench press', 'bench press progress stuck',
        'bench press plateau breakthrough', 'bench press stagnation', 'bench press improvement plateau',
        'overcome bench press plateau', 'bench press progress stalled'
    ],
    'bench_press_frequency': [
        'bench press frequency', 'how often bench press', 'bench press training frequency',
        'bench press schedule', 'bench press workout frequency', 'bench press training schedule',
        'how many times bench press', 'bench press routine frequency'
    ],

    # Squats
    'squat_foot_position': [
        'squat foot position', 'squat stance', 'squat foot placement', 'squat foot width',
        'where to put feet squat', 'squat foot spacing', 'squat foot alignment',
        'squat stance width', 'squat foot angle'
    ],
    'squat_breathing': [
        'squat breathing', 'how to breathe squatting', 'squat breathing technique',
        'breathing during squats', 'squat breath control', 'squat breathing pattern',
        'proper breathing squats', 'squat respiratory technique'
    ],
    'squat_muscle_groups': [
        'squat muscles worked', 'what muscles do squats work', 'squat muscle groups',
        'squat muscle activation', 'muscles used in squats', 'squat muscle benefits',
        'squat muscle development', 'squat muscle targeting'
    ],
    'squat_variations': [
        'squat variations', 'squat types', 'different squats', 'squat alternatives',
        'squat exercise variations', 'squat modifications', 'squat workout variations',
        'squat movement variations', 'squat training variations'
    ],
    'squat_common_mistakes': [
        'squat mistakes', 'squat errors', 'squat form problems', 'squat technique issues',
        'squat common mistakes', 'squat wrong form', 'squat execution errors',
        'squat performance mistakes', 'squat form issues'
    ],
    'squat_mobility': [
        'squat mobility', 'squat flexibility', 'squat mobility exercises', 'squat mobility training',
        'improve squat mobility', 'squat mobility work', 'squat flexibility training',
        'squat mobility drills', 'squat range of motion'
    ],

    # General Training
    'progressive_overload': [
        'progressive overload', 'workout progression', 'training progression', 'exercise progression',
        'progressive training', 'workout advancement', 'training advancement',
        'progressive overload principle', 'workout progress', 'training progress'
    ],
    'workout_structure': [
        'workout structure', 'workout organization', 'workout order', 'exercise order',
        'workout planning', 'training structure', 'workout routine structure',
        'workout layout', 'exercise sequence', 'workout arrangement'
    ],
    'form_vs_weight': [
        'form vs weight', 'form over weight', 'proper form', 'exercise form',
        'technique vs weight', 'form importance', 'weight vs form', 'good form',
        'exercise technique', 'proper technique', 'form first'
    ]
})
# Page configuration
st.set_page_config(
    page_title="💪 AI Exercise Classifier",
    page_icon="💪",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #667eea 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    }

    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.2rem;
    }

    .form-checker-good {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.1rem;
    }

    .form-checker-bad {
        background: linear-gradient(135deg, #dc3545 0%, #fd7e14 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.1rem;
    }

    .live-prediction {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }

    .live-stats {
        background: linear-gradient(135deg, #17a2b8 0%, #138496 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
    }

    .video-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
    }

    .control-panel {
        background: linear-gradient(135deg, #6c757d 0%, #495057 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = None
if 'prediction_confidence' not in st.session_state:
    st.session_state.prediction_confidence = 0.0
if 'form_feedback' not in st.session_state:
    st.session_state.form_feedback = None
if 'frames_processed' not in st.session_state:
    st.session_state.frames_processed = 0
if 'last_prediction_time' not in st.session_state:
    st.session_state.last_prediction_time = 0
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'chatbot_model' not in st.session_state:
    st.session_state.chatbot_model = None
if 'chatbot_tokenizer' not in st.session_state:
    st.session_state.chatbot_tokenizer = None

# Exercise labels
label_map = ['bench press', 'hammer curl', 'pull up', 'push-up', 'squat']
inv_label_map = {i: label for i, label in enumerate(label_map)}


# Custom Attention Layer for form checking
class AdditiveAttention(Layer):
    def __init__(self, units, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, values, query):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(query_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

    def get_config(self):
        config = super(AdditiveAttention, self).get_config()
        config.update({'units': self.units})
        return config


# Load models
@st.cache_resource
def load_exercise_model():
    try:
        possible_paths = [
            r"D:\Downloads\best_exercise_classifier.h5",
            "./best_exercise_classifier.h5",
            "./models/best_exercise_classifier.h5"
        ]

        for path in possible_paths:
            if os.path.exists(path):
                model = tf.keras.models.load_model(path, compile=False)
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                return model

        st.error("⚠️ Classification model not found.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


@st.cache_resource
def load_form_model():
    try:
        possible_paths = [
            r"D:\Downloads\best_attention_model.h5",
            "./best_attention_model.h5",
            "./models/best_attention_model.h5"
        ]

        custom_objects = {'AdditiveAttention': AdditiveAttention}

        for path in possible_paths:
            if os.path.exists(path):
                model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                return model

        return None
    except Exception as e:
        return None


@st.cache_resource
def load_pushup_form_model():
    try:
        possible_paths = [
            r"D:\Downloads\best_pu_attention_model.h5",
            "./best_pu_attention_model.h5",
            "./models/best_pu_attention_model.h5"
        ]

        custom_objects = {'AdditiveAttention': AdditiveAttention}

        for path in possible_paths:
            if os.path.exists(path):
                model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                return model

        return None
    except Exception as e:
        return None


@st.cache_resource
def load_squat_form_model():
    """Load the squat form checking model"""
    try:
        possible_paths = [
            "./best_squat_attention_model.h5",
            "./models/best_squat_attention_model.h5",
            r"D:\Downloads\best_squat_rnn_model.h5",
            "./best_squat_model.h5",
            "./models/best_squat_model.h5"
        ]

        custom_objects = {'AdditiveAttention': AdditiveAttention}

        for path in possible_paths:
            if os.path.exists(path):
                model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
                model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                return model

        return None
    except Exception as e:
        return None


# Load all models
exercise_model = load_exercise_model()
form_model = load_form_model()
pushup_form_model = load_pushup_form_model()
squat_form_model = load_squat_form_model()

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_pose_landmarks(landmarks, include_visibility=False):
    try:
        if not landmarks or len(landmarks) < 33:
            return [0] * (132 if include_visibility else 99)

        keypoints = []
        for lm in landmarks:
            if include_visibility:
                keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
            else:
                keypoints.extend([lm.x, lm.y, lm.z])

        return keypoints
    except Exception:
        return [0] * (132 if include_visibility else 99)


def analyze_form_with_model(form_sequence, exercise_type):
    if len(form_sequence) == 0:
        return None

    # Select appropriate model based on exercise type
    target_model = None
    if exercise_type == 'hammer curl' and form_model is not None:
        target_model = form_model
    elif exercise_type == 'push-up' and pushup_form_model is not None:
        target_model = pushup_form_model
    elif exercise_type == 'squat' and squat_form_model is not None:
        target_model = squat_form_model

    if target_model is None:
        return None

    try:
        landmarks_array = np.array(form_sequence)

        # Ensure proper shape (with visibility data)
        if landmarks_array.shape[1] != 132:
            if landmarks_array.shape[1] == 99:
                reshaped = landmarks_array.reshape(landmarks_array.shape[0], 33, 3)
                visibility = np.ones((landmarks_array.shape[0], 33, 1))
                with_vis = np.concatenate([reshaped, visibility], axis=2)
                landmarks_array = with_vis.reshape(landmarks_array.shape[0], 132)

        max_frames = 100
        if len(landmarks_array) > max_frames:
            landmarks_array = landmarks_array[-max_frames:]

        final_sequence = landmarks_array.tolist()
        while len(final_sequence) < max_frames:
            final_sequence.append([0] * 132)

        processed_sequence = np.array(final_sequence)
        input_data = np.expand_dims(processed_sequence, axis=0)
        form_pred = target_model.predict(input_data, verbose=0)

        # Handle squat model (5-class classification) differently
        if exercise_type == 'squat':
            squat_classes = ['bad_back_wrap', 'bad_back_round', 'bad_inner_thigh', 'correct', 'shallow']
            predicted_class = np.argmax(form_pred[0])
            form_confidence = float(np.max(form_pred[0]))
            predicted_form = squat_classes[predicted_class]

            is_correct = predicted_form == 'correct'

            if predicted_form == 'correct':
                if form_confidence > 0.8:
                    feedback = "Perfect squat form! Excellent depth, posture, and alignment! 🎯"
                elif form_confidence > 0.6:
                    feedback = "Great squat form! Keep maintaining that technique! ✅"
                else:
                    feedback = "Good squat form overall 👍"
            elif predicted_form == 'bad_back_wrap':
                feedback = "⚠️ Back Position Issue: Keep your chest up and avoid excessive forward lean. Engage your core!"
            elif predicted_form == 'bad_back_round':
                feedback = "⚠️ Rounded Back Detected: Maintain neutral spine. Keep chest up and shoulders back!"
            elif predicted_form == 'bad_inner_thigh':
                feedback = "⚠️ Knee Alignment Issue: Keep knees tracking over your toes. Strengthen inner thighs and glutes!"
            elif predicted_form == 'shallow':
                feedback = "⚠️ Insufficient Depth: Go deeper! Aim to get thighs parallel to ground or below!"

            return {
                'is_correct': is_correct,
                'confidence': form_confidence,
                'feedback': feedback,
                'predicted_form': predicted_form,
                'form_type': 'multi_class'
            }

        else:
            # Binary classification for hammer curl and push-up
            form_confidence = float(form_pred[0][0])
            is_correct = form_confidence > 0.5

            if exercise_type == 'hammer curl':
                if form_confidence > 0.8:
                    feedback = "Excellent hammer curl form!" if is_correct else "Poor hammer curl form detected"
                elif form_confidence > 0.6:
                    feedback = "Good hammer curl form" if is_correct else "Hammer curl form needs improvement"
                else:
                    feedback = "Fair hammer curl form" if is_correct else "Check your hammer curl form"
            elif exercise_type == 'push-up':
                if form_confidence > 0.8:
                    feedback = "Excellent push-up form!" if is_correct else "Poor push-up form detected"
                elif form_confidence > 0.6:
                    feedback = "Good push-up form" if is_correct else "Push-up form needs improvement"
                else:
                    feedback = "Fair push-up form" if is_correct else "Check your push-up form"

            return {
                'is_correct': is_correct,
                'confidence': form_confidence,
                'feedback': feedback,
                'form_type': 'binary'
            }

    except Exception as e:
        return None


class LivePredictionSystem:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.sequence_buffer = []
        self.form_buffer = []
        self.prediction_interval = 3  # 3 seconds

    def process_frame(self, frame):
        """Process a single frame and return annotated frame with predictions"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        # Draw pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract keypoints and add to buffer
            keypoints = extract_pose_landmarks(results.pose_landmarks.landmark)
            form_keypoints = extract_pose_landmarks(results.pose_landmarks.landmark, include_visibility=True)

            self.sequence_buffer.append(keypoints)
            self.form_buffer.append(form_keypoints)

            # Keep only last 150 frames (about 5 seconds at 30fps)
            if len(self.sequence_buffer) > 150:
                self.sequence_buffer.pop(0)
                self.form_buffer.pop(0)

        # Check if we should make a prediction
        current_time = time.time()
        if (current_time - st.session_state.last_prediction_time >= self.prediction_interval and
                len(self.sequence_buffer) >= 30):
            self.make_prediction()
            st.session_state.last_prediction_time = current_time

        # Add prediction text to frame
        if st.session_state.current_prediction:
            exercise = st.session_state.current_prediction
            confidence = st.session_state.prediction_confidence

            # Draw prediction text
            cv2.putText(frame, f"Exercise: {exercise.title()}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {confidence:.1%}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Add form feedback if available
            if st.session_state.form_feedback:
                # Split long feedback into multiple lines
                feedback = st.session_state.form_feedback
                if len(feedback) > 50:
                    feedback = feedback[:50] + "..."

                cv2.putText(frame, feedback,
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Draw timer for next prediction
        time_until_next = self.prediction_interval - (current_time - st.session_state.last_prediction_time)
        if time_until_next > 0:
            cv2.putText(frame, f"Next prediction: {time_until_next:.1f}s",
                        (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        st.session_state.frames_processed += 1
        return frame

    def make_prediction(self):
        """Make exercise and form predictions"""
        if exercise_model is None or len(self.sequence_buffer) < 30:
            return

        try:
            # Prepare sequence for prediction
            sequence = self.sequence_buffer[-100:] if len(self.sequence_buffer) > 100 else self.sequence_buffer

            padded_seq = np.zeros((100, 99))
            padded_seq[:len(sequence)] = sequence

            # Normalize
            if padded_seq.std() > 1e-6:
                padded_seq = (padded_seq - padded_seq.mean()) / padded_seq.std()

            # Make prediction
            prediction = exercise_model.predict(np.expand_dims(padded_seq, axis=0), verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = np.max(prediction)

            exercise_name = inv_label_map[predicted_class]

            # Update session state
            st.session_state.current_prediction = exercise_name
            st.session_state.prediction_confidence = confidence

            # Form checking for specific exercises
            if (exercise_name in ['hammer curl', 'push-up', 'squat'] and
                    confidence > 0.5 and len(self.form_buffer) >= 50):
                form_result = analyze_form_with_model(self.form_buffer, exercise_name)
                if form_result:
                    st.session_state.form_feedback = form_result['feedback']
                else:
                    st.session_state.form_feedback = None
            else:
                st.session_state.form_feedback = None

        except Exception as e:
            st.error(f"Prediction error: {e}")


# Initialize prediction system
if 'prediction_system' not in st.session_state:
    st.session_state.prediction_system = LivePredictionSystem()


def save_video_temporarily(uploaded_file):
    """Save uploaded video to temporary file and return path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name


def analyze_video_file(uploaded_file):
    if exercise_model is None:
        st.error("Exercise model not loaded.")
        return None

    temp_path = save_video_temporarily(uploaded_file)

    try:
        pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)
        cap = cv2.VideoCapture(temp_path)
        sequence = []
        form_sequence = []

        while cap.isOpened() and len(sequence) < 150:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                keypoints = extract_pose_landmarks(results.pose_landmarks.landmark)
                sequence.append(keypoints)

                form_keypoints = extract_pose_landmarks(results.pose_landmarks.landmark, include_visibility=True)
                form_sequence.append(form_keypoints)

        cap.release()

        if len(sequence) == 0:
            return None, temp_path

        padded_seq = np.zeros((100, 99))
        padded_seq[:min(len(sequence), 100)] = sequence[:100]

        if padded_seq.std() > 1e-6:
            padded_seq = (padded_seq - padded_seq.mean()) / padded_seq.std()

        prediction = exercise_model.predict(np.expand_dims(padded_seq, axis=0), verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        result = {
            'exercise': inv_label_map[predicted_class],
            'confidence': confidence
        }

        # Form checking for hammer curl, push-up, and squat
        exercise_type = inv_label_map[predicted_class]
        if exercise_type in ['hammer curl', 'push-up', 'squat'] and confidence > 0.5 and len(form_sequence) >= 50:
            form_result = analyze_form_with_model(form_sequence, exercise_type)
            if form_result:
                result['form_check'] = form_result

        return result, temp_path

    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        return None, temp_path


def cleanup_temp_file(temp_path):
    """Clean up temporary file"""
    try:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
    except:
        pass


class FitnessChatbot:
    def __init__(self):
        self.MAX_SEQ_LEN = 20
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.model = None
        self.tokenizer = None
        self.intent_labels = list(intents_data.keys())

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        return ' '.join(words)

    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer"""
        try:
            # Possible paths for the chatbot model
            model_paths = [
                r"D:\Downloads\fitness_lstm_model.h5",
                "./chatbot_model.h5",
                "./models/chatbot_model.h5",
                r"D:\Downloads\fitness_chatbot_model.h5",
                "./fitness_chatbot_model.h5"
            ]

            # Possible paths for the tokenizer
            tokenizer_paths = [
                r"D:\Downloads\fitness_lstm_tokenizer.pkl",
                "./tokenizer.pkl",
                "./models/tokenizer.pkl",
                r"D:\Downloads\chatbot_tokenizer.pkl",
                "./chatbot_tokenizer.pkl"
            ]

            # Load model
            model_loaded = False
            for path in model_paths:
                if os.path.exists(path):
                    self.model = tf.keras.models.load_model(path)
                    model_loaded = True
                    break

            # Load tokenizer
            tokenizer_loaded = False
            for path in tokenizer_paths:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        self.tokenizer = pickle.load(f)
                    tokenizer_loaded = True
                    break

            if model_loaded and tokenizer_loaded:
                return True
            else:
                missing = []
                if not model_loaded:
                    missing.append("model")
                if not tokenizer_loaded:
                    missing.append("tokenizer")
                st.warning(f"⚠️ Could not load chatbot {', '.join(missing)}. Files not found in expected locations.")
                return False

        except Exception as e:
            st.error(f"❌ Error loading chatbot components: {str(e)}")
            return False

    def predict_intent(self, user_input, confidence_threshold=0.3):
        if not self.model or not self.tokenizer:
            return 'general_advice', 0.0

        processed = self.preprocess_text(user_input)
        seq = self.tokenizer.texts_to_sequences([processed])
        padded = pad_sequences(seq, maxlen=self.MAX_SEQ_LEN)
        preds = self.model.predict(padded, verbose=0)
        confidence = np.max(preds)
        intent_idx = np.argmax(preds)

        if confidence >= confidence_threshold:
            return self.intent_labels[intent_idx], confidence
        else:
            return 'general_advice', confidence

    def get_response(self, user_input):
        try:
            intent, confidence = self.predict_intent(user_input)
            return {
                'response': responses[intent],
                'intent': intent,
                'confidence': float(confidence)
            }
        except Exception as e:
            return {
                'response': responses['general_advice'],
                'intent': 'general_advice',
                'confidence': 0.0,
                'error': str(e)
            }


# Initialize chatbot
@st.cache_resource
def load_chatbot():
    try:
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')

        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        chatbot = FitnessChatbot()

        # Try to load pre-trained model and tokenizer
        with st.spinner('Loading AI Fitness Coach...'):
            if chatbot.load_model_and_tokenizer():
                st.success("✅ Chatbot loaded from saved files!")
                return chatbot
            else:
                st.info("ℹ️ Pre-trained chatbot not found. The chatbot feature will be limited.")
                return None

    except Exception as e:
        st.error(f"Failed to load chatbot: {e}")
        return None


# Main Application
def main():
    st.markdown("""
    <div class="main-header">
        <h1> WELCOME TO I-COACH</h1>
        <p>AI-powered exercise recognition with advanced form checking for squats, hammer curls, and push-ups</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Model Status")

        if exercise_model:
            st.success("✅ Classification Model: Loaded")
        else:
            st.error("❌ Classification Model: Not Available")

        if form_model:
            st.success("✅ Hammer Curl Form Checker: AI-Powered")
        else:
            st.warning("⚠️ Hammer Curl Form Checker: Not Available")

        if pushup_form_model:
            st.success("✅ Push-up Form Checker: AI-Powered")
        else:
            st.warning("⚠️ Push-up Form Checker: Not Available")

        if squat_form_model:
            st.success("✅ Squat Form Checker: AI-Powered ")
        else:
            st.warning("⚠️ Squat Form Checker: Not Available")

        st.markdown("### 🏋️ Supported Exercises")
        for exercise in label_map:
            if exercise in ['hammer curl', 'push-up', 'squat']:
                st.write(f"• {exercise.title()} 🤖 (with AI form checking)")
            else:
                st.write(f"• {exercise.title()}")

        st.markdown("### 📋 Instructions")
        st.markdown("""
        **Live Prediction:**
        1. Click "Start Camera" to begin
        2. Position yourself in camera view
        3. Start performing exercises
        4. Get predictions every 3 seconds
        5. Receive real-time form feedback

        **Video Upload:**
        1. Upload MP4/AVI video file
        2. Get instant analysis with form checking
        3. View your uploaded video

        **Advanced Form Checking:**
        - **Squats**: 5 specific form types detected
        - **Hammer Curls**: Controlled movement analysis
        - **Push-ups**: Body alignment and range of motion
        """)

    # Main content
    # Main content
    tab1, tab2, tab3 = st.tabs(["📹 Live Camera", "📁 Video Upload", "🤖 AI Fitness Coach"])

    with tab1:
        st.markdown("### 📹 Live Exercise Recognition")

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown('<div class="video-container">', unsafe_allow_html=True)

            # Camera controls
            camera_placeholder = st.empty()

            if not st.session_state.camera_running:
                if st.button("🎥 Start Camera", key="start_cam"):
                    st.session_state.camera_running = True
                    st.rerun()
            else:
                if st.button("⏹️ Stop Camera", key="stop_cam"):
                    st.session_state.camera_running = False
                    st.rerun()

            # Camera feed
            if st.session_state.camera_running:
                try:
                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("❌ Cannot access camera. Please check your camera permissions.")
                        st.session_state.camera_running = False
                    else:
                        frame_placeholder = st.empty()

                        for _ in range(300):  # Run for about 10 seconds at 30fps
                            if not st.session_state.camera_running:
                                break

                            ret, frame = cap.read()
                            if not ret:
                                break

                            # Process frame
                            processed_frame = st.session_state.prediction_system.process_frame(frame)

                            # Convert to RGB for display
                            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

                            time.sleep(0.033)  # ~30fps

                        cap.release()
                        st.session_state.camera_running = False

                except Exception as e:
                    st.error(f"Camera error: {str(e)}")
                    st.session_state.camera_running = False

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            st.markdown("### 📊 Live Stats")

            # Current prediction display
            if st.session_state.current_prediction:
                st.markdown(f"""
                <div class="live-prediction">
                    <strong>{st.session_state.current_prediction.title()}</strong><br>
                    <small>{st.session_state.prediction_confidence:.1%} confidence</small>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="live-prediction">
                    <strong>No Exercise Detected</strong><br>
                    <small>Start exercising to see predictions</small>
                </div>
                """, unsafe_allow_html=True)

            # Form feedback
            if st.session_state.form_feedback:
                if "Perfect" in st.session_state.form_feedback or "Excellent" in st.session_state.form_feedback:
                    st.markdown(f"""
                    <div class="form-checker-good">
                        <strong>Form Check:</strong><br>
                        {st.session_state.form_feedback}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="form-checker-bad">
                        <strong>Form Check:</strong><br>
                        {st.session_state.form_feedback}
                    </div>
                    """, unsafe_allow_html=True)

            # Stats
            st.markdown(f"""
            <div class="live-stats">
                <strong>Frames Processed:</strong> {st.session_state.frames_processed}
            </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown("### 📁 Video Upload Analysis")

        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze exercise form and classification"
        )

        if uploaded_file is not None:
            st.markdown("#### 📋 Analysis Results")

            with st.spinner('🔄 Analyzing your video... This may take a moment.'):
                result, temp_path = analyze_video_file(uploaded_file)

            if result:
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Display video
                    st.markdown("#### 🎥 Your Video")
                    st.video(uploaded_file)

                with col2:
                    # Display results
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>🏋️ Exercise Detected</h3>
                        <h2>{result['exercise'].title()}</h2>
                        <p>Confidence: {result['confidence']:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Form checking results
                    if 'form_check' in result:
                        form_result = result['form_check']

                        if form_result['is_correct']:
                            st.markdown(f"""
                            <div class="form-checker-good">
                                <h4>✅ Form Analysis</h4>
                                <p>{form_result['feedback']}</p>
                                <small>Confidence: {form_result['confidence']:.1%}</small>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="form-checker-bad">
                                <h4>⚠️ Form Analysis</h4>
                                <p>{form_result['feedback']}</p>
                                <small>Confidence: {form_result['confidence']:.1%}</small>
                            </div>
                            """, unsafe_allow_html=True)

                    # Exercise-specific tips
                    exercise_tips = {
                        'squat': "💡 **Squat Tips:** Keep chest up, knees tracking over toes, go to parallel depth",
                        'push-up': "💡 **Push-up Tips:** Maintain straight line from head to heels, full range of motion",
                        'hammer curl': "💡 **Hammer Curl Tips:** Control the weight, avoid swinging, squeeze at the top",
                        'pull up': "💡 **Pull-up Tips:** Full range of motion, controlled movement, engage lats",
                        'bench press': "💡 **Bench Press Tips:** Proper bar path, stable shoulders, controlled descent"
                    }

                    if result['exercise'] in exercise_tips:
                        st.info(exercise_tips[result['exercise']])

            else:
                st.error("❌ Could not analyze the video. Please ensure the video shows clear exercise movements.")

            # Cleanup
            cleanup_temp_file(temp_path)

    with tab3:
        st.markdown("### 🤖 AI Fitness Coach")
        st.markdown("Ask me anything about fitness, exercise form, or workout routines!")

        # Initialize chatbot
        if st.session_state.chatbot_model is None:
            st.session_state.chatbot_model = load_chatbot()

        # Chat interface
        if st.session_state.chatbot_model:
            # Display chat history
            for i, (user_msg, bot_response) in enumerate(st.session_state.chat_history):
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                           padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                    <strong>You:</strong> {user_msg}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                           padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                    <strong>AI Coach:</strong> {bot_response}
                </div>
                """, unsafe_allow_html=True)

            # Input for new message
            user_input = st.text_input("Ask your fitness question:", key="chat_input")

            if st.button("Send", key="send_msg") and user_input.strip():
                with st.spinner('🤔 Thinking...'):
                    response_data = st.session_state.chatbot_model.get_response(user_input)

                    # Add to chat history
                    st.session_state.chat_history.append((user_input, response_data['response']))

                    # Keep only last 10 exchanges
                    if len(st.session_state.chat_history) > 10:
                        st.session_state.chat_history.pop(0)

                    st.rerun()

            # Quick questions
            st.markdown("#### Quick Questions:")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("How to improve squat form?"):
                    response_data = st.session_state.chatbot_model.get_response("How to improve squat form?")
                    st.session_state.chat_history.append(("How to improve squat form?", response_data['response']))
                    st.rerun()

                if st.button("Best exercises for beginners?"):
                    response_data = st.session_state.chatbot_model.get_response("Best exercises for beginners?")
                    st.session_state.chat_history.append(("Best exercises for beginners?", response_data['response']))
                    st.rerun()

            with col2:
                if st.button("How many reps should I do?"):
                    response_data = st.session_state.chatbot_model.get_response("How many reps should I do?")
                    st.session_state.chat_history.append(("How many reps should I do?", response_data['response']))
                    st.rerun()

                if st.button("Workout routine suggestions?"):
                    response_data = st.session_state.chatbot_model.get_response("Workout routine suggestions?")
                    st.session_state.chat_history.append(("Workout routine suggestions?", response_data['response']))
                    st.rerun()

        else:
            st.warning(
                "⚠️ AI Fitness Coach is not available. Please ensure the chatbot model files are properly loaded.")
            st.markdown("""
            ### 💡 General Fitness Tips:
            - **Consistency is key**: Regular exercise is more important than intensity
            - **Form over weight**: Perfect your technique before increasing weight
            - **Progressive overload**: Gradually increase difficulty over time
            - **Rest and recovery**: Allow adequate rest between workout sessions
            - **Stay hydrated**: Drink plenty of water before, during, and after workouts
            - **Listen to your body**: Stop if you feel pain or excessive fatigue
            """)


if __name__ == "__main__":
    main()