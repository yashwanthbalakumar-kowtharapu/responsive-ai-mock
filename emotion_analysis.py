import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

def process_frame(base64_string):
    try:
        # Remove header information from base64 string
        base64_string = base64_string.split(",")[1]
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_string)
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode numpy array to OpenCV image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Resize the frame
        frame = cv2.resize(frame, (0, 0), fx=0.7, fy=0.7)
        # Crop the frame if needed
        frame = frame[:, 50:, :]
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception as e:
        print(f"Error processing frame: {str(e)}")
        return None

def detect_emotions(frame):
    try:
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            return None
            
        # Placeholder for emotion detection (including "nervous")
        return [{
            'emotions': {
                'angry': 0.1,
                'disgust': 0.1,
                'fear': 0.1,
                'happy': 0.2,
                'sad': 0.1,
                'surprise': 0.1,
                'neutral': 0.2,
                'nervous': 0.1  # Added "nervous"
            }
        }]
    except Exception as e:
        print(f"Error detecting emotions: {str(e)}")
        return None

def calculate_average_emotions(emotion_data):
    num_frames = len(emotion_data)
    if num_frames == 0:
        return {
            'angry': 0.0,
            'disgust': 0.0,
            'fear': 0.0,
            'happy': 0.0,
            'sad': 0.0,
            'surprise': 0.0,
            'neutral': 1.0,  # Default to neutral if no frames
            'nervous': 0.0  # Default to 0.0 for nervous
        }
    
    emotion_sum = {}
    
    # Calculate sum of emotions
    for frame in emotion_data:
        for key, value in frame.items():
            emotion_sum[key] = emotion_sum.get(key, 0) + value

    # Calculate average of emotions
    average_emotions = {key: value / num_frames for key, value in emotion_sum.items()}
    
    # Round the values to three decimal points
    final_emotions = {key: round(value, 3) for key, value in average_emotions.items()}
    
    return final_emotions

def analyze_fun(frames):
    try:
        emotion_data = []

        for frame_data in frames:
            try:
                processed_frame = process_frame(frame_data)
                if processed_frame is not None:
                    emotions = detect_emotions(processed_frame)
                    if emotions and len(emotions) > 0:
                        emotion_data.append(emotions[0]['emotions'])
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                continue

        average_emotions = calculate_average_emotions(emotion_data)
        return average_emotions
    except Exception as e:
        print(f"Error in analyze_fun: {str(e)}")
        return {
            'angry': 0.0,
            'disgust': 0.0,
            'fear': 0.0,
            'happy': 0.0,
            'sad': 0.0,
            'surprise': 0.0,
            'neutral': 1.0,  # Default to neutral on error
            'nervous': 0.0  # Default to 0.0 for nervous
        }
