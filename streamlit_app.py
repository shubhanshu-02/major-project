import streamlit as st
import cv2
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

@st.cache_resource
def load_models():
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    image_model = load_model('model.h5')
    import joblib
    text_model = joblib.load('text_emotion.pkl')
    return face_classifier, image_model, text_model

face_classifier, image_model, text_model = load_models()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def predict_text_emotion(text):
    try:
        prediction = text_model.predict_proba([text])[0]
        text_emotions = text_model.classes_
        
        # Map text emotions to image emotion labels
        emotion_map = {'anger': 'Angry', 'disgust': 'Disgust', 'fear': 'Fear', 
                      'joy': 'Happy', 'neutral': 'Neutral', 'sadness': 'Sad', 
                      'surprise': 'Surprise', 'shame': 'Sad'}
        
        mapped_emotions = {}
        for i, text_emotion in enumerate(text_emotions):
            mapped_label = emotion_map.get(text_emotion, text_emotion.capitalize())
            if mapped_label in mapped_emotions:
                mapped_emotions[mapped_label] += prediction[i]
            else:
                mapped_emotions[mapped_label] = prediction[i]
                
        return mapped_emotions
    except Exception as e:
        st.error(f"Text emotion prediction error: {str(e)}")
        return None

def predict_image_emotion(image):
    image_array = np.array(image)
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        return None, image
    
    results = []
    # Create a copy of the image for drawing
    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)
    
    try:
        # Try to use a default font
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        # Fall back to default font if arial is not available
        font = ImageFont.load_default()
    
    for face_num, (x, y, w, h) in enumerate(faces, 1):
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        prediction = image_model.predict(roi)[0]
        result = {emotion_labels[i]: prediction[i] for i in range(len(emotion_labels))}
        results.append(result)
        
        # Get the predicted emotion with highest probability
        max_emotion = max(result, key=result.get)
        confidence = result[max_emotion]
        
        draw.rectangle([(x, y), (x + w, y + h)], outline="yellow", width=20)
        
        try:
            number_font = ImageFont.truetype("arial.ttf", 80)
        except:
            number_font = ImageFont.load_default()
        
        # Draw emotion label with confidence 
        label = f"{face_num}.{max_emotion}: {confidence:.2f}"
        try:
            big_font = ImageFont.truetype("arial.ttf", 100)
        except:
            big_font = ImageFont.load_default()
            
        text_bbox = draw.textbbox((0, 0), label, font=big_font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        draw.rectangle([(x, y-text_height-10), (x + text_width + 20, y)], fill="black")

        draw.text((x + 10, y - text_height - 5), label, fill="white", font=big_font)
    
    return results, image_with_boxes

st.title("Emotion Detection System")

option = st.selectbox("Choose input type:", ["Text", "Image", "Camera"])

if option == "Text":
    text_input = st.text_area("Enter text:", placeholder="Type your text here to analyze emotions...")
    if st.button("Analyze Text") and text_input:
        with st.spinner('Analyzing text emotion...'):
            emotions = predict_text_emotion(text_input)
            
        if emotions:
            st.subheader("Text Emotion Analysis")
            df = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Probability'])
            df = df.sort_values('Probability', ascending=False)
            
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(df.set_index('Emotion'))
            with col2:
                st.dataframe(df)
                
            # Show the dominant emotion
            dominant_emotion = df.iloc[0]['Emotion']
            confidence = df.iloc[0]['Probability']
            st.success(f"Dominant emotion: **{dominant_emotion}** (confidence: {confidence:.2%})")
        else:
            st.error("Failed to analyze text emotion. Please check if the model is properly loaded.")

elif option == "Image":
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        results, image_with_boxes = predict_image_emotion(image)
        
        st.subheader("Detected Faces")
        st.image(image_with_boxes, caption="Image with Face Detection", use_column_width=True)
        
        if results:
            st.subheader("Emotion Analysis Results")
            for i, emotions in enumerate(results):
                st.write(f"**Face {i+1}:**")
                df = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Probability'])
                df = df.sort_values('Probability', ascending=False)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.bar_chart(df.set_index('Emotion'))
                with col2:
                    st.dataframe(df)
                st.write("---")
        else:
            st.error("No face detected in the image")

elif option == "Camera":
    with st.spinner('Starting camera...'):
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("Could not open camera. Please check if your camera is connected and not being used by another application.")
            else:
                camera_placeholder = st.empty()
                results_placeholder = st.empty()
                stop_button = st.button("Stop Camera")
                
                frame_count = 0
                while not stop_button:
                    ret, frame = cap.read()
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
                        
                        predictions = []
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                            roi_gray = gray[y:y+h, x:x+w]
                            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
                            
                            if np.sum([roi_gray]) != 0:
                                roi = roi_gray.astype('float') / 255.0
                                roi = img_to_array(roi)
                                roi = np.expand_dims(roi, axis=0)
                                
                                prediction = image_model.predict(roi)[0]
                                emotion_probs = {emotion_labels[i]: prediction[i] for i in range(len(emotion_labels))}
                                predictions.append(emotion_probs)
                                
                                label = emotion_labels[prediction.argmax()]
                                cv2.putText(frame, f'{label} ({prediction.max():.2f})', (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        camera_placeholder.image(frame_rgb, channels="RGB", caption="Live Camera Feed")
                        
                        if predictions and frame_count % 30 == 0:
                            with results_placeholder.container():
                                st.subheader("Real-time Emotion Detection")
                                for i, emotions in enumerate(predictions):
                                    st.write(f"**Face {i+1}:**")
                                    df = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Probability'])
                                    df = df.sort_values('Probability', ascending=False)
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.bar_chart(df.set_index('Emotion'))
                                    with col2:
                                        st.dataframe(df)
                        
                        frame_count += 1
                    else:
                        break
                
                cap.release()
        except Exception as e:
            st.error(f"Camera error: {str(e)}")