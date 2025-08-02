import os
import logging
import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import imutils
from pathlib import Path
from NeuroScan.utils.logging import logger
from NeuroScan.utils.helpers import read_yaml, create_directories
from NeuroScan.constants.paths import CONFIG_PATH
from NeuroScan.utils.logging import logger


config = read_yaml(CONFIG_PATH)

# Function to crop the brain MRI images
def crop_image(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_thresh = cv2.threshold(img_blur, 45, 255, cv2.THRESH_BINARY)[1]
    img_thresh = cv2.erode(img_thresh, None, iterations=2)
    img_thresh = cv2.dilate(img_thresh, None, iterations=2)

    contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    
    if not contours:
        return None
    
    c = max(contours, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()])[0]
    extRight = tuple(c[c[:, :, 0].argmax()])[0]
    extTop = tuple(c[c[:, :, 1].argmin()])[0]
    extBottom = tuple(c[c[:, :, 1].argmax()])[0]

    new_img = image[extTop[1]:extBottom[1], extLeft[0]:extRight[0]]
    return new_img

def preprocess_image(img):
    cropped_img = crop_image(img)
    
    if cropped_img is None:
        st.error("Could not process the image. Please upload a clear brain MRI scan.")
        return None
    
    resized_img = cv2.resize(cropped_img, (240, 240))
    
    if len(resized_img.shape) == 2:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)
    elif resized_img.shape[2] == 4:
        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGBA2RGB)
    
    preprocessed_img = np.expand_dims(resized_img, axis=0)
    
    return preprocessed_img, cropped_img

def main():
    st.set_page_config(page_title="Brain MRI Tumor Classification", layout="wide")
    
    st.title("Brain MRI Tumor Classification")
    st.markdown("""
    This application uses a deep learning model to classify brain MRI scans into four categories:
    - Glioma
    - Meningioma
    - Pituitary Tumor
    - No Tumor
    """)
    
    uploaded_file = st.file_uploader("Upload a brain MRI scan", type=["jpg", "jpeg", "png"])
    
    @st.cache_resource
    def load_classification_model():
        try:
            model = load_model(config['model_trainer']['model_path'], compile=False)
            return model
        except:
            st.error("Error loading the model. Please make sure the model file exists.")
            return None
    
    model = load_classification_model()
    
    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        original_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        if model is not None:
            processed_img, cropped_img = preprocess_image(original_img)
            
            if processed_img is not None:
                with col2:
                    st.subheader("Processed Image")
                    st.image(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                prediction = model.predict(processed_img)
                class_idx = np.argmax(prediction[0])
                confidence = prediction[0][class_idx] * 100
                
                st.subheader("Prediction Results")
                st.markdown(f"**Diagnosis:** {class_labels[class_idx]}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                
                st.subheader("Probability Distribution")
                prob_df = {
                    'Class': class_labels,
                    'Probability (%)': [float(p*100) for p in prediction[0]]
                }
                
                st.bar_chart(prob_df, x='Class', y='Probability (%)')
                
                st.warning("""
                **Disclaimer:** This application is for educational purposes only. 
                The predictions should not be used for actual medical diagnosis.
                Always consult with qualified medical professionals for proper diagnosis and treatment.
                """)

if __name__ == "__main__":
    main()
