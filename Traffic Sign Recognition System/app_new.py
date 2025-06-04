import streamlit as st
import numpy as np
import cv2
import tempfile
import os
from tensorflow.keras.models import load_model
from PIL import Image

# Constants
frameWidth = 640
frameHeight = 480
threshold = 0.75  # Confidence threshold

# Load the model
@st.cache_resource
def load_traffic_model():
    try:
        return load_model("model.h5")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def enhanced_preprocessing(img):
    """Improved preprocessing pipeline"""
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE for better contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    
    # Normalize
    img = img/255.0
    
    return img

def getClassName(classNo):
    class_names = [
        'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
        'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
        'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
        'No passing', 'No passing for vehicles over 3.5 metric tons',
        'Right-of-way at the next intersection', 'Priority road', 'Yield',
        'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
        'No entry', 'General caution', 'Dangerous curve to the left',
        'Dangerous curve to the right', 'Double curve', 'Bumpy road',
        'Slippery road', 'Road narrows on the right', 'Road work',
        'Traffic signals', 'Pedestrians', 'Children crossing',
        'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
        'End of all speed and passing limits', 'Turn right ahead',
        'Turn left ahead', 'Ahead only', 'Go straight or right',
        'Go straight or left', 'Keep right', 'Keep left',
        'Roundabout mandatory', 'End of no passing',
        'End of no passing by vehicles over 3.5 metric tons'
    ]
    return class_names[classNo] if classNo < len(class_names) else "Unknown"

# Streamlit app
st.title("Traffic Sign Detection")
st.write("Upload an image or use your webcam to detect traffic signs")

# Load the model
model = load_traffic_model()
if model is None:
    st.stop()

# Create two columns for layout
col1, col2 = st.columns(2)

with col1:
    st.header("Input")
    input_type = st.radio("Select input type:", ["Upload Image", "Webcam"])

    if input_type == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Read the image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image with enhanced preprocessing
            processed_img = cv2.resize(img, (32, 32))
            processed_img = enhanced_preprocessing(processed_img)
            processed_img = processed_img.reshape(1, 32, 32, 1)
            
            # Make prediction
            predictions = model.predict(processed_img)
            classIndex = np.argmax(predictions, axis=1)[0]
            probabilityValue = np.amax(predictions)
            
            # Display results
            with col2:
                st.header("Results")
                st.image(img_rgb, caption="Original Image", use_column_width=True)
                
                if probabilityValue > threshold:
                    st.success(f"**Detected Sign:** {getClassName(classIndex)}")
                    st.success(f"**Confidence:** {probabilityValue*100:.2f}%")
                else:
                    st.warning(f"**Possible Sign:** {getClassName(classIndex)} (Low confidence)")
                    st.info(f"Confidence: {probabilityValue*100:.2f}% (Threshold: {threshold*100}%)")
                
                # Show top predictions if needed
                if st.checkbox("Show detailed predictions"):
                    top5 = np.argsort(predictions[0])[-5:][::-1]
                    st.write("Top 5 Predictions:")
                    for i in top5:
                        st.write(f"- {getClassName(i)}: {predictions[0][i]*100:.2f}%")

    else:  # Webcam
        st.write("Capture an image from your webcam")
        camera_image = st.camera_input("Take a picture of traffic sign")
        
        if camera_image is not None:
            # Convert to OpenCV format
            bytes_data = camera_image.getvalue()
            img_array = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Process the image
            processed_img = cv2.resize(img, (32, 32))
            processed_img = enhanced_preprocessing(processed_img)
            processed_img = processed_img.reshape(1, 32, 32, 1)
            
            # Make prediction
            predictions = model.predict(processed_img)
            classIndex = np.argmax(predictions, axis=1)[0]
            probabilityValue = np.amax(predictions)
            
            # Display results
            with col2:
                st.header("Results")
                st.image(img_rgb, caption="Captured Image", use_column_width=True)
                
                if probabilityValue > threshold:
                    st.success(f"**Detected Sign:** {getClassName(classIndex)}")
                    st.success(f"**Confidence:** {probabilityValue*100:.2f}%")
                else:
                    st.warning(f"**Possible Sign:** {getClassName(classIndex)} (Low confidence)")
                    st.info(f"Confidence: {probabilityValue*100:.2f}% (Threshold: {threshold*100}%)")
                
                # Show processed image for debugging
                if st.checkbox("Show processed image (what model sees)"):
                    st.image(processed_img[0,:,:,0], caption="Processed Image", clamp=True)

# Add tips for better results
st.markdown("""
### Tips for Better Detection:
1. Ensure the sign is clearly visible and centered
2. Capture in good lighting conditions
3. Avoid extreme angles or glare
4. Crop unnecessary background if possible
5. Try multiple angles if detection fails
""")

# Add some styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stAlert {
        padding: 15px;
        border-radius: 10px;
    }
    .st-bb {
        background-color: #4CAF50 !important;
    }
</style>
""", unsafe_allow_html=True)