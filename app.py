import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
from pathlib import Path

try:
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.models import load_model
except ImportError:
    st.error("TensorFlow not installed. Please install tensorflow>=2.16.0")
    st.stop()

try:
    import detect_mask_image
except ImportError:
    st.warning("detect_mask_image module not found")

# Setting custom Page Title and Icon with changed layout and sidebar state
st.set_page_config(
    page_title='Face Mask Detector',
    page_icon='😷',
    layout='centered',
    initial_sidebar_state='expanded'
)

# Create necessary directories
PATH_IMAGES = Path("./images")
PATH_CSS = Path("./css")
PATH_FACE_DETECTOR = Path("./face_detector")
PATH_IMAGES.mkdir(exist_ok=True)
PATH_CSS.mkdir(exist_ok=True)

def local_css(file_name):
    """Method for reading styles.css and applying necessary changes to HTML"""
    css_file = Path(file_name)
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {file_name}")

def load_detection_models():
    """Load face detector and mask detector models"""
    try:
        prototxt_path = PATH_FACE_DETECTOR / "deploy.prototxt"
        weights_path = PATH_FACE_DETECTOR / "res10_300x300_ssd_iter_140000.caffemodel"
        
        if not prototxt_path.exists() or not weights_path.exists():
            st.error("Face detector model files not found. Please ensure face_detector files are in the correct location.")
            return None, None
        
        net = cv2.dnn.readNet(str(prototxt_path), str(weights_path))
        
        mask_model_path = "mask_detector.model"
        if not Path(mask_model_path).exists():
            st.error("Mask detector model not found. Please ensure mask_detector.model exists.")
            return net, None
        
        model = load_model(mask_model_path)
        return net, model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

def process_image(image_path, net, model):
    """Process image and return detected faces with mask status"""
    if net is None or model is None:
        st.error("Models not loaded properly")
        return None
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            st.error("Could not read image file")
            return None
        
        (h, w) = image.shape[:2]
        
        # Construct blob from image
        blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # Get face detections
        net.setInput(blob)
        detections = net.forward()
        
        # Process detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:
                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure bounding boxes are within image dimensions
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
                
                # Extract face ROI and preprocess
                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
                
                # Predict mask status
                (mask, withoutMask) = model.predict(face, verbose=0)[0]
                
                # Determine label and color
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                
                # Draw on image
                cv2.putText(image, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
        # Convert to RGB for display
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def main():
    """Main app function"""
    local_css("css/styles.css")
    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)
    
    # Load models once
    net, model = load_detection_models()
    
    activities = ["Image", "Webcam"]
    st.sidebar.markdown("# Mask Detection on?")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)
    
    if choice == 'Image':
        st.markdown('<h2 align="center">Detection on Image</h2>', unsafe_allow_html=True)
        st.markdown("### Upload your image here ⬇")
        image_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])
        
        if image_file is not None:
            try:
                our_image = Image.open(image_file)
                image_path = PATH_IMAGES / "out.jpg"
                our_image.save(str(image_path))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<h3 align="center">Uploaded Image</h3>', unsafe_allow_html=True)
                    st.image(image_file, use_column_width=True)
                
                st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)
                
                if st.button('Process Image'):
                    if net is not None and model is not None:
                        with st.spinner('Processing...'):
                            result_img = process_image(str(image_path), net, model)
                            if result_img is not None:
                                with col2:
                                    st.markdown('<h3 align="center">Processed Image</h3>', unsafe_allow_html=True)
                                    st.image(result_img, use_column_width=True)
                    else:
                        st.error("Models could not be loaded. Please check the model files.")
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    elif choice == 'Webcam':
        st.markdown('<h2 align="center">Detection on Webcam</h2>', unsafe_allow_html=True)
        st.markdown('<h3 align="center">This feature will be available soon!</h3>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
