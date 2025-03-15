import streamlit as st
import os
import cv2
import torch
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLO model
model_path = "best.pt"
model = YOLO(model_path)

# PCB Defect Classes
CLASSES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']

# Directory to store detected images
DETECTED_FOLDER = "detected_images"
os.makedirs(DETECTED_FOLDER, exist_ok=True)

# Streamlit UI
st.title("PCB Defect Detection")
st.write("Upload PCB images to detect defects.")

uploaded_files = st.file_uploader("Upload Images", accept_multiple_files=True, type=["jpg", "png", "jpeg"])

def detect_defects(image):
    results = model(image)
    defects = set()
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls.item())
            defect_name = CLASSES[cls]
            defects.add(defect_name)
    
    return list(defects), results[0]  # Return both defects and results

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Create columns for each image
        col1, col2 = st.columns(2)
        
        # Convert uploaded file to numpy array for detection
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        
        # Detect defects and get results
        defects, results = detect_defects(image)
        
        # Plot results with bounding boxes
        res_plotted = results.plot()  # Returns numpy array of image with boxes
        
        # Save the annotated image
        output_path = os.path.join(DETECTED_FOLDER, uploaded_file.name)
        cv2.imwrite(output_path, res_plotted)
        
        # Display image with bounding boxes in first column
        with col1:
            st.image(res_plotted, caption=uploaded_file.name, use_column_width=True)
        
        # Display defect list in second column
        with col2:
            if defects:
                st.write("### Detected Defects:")
                for defect in defects:
                    st.write(f"- {defect}")
            else:
                st.write("### No defects detected")
        
        # Add a separator between images
        st.markdown("---")
