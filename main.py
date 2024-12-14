import os
import cv2
import torch
import numpy as np
import streamlit as st
from tempfile import NamedTemporaryFile
from model_py.PreDNN import predict_PredRNN
custom_style = '''
<style>
    /* Main background and app styling */
    .stApp {
        background: linear-gradient(135deg, #af7ac5 0%, #8e44ad 100%);
    }
    
    /* Title styling */
    .main-title {
        color: white;
        font-family: 'Arial Black', sans-serif;
        font-size: 48px;
        text-align: center;
        padding: 20px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        background: rgba(0, 0, 0, 0.2);
        border-radius: 10px;
        animation: fadeIn 1.5s ease-in;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #f8f9fa;
        font-family: Arial, sans-serif;
        font-size: 28px;
        text-align: center;
        padding: 15px;
        margin-bottom: 30px;
        background: rgba(0, 0, 0, 0.1);
        border-radius: 8px;
        animation: slideIn 1.5s ease-out;
    }
    
    /* Description text styling */
    .description {
        color: white;
        font-family: Arial, sans-serif;
        font-size: 18px;
        text-align: center;
        padding: 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 5px;
        margin: 20px 0;
        line-height: 1.6;
        animation: fadeIn 2s ease-in;
    }
    
    /* Animations */
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    @keyframes slideIn {
        0% { transform: translateY(-50px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
    
    /* Container styling */
    .content-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 30px;
        border-radius: 15px;
        backdrop-filter: blur(5px);
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Custom button styling */
    .stButton>button {
        background-color: #8e44ad;
        color: white;
        border: 2px solid white;
        border-radius: 25px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: white;
        color: #8e44ad;
        transform: scale(1.05);
    }
    
    /* File uploader styling */
    .stFileUploader {
        padding: 20px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        border: 2px dashed white;
    }
</style>
'''

# Apply the custom styling
st.markdown(custom_style, unsafe_allow_html=True)

# Main content with custom classes
st.markdown('<h1 class="main-title">CS4045 - Deep Learning Course Project</h1>', unsafe_allow_html=True)
st.markdown('<h2 class="subtitle">Predicting and Generating Video Sequences Using Deep Learning</h2>', unsafe_allow_html=True)

# Create a container for the main content
st.markdown('<p class="description">Upload a video and select a model for processing and prediction. Our advanced deep learning models will analyze your video and generate predictions based on the learned patterns.</p>', unsafe_allow_html=True)


# Utility Function to Save Video
def save_video(output_name, frames, predicted_frames, output_path, fps=1):
    processed_frames = [cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR) for f in frames]
    for pred_frame in predicted_frames:
        processed_frames.append(cv2.cvtColor((pred_frame[0] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (64, 64))
    for frame in processed_frames:
        out.write(frame)
    out.release()
    st.write(f"{output_name} saved.")
    with open(output_path, "rb") as f:
        st.download_button(f"Download {output_name}", f, file_name=output_name)

# File Upload
uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi"])
if uploaded_file:
    # Save uploaded file temporarily
    temp_file = NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    video_path = temp_file.name

    
    # Process video and prepare input
    st.write("Processing video frames...")
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (64, 64))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0
        frames.append(gray_frame)

    cap.release()

    if len(frames) < 15:
        st.error("Uploaded video is too short for prediction (less than 15 frames).")
    else:
        # Select random middle 10 frames
        middle_frame_idx = len(frames) // 2
        start_idx = max(0, middle_frame_idx - 5)
        end_idx = min(len(frames), middle_frame_idx + 5)

        selected_frames = frames[start_idx:end_idx]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_frames = np.expand_dims(np.stack(selected_frames), axis=1)
        input_tensor = torch.tensor(input_frames, dtype=torch.float32).unsqueeze(0).to(device)
        # Run selected models

        st.write("Running PredRNN...")
        try:
            predicted_frames = predict_PredRNN(input_tensor, device)
            save_video("PredRNN_Output.mp4", selected_frames, predicted_frames, "PredRNN_output.mp4")
        except Exception as e:
            st.error(f"PredRNN Error: {str(e)}")