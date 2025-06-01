import streamlit as st
import cv2
import numpy as np
import time
import logging
import insightface
import threading
from queue import Queue
import os
import glob
import torch
import asyncio
from typing import Tuple, List, Dict

# Fix for asyncio event loop
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for hardware acceleration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using hardware acceleration: {torch.cuda.is_available()}")

# Create upload directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Page config
st.set_page_config(page_title="Face Recognition Demo", layout="wide")
st.title("Face Recognition Demo")

# Initialize face analysis model
@st.cache_resource
def load_insightface_model():
    model = insightface.app.FaceAnalysis(
        allowed_modules=['detection', 'recognition', 'genderage'],
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    model.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
    return model

# Get available video files
def get_video_files():
    video_files = glob.glob("media/*.mp4") + glob.glob("media/*.avi")
    return [os.path.basename(f) for f in video_files]

# Configuration
FRAME_SKIP = 2
FRAME_BUFFER_SIZE = 10
MAX_WORKERS = 3

# Create processing queues
frame_queue = Queue(maxsize=FRAME_BUFFER_SIZE)
result_queue = Queue(maxsize=FRAME_BUFFER_SIZE)

# Source type options
SOURCE_TYPES = ["Sample Videos", "Live Camera Feed", "Upload Your Video"]

def process_single_frame(frame, model):
    try:
        start_time = time.time()
        
        # Optimize image size for processing
        height, width = frame.shape[:2]
        if width > 1280:
            scale = 1280 / width
            frame = cv2.resize(frame, (1280, int(height * scale)))
            
        # Prepare image for analysis
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Check processing time
        if time.time() - start_time > 2.0:
            logger.warning("Processing is taking longer than expected")
            return frame, []
            
        # Analyze faces in the image
        faces = model.get(frame_rgb)
        
        # Draw results on the image
        for face in faces:
            bbox = face.bbox.astype(int)
            # Draw face rectangle
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            
            try:
                # Draw facial features
                landmarks = face.landmark_2d_106
                if landmarks is not None:
                    landmarks = landmarks.astype(np.int32)
                    for landmark in landmarks:
                        cv2.circle(frame, (landmark[0], landmark[1]), 1, (0, 0, 255), -1)
            except Exception as e:
                logger.warning("Could not draw facial features")
            
            try:
                # Add demographic information
                gender = 'Male' if face.gender == 1 else 'Female'
                age = int(face.age)
                info_text = f"{gender}, {age} years"
                cv2.putText(frame, info_text, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                logger.warning("Could not add demographic information")
            
            try:
                # Add face analysis confidence
                if hasattr(face, 'embedding') and face.embedding is not None:
                    confidence = np.linalg.norm(face.embedding)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                              (bbox[0], bbox[3] + 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            except Exception as e:
                logger.warning("Could not add confidence score")
        
        return frame, faces
            
    except Exception as e:
        logger.error("Error processing image")
        return frame, []

def frame_producer(cap, stop_event, model):
    frame_count = 0
    errors = 0
    last_frame_time = time.time()
    
    while not stop_event.is_set():
        try:
            # Check for video feed issues
            if time.time() - last_frame_time > 5.0:
                logger.warning("Video feed interrupted, attempting to restore...")
                cap.release()
                time.sleep(1)
                if source_type == "Live Camera Feed":
                    cap = cv2.VideoCapture(video_source)
                else:
                    cap = cv2.VideoCapture(video_source)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                last_frame_time = time.time()
                continue

            ret, frame = cap.read()
            if not ret:
                errors += 1
                if errors > 30:
                    if source_type == "Sample Videos":
                        logger.info("Video playback completed")
                        break
                    else:
                        logger.warning("Connection lost, attempting to reconnect...")
                        cap.release()
                        time.sleep(1)
                        cap = cv2.VideoCapture(video_source)
                        errors = 0
                continue
            
            errors = 0
            last_frame_time = time.time()
            
            if frame_count % FRAME_SKIP == 0:
                try:
                    processed_frame, faces = process_single_frame(frame.copy(), model)
                    if not frame_queue.full():
                        frame_queue.put((frame_count, processed_frame), timeout=1.0)
                except Exception as e:
                    logger.error("Error analyzing frame")
                    if not frame_queue.full():
                        frame_queue.put((frame_count, frame))
            
            frame_count += 1
            
        except Exception as e:
            logger.error("Error processing video feed")
            time.sleep(0.1)
        
        time.sleep(0.01)

def frame_consumer(stop_event):
    last_frame_count = -1
    last_frame_time = time.time()
    
    while not stop_event.is_set():
        try:
            # Check for consumer hanging
            if time.time() - last_frame_time > 5.0:
                logger.warning("Consumer thread appears stuck, resetting...")
                last_frame_time = time.time()
                continue
                
            frame_count, processed_frame = frame_queue.get(timeout=1.0)
            if frame_count > last_frame_count:
                try:
                    if not result_queue.full():
                        result_queue.put((frame_count, processed_frame), timeout=1.0)
                    last_frame_count = frame_count
                    last_frame_time = time.time()
                except Exception as e:
                    logger.error(f"Error in result queue put: {str(e)}")
        except Exception as e:
            if not isinstance(e, queue.Empty):
                logger.error(f"Error in frame consumer: {str(e)}")
            continue

# Initialize session state for button counter
if 'button_counter' not in st.session_state:
    st.session_state.button_counter = 0

# Sidebar controls
st.sidebar.title("Controls")

# Display system status
if torch.cuda.is_available():
    st.sidebar.success("✓ System ready for fast processing")
else:
    st.sidebar.warning("⚠ System running in compatibility mode")

# Source type selection
source_type = st.sidebar.selectbox(
    "Choose Video Source",
    SOURCE_TYPES
)

# Source selection based on type
if source_type == "Sample Videos":
    video_files = get_video_files()
    if not video_files:
        st.sidebar.error("No sample videos available!")
        st.stop()
    video_source = st.sidebar.selectbox(
        "Choose a Sample Video",
        video_files
    )
    video_source = os.path.join("media", video_source)
elif source_type == "Live Camera Feed":
    video_source = st.sidebar.text_input(
        "Enter Camera URL",
        "rtsp://username:password@ip:port/stream"
    )
else:  # Upload Your Video
    uploaded_file = st.sidebar.file_uploader("Upload Your Video", type=['mp4', 'avi'])
    if uploaded_file is not None:
        # Save uploaded file
        upload_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_source = upload_path
        st.sidebar.success("✓ Video uploaded successfully!")
    else:
        st.sidebar.info("Please select a video file to upload")
        st.stop()

# Processing settings
st.sidebar.markdown("---")
st.sidebar.markdown("### Processing Settings")
frame_skip = st.sidebar.slider("Processing Speed", 1, 10, FRAME_SKIP,
                             help="Higher values = faster processing but less smooth video")

# Features info
st.sidebar.markdown("---")
st.sidebar.markdown("### Features")
st.sidebar.markdown("This demo can detect:")
st.sidebar.markdown("✓ Faces in video")
st.sidebar.markdown("✓ Age estimation")
st.sidebar.markdown("✓ Gender recognition")
st.sidebar.markdown("✓ Facial features")
st.sidebar.markdown("✓ Analysis confidence")

# Start button
if st.sidebar.button("Start Analysis", use_container_width=True):
    # Load model
    model = load_insightface_model()
    
    # Create video capture object
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error(f"Could not open video source: {video_source}")
        st.stop()

    # Create placeholder for video frame
    frame_placeholder = st.empty()
    stop_button_placeholder = st.empty()
    
    # Create stop event
    stop_event = threading.Event()
    
    # Start producer and consumer threads
    producer_thread = threading.Thread(target=frame_producer, args=(cap, stop_event, model))
    consumer_thread = threading.Thread(target=frame_consumer, args=(stop_event,))
    
    producer_thread.start()
    consumer_thread.start()
    
    fps_time = time.time()
    frames_processed = 0
    
    try:
        while True:
            if not result_queue.empty():
                frame_count, processed_frame = result_queue.get()
                
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Calculate and display FPS
                frames_processed += 1
                if frames_processed % 30 == 0:
                    current_time = time.time()
                    fps = 30 / (current_time - fps_time)
                    fps_time = current_time
                    st.sidebar.text(f"FPS: {fps:.2f}")
                
                # Display the frame
                frame_placeholder.image(rgb_frame, use_container_width=True)
            
            # Check for stop button with unique key
            st.session_state.button_counter += 1
            if stop_button_placeholder.button("Stop Recognition", key=f"stop_{st.session_state.button_counter}"):
                break
                
            time.sleep(0.001)  # Small delay to prevent UI freezing
            
    except Exception as e:
        logger.error(f"Error in main loop: {str(e)}")
        logger.exception("Full traceback:")
    
    finally:
        # Clean up
        stop_event.set()
        producer_thread.join()
        consumer_thread.join()
        cap.release()
        
        # Clean up uploaded file if it exists
        if source_type == "Upload Your Video" and os.path.exists(video_source):
            try:
                os.remove(video_source)
                logger.info(f"Cleaned up uploaded file: {video_source}")
            except Exception as e:
                logger.error(f"Error cleaning up uploaded file: {str(e)}")
        
        st.success("Recognition completed!") 