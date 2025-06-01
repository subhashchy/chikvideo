import streamlit as st
import cv2
import numpy as np
import time
import logging
from roboflow import Roboflow
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import os
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create upload directory if it doesn't exist
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Page config
st.set_page_config(page_title="Traffic Violation Detection", layout="wide")
st.title("Traffic Violation Detection")

# API Configuration
API_KEY = "eH8jjiNqGKLLNTHW26Zl"

# Initialize Roboflow globally
logger.info("Initializing Roboflow models...")
rf = Roboflow(api_key=API_KEY)

# Define models configuration
MODELS = {
    "Triple Riding Detection": {
        "project": "traffic-violation-2",
        "version": 2,
        "instance": None,
        "classes": {
            "helmet": (0, 255, 0),  # Green
            "motorcycle": (255, 0, 0),  # Blue
            "no_helmet": (0, 0, 255)  # Red
        }
    },
    "No Helmet Detection": {
        "project": "rider-plate-headcls3",
        "version": 4,
        "instance": None,
        "classes": {
            "with_helmet": (0, 255, 0),  # Green
            "without_helmet": (0, 0, 255),  # Red
            "number_plate": (255, 255, 0)  # Yellow
        }
    }
}

# Configuration
FRAME_SKIP = 3
CONFIDENCE_THRESHOLD = 0.4
MAX_WORKERS = 3
FRAME_BUFFER_SIZE = 5

# Get available video files
def get_video_files():
    video_files = glob.glob("media/*.mp4") + glob.glob("media/*.avi")
    return [os.path.basename(f) for f in video_files]

# Source type options
SOURCE_TYPES = ["Video File", "RTSP Stream", "Upload Video"]

# Create a thread-safe queue for frames
frame_queue = Queue(maxsize=FRAME_BUFFER_SIZE)
result_queue = Queue(maxsize=FRAME_BUFFER_SIZE)

# Thread-safe counter for unique temp files
temp_file_counter = 0
temp_file_lock = threading.Lock()

def get_temp_filename():
    global temp_file_counter
    with temp_file_lock:
        temp_file_counter += 1
        return f"temp_frame_{temp_file_counter}.jpg"

def process_single_frame(frame, selected_mode):
    try:
        # Save frame temporarily with unique name
        temp_path = get_temp_filename()
        cv2.imwrite(temp_path, frame)
        
        try:
            # Get predictions using the selected model
            model_config = MODELS[selected_mode]
            if model_config["instance"] is None:
                project = rf.workspace().project(model_config["project"])
                model_config["instance"] = project.version(model_config["version"]).model
                
            result = model_config["instance"].predict(temp_path, confidence=40, overlap=30).json()
            logger.info(f"Raw API Response: {result}")
            
            # Process predictions manually
            for pred in result.get("predictions", []):
                x1 = int(pred['x'] - pred['width'] / 2)
                y1 = int(pred['y'] - pred['height'] / 2)
                x2 = int(pred['x'] + pred['width'] / 2)
                y2 = int(pred['y'] + pred['height'] / 2)
                
                # Get color based on class from model config
                class_name = pred['class'].lower()
                color = None
                for key, col in model_config["classes"].items():
                    if key in class_name:
                        color = col
                        break
                if color is None:
                    color = (255, 255, 255)  # White for unknown classes
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{pred['class']}: {pred['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return frame, result
        finally:
            # Clean up temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
    except Exception as e:
        logger.error(f"Error in process_frame: {str(e)}")
        logger.exception("Full traceback:")
        return frame, {"predictions": []}

def frame_producer(cap, stop_event, selected_mode):
    frame_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if source_type == "Video File":
                    break
                else:
                    # For RTSP, try to reconnect
                    logger.warning("Lost RTSP connection, attempting to reconnect...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(video_source)
                    continue
                
            # Process every nth frame
            if frame_count % FRAME_SKIP == 0:
                # Submit frame for processing
                future = executor.submit(process_single_frame, frame.copy(), selected_mode)
                frame_queue.put((frame_count, future))
                
            frame_count += 1
            
            # Control processing speed
            time.sleep(0.01)

def frame_consumer(stop_event):
    last_frame_count = -1
    while not stop_event.is_set():
        try:
            frame_count, future = frame_queue.get(timeout=1.0)
            if frame_count > last_frame_count:
                processed_frame, result = future.result(timeout=2.0)
                result_queue.put((frame_count, processed_frame))
                last_frame_count = frame_count
        except:
            continue

# Initialize session state for button counter
if 'button_counter' not in st.session_state:
    st.session_state.button_counter = 0

# Sidebar controls
st.sidebar.title("Controls")

# Mode selection
detection_mode = st.sidebar.selectbox(
    "Select Detection Mode",
    list(MODELS.keys())
)

# Source type selection
source_type = st.sidebar.selectbox(
    "Select Source Type",
    SOURCE_TYPES
)

# Source selection based on type
if source_type == "Video File":
    video_files = get_video_files()
    if not video_files:
        st.sidebar.error("No video files found in media directory!")
        st.stop()
    video_source = st.sidebar.selectbox(
        "Select Video File",
        video_files
    )
    video_source = os.path.join("media", video_source)
elif source_type == "RTSP Stream":
    video_source = st.sidebar.text_input(
        "Enter RTSP URL",
        "rtsp://username:password@ip:port/stream"
    )
else:  # Upload Video
    uploaded_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi'])
    if uploaded_file is not None:
        # Save uploaded file
        upload_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(upload_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        video_source = upload_path
        st.sidebar.success("Video uploaded successfully!")
    else:
        st.sidebar.warning("Please upload a video file")
        st.stop()

confidence = st.sidebar.slider("Detection Confidence", 0.0, 1.0, CONFIDENCE_THRESHOLD)
frame_skip = st.sidebar.slider("Frame Skip", 1, 10, FRAME_SKIP)

# Display current mode info
st.sidebar.markdown("---")
st.sidebar.markdown(f"### {detection_mode} Info")
st.sidebar.markdown("Detection Classes:")
for class_name, color in MODELS[detection_mode]["classes"].items():
    # Convert RGB to hex for display
    hex_color = "#{:02x}{:02x}{:02x}".format(*color)
    st.sidebar.markdown(f'<span style="color: {hex_color}">■</span> {class_name.replace("_", " ").title()}', unsafe_allow_html=True)

# Main content
if st.sidebar.button("Start Detection"):
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
    producer_thread = threading.Thread(target=frame_producer, args=(cap, stop_event, detection_mode))
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
            if stop_button_placeholder.button("Stop Detection", key=f"stop_{st.session_state.button_counter}"):
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
        if source_type == "Upload Video" and os.path.exists(video_source):
            try:
                os.remove(video_source)
                logger.info(f"Cleaned up uploaded file: {video_source}")
            except Exception as e:
                logger.error(f"Error cleaning up uploaded file: {str(e)}")
        
        st.success("Detection completed!") 