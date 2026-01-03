import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# --- Load Model ---
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="models/best_float32.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Detection Logic ---
def detect_damage(frame):
    h, w, _ = frame.shape
    # Preprocess
    img = cv2.resize(frame, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predict
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # YOLO11 Drawing (Output shape [1, 5, 8400])
    for i in range(8400):
        score = output[4][i]
        if score > 0.45:
            xc, yc, wb, hb = output[0][i], output[1][i], output[2][i], output[3][i]
            x1, y1 = int((xc - wb/2) * w), int((yc - hb/2) * h)
            x2, y2 = int((xc + wb/2) * w), int((yc + hb/2) * h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, f"DENT {int(score*100)}%", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame

# --- UI Layout ---
st.title("🛡️ AI Car Guard")
mode = st.sidebar.selectbox("Choose Mode", ["Live Webcam", "Upload Video"])

if mode == "Live Webcam":
    # Live detection using WebRTC
    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        processed = detect_damage(img)
        return av.VideoFrame.from_ndarray(processed, format="bgr24")

    webrtc_streamer(key="car-scan", video_frame_callback=video_frame_callback,
                    media_stream_constraints={"video": True, "audio": False})

else:
    uploaded_video = st.file_uploader("Upload car video", type=["mp4", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        
        # Output Setup
        cap = cv2.VideoCapture(tfile.name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        output_path = "detected_damage.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        st_frame = st.empty()
        progress_bar = st.progress(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            processed = detect_damage(frame)
            out.write(processed) # Save with boxes
            st_frame.image(processed, channels="BGR")
            
        cap.release()
        out.release()
        
        with open(output_path, "rb") as file:
            st.download_button("📥 DOWNLOAD DETECTED VIDEO", file, "damage_report.mp4")