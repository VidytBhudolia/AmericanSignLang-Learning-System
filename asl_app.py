import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import time
import random
from collections import deque
import contextlib
import warnings
warnings.filterwarnings('ignore')

try:
    torch.backends.mkldnn.enabled = False
    torch.set_num_threads(1)
except Exception:
    pass

st.set_page_config(
    page_title="ASL Learning System",
    page_icon="🤟",
    layout="wide"
)


if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'alphabet_order' not in st.session_state:
    st.session_state.alphabet_order = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'word_index' not in st.session_state:
    st.session_state.word_index = 0
if 'letter_index' not in st.session_state:
    st.session_state.letter_index = 0
if 'evaluation_words' not in st.session_state:
    st.session_state.evaluation_words = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = deque(maxlen=30)
if 'correct_count' not in st.session_state:
    st.session_state.correct_count = 0
if 'last_feedback_time' not in st.session_state:
    st.session_state.last_feedback_time = 0
if 'device' not in st.session_state:
    st.session_state.device = None
if 'available_cameras' not in st.session_state:
    st.session_state.available_cameras = None


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

@contextlib.contextmanager
def get_camera(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    try:
        yield cap
    finally:
        cap.release()

def detect_cameras(max_test=5):
    available_cameras = []
    for i in range(max_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()
    return available_cameras if available_cameras else [0]


def load_asl_images():
    asl_images = {}
    asl_folder = Path(__file__).parent / "asl_images"
    
    if asl_folder.exists():
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                img_path = asl_folder / f"{letter}{ext}"
                if img_path.exists():
                    asl_images[letter] = str(img_path)
                    break
    return asl_images


def load_evaluation_words():
    words = []
    words_folder = Path(__file__).parent / "evaluation_words"
    
    if words_folder.exists():
        word_files = list(words_folder.glob("*.txt"))
        for word_file in word_files:
            with open(word_file, 'r') as f:
                file_words = [line.strip().upper() for line in f if line.strip()]
                words.extend(file_words)
    
    if not words:
        words = ['HELLO', 'WORLD', 'PEACE', 'LEARN', 'SIGN', 'LANGUAGE', 'TEACH']
    
    return words


class MultiScaleDetector:
    def __init__(self, model, scales=[0.8, 1.0, 1.2], device='cpu'):
        self.model = model
        self.scales = scales
        self.device = device
    
    def predict(self, frame, confidence=0.5):
        all_predictions = []
        h, w = frame.shape[:2]
        
        for scale in self.scales:
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_frame = cv2.resize(frame, (new_w, new_h))
            
            results = self.model(scaled_frame, conf=confidence, verbose=False, device=self.device)
            
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes
                for box in boxes:
                    all_predictions.append({
                        'class': int(box.cls[0]),
                        'conf': float(box.conf[0]),
                        'scale': scale
                    })
        
        if all_predictions:
            best = max(all_predictions, key=lambda x: x['conf'])
            letter = chr(65 + best['class']) if best['class'] < 26 else None
            
            results = self.model(frame, conf=confidence, verbose=False, device=self.device)
            return letter, best['conf'], results[0].plot()
        
        results = self.model(frame, conf=confidence, verbose=False, device=self.device)
        return None, 0.0, results[0].plot()


class LandmarkGuidedDetector:
    def __init__(self):
        try:
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.available = True
        except ImportError:
            self.available = False
    
    def get_hand_angles(self, landmarks):
        def angle_between_points(p1, p2, p3):
            v1 = np.array([p1.x - p2.x, p1.y - p2.y])
            v2 = np.array([p3.x - p2.x, p3.y - p2.y])
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            return np.degrees(angle)
        
        lm = landmarks.landmark
        angles = {
            'thumb': angle_between_points(lm[1], lm[2], lm[3]),
            'index': angle_between_points(lm[5], lm[6], lm[7]),
            'middle': angle_between_points(lm[9], lm[10], lm[11]),
            'ring': angle_between_points(lm[13], lm[14], lm[15]),
            'pinky': angle_between_points(lm[17], lm[18], lm[19])
        }
        return angles
    
    def refine_prediction(self, frame, yolo_letter, yolo_conf):
        if not self.available or yolo_conf > 0.85:
            return yolo_letter, yolo_conf
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return yolo_letter, yolo_conf
        
        landmarks = results.multi_hand_landmarks[0]
        angles = self.get_hand_angles(landmarks)
        
        if yolo_letter in ['M', 'N']:
            if angles['middle'] < 90:
                return 'M', yolo_conf
            else:
                return 'N', yolo_conf
        
        elif yolo_letter in ['B', 'V']:
            thumb_extended = angles['thumb'] > 120
            if thumb_extended:
                return 'V', yolo_conf
            else:
                return 'B', yolo_conf
        
        elif yolo_letter in ['K', 'P']:
            index_extended = angles['index'] > 140
            if index_extended:
                return 'K', yolo_conf
            else:
                return 'P', yolo_conf
        
        return yolo_letter, yolo_conf
    
    def close(self):
        if self.available:
            self.mp_hands.close()


def get_prediction(frame, model, confidence, device='cpu', use_multiscale=False, use_landmarks=False):
    if use_multiscale:
        detector = MultiScaleDetector(model, scales=[0.8, 1.0, 1.2], device=device)
        predicted_letter, conf, annotated_frame = detector.predict(frame, confidence)
    else:
        results = model(frame, conf=confidence, verbose=False, device=device)
        
        if len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_idx = boxes.conf.argmax()
            class_id = int(boxes.cls[best_idx])
            conf = float(boxes.conf[best_idx])
            predicted_letter = chr(65 + class_id) if class_id < 26 else None
            annotated_frame = results[0].plot()
        else:
            predicted_letter, conf = None, 0.0
            annotated_frame = results[0].plot()
    
    if use_landmarks and predicted_letter:
        landmark_detector = LandmarkGuidedDetector()
        predicted_letter, conf = landmark_detector.refine_prediction(frame, predicted_letter, conf)
        landmark_detector.close()
    
    return predicted_letter, conf, annotated_frame



def check_stable_prediction(target_letter, current_prediction, stability_duration=3.0):
    current_time = time.time()
    
    if current_prediction == target_letter:
        st.session_state.prediction_history.append({
            'letter': current_prediction,
            'time': current_time
        })
        
        if len(st.session_state.prediction_history) > 0:
            stable_start = None
            for pred in st.session_state.prediction_history:
                if pred['letter'] == target_letter:
                    if stable_start is None:
                        stable_start = pred['time']
                else:
                    stable_start = None
            
            if stable_start and (current_time - stable_start) >= stability_duration:
                return True, current_time - stable_start
    else:
        st.session_state.prediction_history.clear()
    
    return False, 0.0


def prediction_mode(model, confidence, ml_settings=None):
    if ml_settings is None:
        ml_settings = {'device': 'cpu', 'use_multiscale': False, 'use_landmarks': False, 'camera_index': 0}
    
    st.title("🔮 Prediction Mode")
    st.markdown("Real-time ASL alphabet detection")
    
    with st.sidebar:
        show_landmarks = st.checkbox("👁️ Show Landmarks", value=False)
        mp_det_conf = 0.5
        mp_track_conf = 0.5
        frame_skip = st.slider("Frame Skip", 1, 10, 1)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📸 Camera Feed")
        video_placeholder = st.empty()
        stats_placeholder = st.empty()
        
        run = st.checkbox("▶️ Start Detection", key="pred_run")
        
        if run:
            cap = cv2.VideoCapture(ml_settings['camera_index'])
            if not cap.isOpened():
                st.error("❌ Cannot access webcam. Please check permissions.")
                return
            
            frame_count = 0
            detection_stats = {"total": 0, "detected": 0, "last_letter": None}
            
            mp_hands = None
            hands = None
            mp_drawing = None
            
            if show_landmarks:
                try:
                    import mediapipe as mp
                    mp_hands = mp.solutions.hands
                    mp_drawing = mp.solutions.drawing_utils
                    hands = mp_hands.Hands(
                        static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=mp_det_conf,
                        min_tracking_confidence=mp_track_conf
                    )
                except ImportError:
                    st.warning("⚠️ MediaPipe not installed. Run: pip install mediapipe")
                    show_landmarks = False
            
            try:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Failed to read frame")
                        break
                    
                    frame_count += 1
                    
                    if frame_count % frame_skip == 0:
                        frame = np.ascontiguousarray(frame)
                        predicted_letter, conf, annotated_frame = get_prediction(
                            frame, model, confidence,
                            device=ml_settings['device'],
                            use_multiscale=ml_settings['use_multiscale'],
                            use_landmarks=ml_settings['use_landmarks']
                        )
                        
                        if show_landmarks and hands:
                            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image_rgb.flags.writeable = False
                            hands_results = hands.process(image_rgb)
                            image_rgb.flags.writeable = True
                            
                            if hands_results and hands_results.multi_hand_landmarks:
                                for hand_landmarks in hands_results.multi_hand_landmarks:
                                    mp_drawing.draw_landmarks(
                                        annotated_frame,
                                        hand_landmarks,
                                        mp_hands.HAND_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(250,44,250), thickness=2, circle_radius=1)
                                    )
                        
                        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        
                        detection_stats["total"] += 1
                        if predicted_letter:
                            detection_stats["detected"] += 1
                            detection_stats["last_letter"] = predicted_letter
                        
                        video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                        
                        with stats_placeholder.container():
                            col_s1, col_s2, col_s3 = st.columns(3)
                            with col_s1:
                                st.metric("Frames Processed", detection_stats["total"])
                            with col_s2:
                                st.metric("Detections", detection_stats["detected"])
                            with col_s3:
                                if detection_stats["last_letter"]:
                                    st.metric("Last Detected", detection_stats["last_letter"])
                    
                    time.sleep(0.01)
            
            finally:
                if hands:
                    hands.close()
                cap.release()
        else:
            placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder_img, "Press 'Start Detection' to begin", 
                       (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            video_placeholder.image(placeholder_img, channels="RGB", use_column_width=True)
    
    with col2:
        st.markdown("### ℹ️ Info")
        
        st.info("""
        **Instructions:**
        1. Click 'Start Detection'
        2. Show ASL letters to camera
        3. Letters A-Z are supported
        """)
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        st.markdown(f"""
        - **Confidence:** {confidence}
        - **Frame Skip:** Every {frame_skip} frames
        - **MediaPipe:** {'✓ Enabled' if show_landmarks else '✗ Disabled'}
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.success("""
        - Good lighting helps
        - Keep hand centered
        - Plain background
        - Clear finger position
        """)


def learning_mode(model, confidence, asl_images, ml_settings=None):
    if ml_settings is None:
        ml_settings = {'device': 'cpu', 'use_multiscale': False, 'use_landmarks': False, 'camera_index': 0}
    st.title("📚 Learning Mode")
    st.markdown("Practice ASL alphabets with real-time feedback")
    
    with st.sidebar:
        order_mode = st.radio("Order", ["Sequential", "Random"], key="learning_order")
        
        if st.button("🔀 Shuffle") and order_mode == "Random":
            random.shuffle(st.session_state.alphabet_order)
            st.session_state.current_index = 0
        
        stability_duration = st.slider("Hold (sec)", 1.0, 5.0, 2.0, 0.5, key="learning_stability")
        auto_advance = st.checkbox("Auto Next", value=True, key="auto_advance")
        
        st.markdown("---")
        st.metric("📍 Current Position", f"{st.session_state.current_index + 1} / {len(st.session_state.alphabet_order)}")
        st.metric("✅ Correct Attempts", st.session_state.correct_count)
    
    current_letter = st.session_state.alphabet_order[st.session_state.current_index]
    
    st.markdown("### 🎮 Navigation")
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 1, 1])
    
    with col_nav1:
        if st.button("⏮️ Previous", disabled=(st.session_state.current_index == 0), use_container_width=True):
            st.session_state.current_index -= 1
            st.session_state.prediction_history.clear()
            st.session_state.correct_count = 0
            st.rerun()
    
    with col_nav2:
        if st.button("⏭️ Next", disabled=(st.session_state.current_index >= len(st.session_state.alphabet_order) - 1), use_container_width=True):
            st.session_state.current_index += 1
            st.session_state.prediction_history.clear()
            st.session_state.correct_count = 0
            st.rerun()
    
    with col_nav3:
        st.metric("Letter", f"{st.session_state.current_index + 1}/{len(st.session_state.alphabet_order)}")
    
    with col_nav4:
        if auto_advance:
            st.success("✓ Auto")
        else:
            st.info("Manual")
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.markdown("### 🎯 Target Letter")
        
        letter_container = st.container()
        with letter_container:
            st.markdown(f"<h1 style='text-align: center; font-size: 120px; color: #4CAF50; margin: 20px 0;'>{current_letter}</h1>", 
                       unsafe_allow_html=True)
        
        image_container = st.container()
        with image_container:
            if current_letter in asl_images:
                st.markdown("#### 👋 ASL Sign Reference")
                asl_img = Image.open(asl_images[current_letter])
                st.image(asl_img, use_column_width=True, caption=f"ASL Sign for '{current_letter}'")
            else:
                st.info(f"📷 Add image: asl_images/{current_letter}.png")
        
        st.markdown("---")
        st.markdown("### 📊 Your Progress")
        progress_percent = (st.session_state.current_index / len(st.session_state.alphabet_order))
        st.progress(progress_percent)
        st.caption(f"{int(progress_percent * 100)}% Complete")
    
    with col1:
        st.markdown("### 📸 Your Practice")
        video_placeholder = st.empty()
        feedback_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        run = st.checkbox("▶️ Start Learning", key="learn_run")
        
        if run:
            cap = cv2.VideoCapture(ml_settings['camera_index'])
            if not cap.isOpened():
                st.error("❌ Cannot access webcam. Please check permissions.")
                return
            
            try:
                target_letter = st.session_state.alphabet_order[st.session_state.current_index]
                
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("⚠️ Failed to read frame")
                        break
                    
                    frame = np.ascontiguousarray(frame)
                    predicted_letter, conf, annotated_frame = get_prediction(
                        frame, model, confidence,
                        device=ml_settings['device'],
                        use_multiscale=ml_settings['use_multiscale'],
                        use_landmarks=ml_settings['use_landmarks']
                    )
                    
                    is_stable, stable_time = check_stable_prediction(
                        target_letter, predicted_letter, stability_duration
                    )
                    
                    # Draw MediaPipe landmarks if enabled
                    if ml_settings.get('use_landmarks', False):
                        try:
                            import mediapipe as mp
                            mp_hands = mp.solutions.hands
                            mp_drawing = mp.solutions.drawing_utils
                            with mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                                              min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                hands_results = hands.process(image_rgb)
                                if hands_results and hands_results.multi_hand_landmarks:
                                    for hand_landmarks in hands_results.multi_hand_landmarks:
                                        mp_drawing.draw_landmarks(
                                            annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(250,44,250), thickness=2, circle_radius=1)
                                        )
                        except ImportError:
                            pass
                    
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    
                    if predicted_letter:
                        color = (0, 255, 0) if predicted_letter == target_letter else (255, 100, 100)
                        cv2.putText(annotated_frame_rgb, f"You: {predicted_letter}", 
                                   (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
                    
                    cv2.putText(annotated_frame_rgb, f"Target: {target_letter}", 
                               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 200, 255), 4)
                    
                    if stable_time > 0:
                        progress = min(stable_time / stability_duration, 1.0) * 100
                        cv2.putText(annotated_frame_rgb, f"Hold: {stable_time:.1f}s ({progress:.0f}%)", 
                                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 100), 3)
                    
                    video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                    
                    if is_stable and time.time() - st.session_state.last_feedback_time > 1.0:
                        feedback_placeholder.success(f"✅ Perfect! You held '{target_letter}' correctly!")
                        st.session_state.correct_count += 1
                        st.session_state.last_feedback_time = time.time()
                        
                        if auto_advance and st.session_state.current_index < len(st.session_state.alphabet_order) - 1:
                            time.sleep(0.2)
                            st.session_state.current_index += 1
                            st.session_state.prediction_history.clear()
                            st.session_state.correct_count = 0
                            cap.release()
                            st.rerun()
                    elif predicted_letter and predicted_letter != target_letter:
                        feedback_placeholder.warning(f"⚠️ You're showing '{predicted_letter}', try '{target_letter}'")
                    elif stable_time > 0:
                        progress_placeholder.info(f"🎯 Keep holding... {stable_time:.1f}s / {stability_duration}s")
                    
                    time.sleep(0.03)
                    
            finally:
                cap.release()
        else:
            placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder_img, "Click 'Start Learning' to begin", 
                       (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            video_placeholder.image(placeholder_img, channels="RGB", use_column_width=True)



def evaluation_mode(model, confidence, asl_images, ml_settings=None):
    if ml_settings is None:
        ml_settings = {'device': 'cpu', 'use_multiscale': False, 'use_landmarks': False, 'camera_index': 0}
    st.title("🎓 Evaluation Mode")
    st.markdown("Test your ASL skills by spelling words")
    
    if not st.session_state.evaluation_words:
        st.session_state.evaluation_words = load_evaluation_words()
    
    with st.sidebar:
        if st.button("🔄 New Words"):
            random.shuffle(st.session_state.evaluation_words)
            st.session_state.word_index = 0
            st.session_state.letter_index = 0
            st.session_state.score = 0
            st.rerun()
        
        stability_duration = st.slider("Hold (sec)", 1.0, 5.0, 2.0, 0.5, key="eval_stability")
        
        st.metric("🏆 Score", st.session_state.score)
    
    if st.session_state.word_index < len(st.session_state.evaluation_words):
        current_word = st.session_state.evaluation_words[st.session_state.word_index]
        
        if st.session_state.letter_index < len(current_word):
            current_letter = current_word[st.session_state.letter_index]
            
            st.markdown("---")
            st.markdown(f"### 📖 Spell This Word: **{current_word}**")
            
            word_progress = ""
            for i, letter in enumerate(current_word):
                if i < st.session_state.letter_index:
                    word_progress += f"✅ **{letter}** "
                elif i == st.session_state.letter_index:
                    word_progress += f"👉 **<span style='color: orange; font-size: 24px;'>{letter}</span>** "
                else:
                    word_progress += f"⬜ {letter} "
            
            st.markdown(word_progress, unsafe_allow_html=True)
            st.caption(f"Letter {st.session_state.letter_index + 1} of {len(current_word)}")
            
            col_skip1, col_skip2 = st.columns([3, 1])
            with col_skip2:
                if st.button("⏭️ Skip Letter", use_container_width=True, help="Skip to next letter (no points)"):
                    st.session_state.letter_index += 1
                    st.session_state.prediction_history.clear()
                    st.warning(f"⚠️ Skipped letter '{current_letter}'")
                    time.sleep(0.5)
                    st.rerun()
            
            st.markdown("---")
            
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.markdown("### 🎯 Target Letter")
                
                st.markdown(f"<h1 style='text-align: center; font-size: 140px; color: #FF5722; margin: 30px 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{current_letter}</h1>", 
                           unsafe_allow_html=True)
                
                st.markdown("---")
                st.markdown("### 📊 Progress")
                progress_percent = st.session_state.letter_index / len(current_word)
                st.progress(progress_percent)
                st.caption(f"Word Progress: {int(progress_percent * 100)}%")
                
                st.metric("✅ Letters Completed", st.session_state.letter_index)
                st.metric("🏆 Current Score", st.session_state.score)
            
            with col1:
                st.markdown("### 📸 Show the Sign")
                video_placeholder = st.empty()
                feedback_placeholder = st.empty()
                
                run = st.checkbox("▶️ Start Evaluation", key="eval_run")
                
                if run:
                    cap = cv2.VideoCapture(ml_settings['camera_index'])
                    if not cap.isOpened():
                        st.error("❌ Cannot access webcam. Please check permissions.")
                        return
                    
                    try:
                        while run:
                            ret, frame = cap.read()
                            if not ret:
                                st.warning("⚠️ Failed to read frame")
                                break
                            
                            frame = np.ascontiguousarray(frame)
                            predicted_letter, conf, annotated_frame = get_prediction(
                                frame, model, confidence,
                                device=ml_settings['device'],
                                use_multiscale=ml_settings['use_multiscale'],
                                use_landmarks=ml_settings['use_landmarks']
                            )
                            
                            is_stable, stable_time = check_stable_prediction(
                                current_letter, predicted_letter, stability_duration
                            )
                            
                            # Draw MediaPipe landmarks if enabled
                            if ml_settings.get('use_landmarks', False):
                                try:
                                    import mediapipe as mp
                                    mp_hands = mp.solutions.hands
                                    mp_drawing = mp.solutions.drawing_utils
                                    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                                                      min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        hands_results = hands.process(image_rgb)
                                        if hands_results and hands_results.multi_hand_landmarks:
                                            for hand_landmarks in hands_results.multi_hand_landmarks:
                                                mp_drawing.draw_landmarks(
                                                    annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=2),
                                                    mp_drawing.DrawingSpec(color=(250,44,250), thickness=2, circle_radius=1)
                                                )
                                except ImportError:
                                    pass
                            
                            annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            
                            if predicted_letter:
                                color = (0, 255, 0) if predicted_letter == current_letter else (255, 100, 100)
                                cv2.putText(annotated_frame_rgb, f"Detected: {predicted_letter}", 
                                           (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 4)
                            
                            cv2.putText(annotated_frame_rgb, f"Spell: {current_letter}", 
                                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 200, 255), 4)
                            
                            if stable_time > 0:
                                progress = min(stable_time / stability_duration, 1.0) * 100
                                cv2.putText(annotated_frame_rgb, f"Hold: {stable_time:.1f}s ({progress:.0f}%)", 
                                           (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 100), 3)
                            
                            video_placeholder.image(annotated_frame_rgb, channels="RGB", use_column_width=True)
                            
                            if is_stable and time.time() - st.session_state.last_feedback_time > 1.0:
                                feedback_placeholder.success(f"✅ Correct! Letter '{current_letter}' recognized!")
                                st.session_state.score += 1
                                st.session_state.letter_index += 1
                                st.session_state.prediction_history.clear()
                                st.session_state.last_feedback_time = time.time()
                                cap.release()
                                time.sleep(0.2)
                                st.rerun()
                            elif predicted_letter and predicted_letter != current_letter:
                                feedback_placeholder.warning(f"❌ Wrong! You showed '{predicted_letter}', need '{current_letter}'")
                            
                            time.sleep(0.03)
                            
                    finally:
                        cap.release()
                else:
                    placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder_img, "Click 'Start Evaluation' to begin", 
                               (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    video_placeholder.image(placeholder_img, channels="RGB", use_column_width=True)
        
        else:
            st.success(f"🎉 Word '{current_word}' completed!")
            st.balloons()
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("➡️ Next Word", use_container_width=True):
                    st.session_state.word_index += 1
                    st.session_state.letter_index = 0
                    st.session_state.prediction_history.clear()
                    st.rerun()
            with col_btn2:
                if st.button("🔄 Restart Evaluation", use_container_width=True):
                    st.session_state.word_index = 0
                    st.session_state.letter_index = 0
                    st.session_state.score = 0
                    st.rerun()
    
    else:
        st.success("🏆 All words completed!")
        st.balloons()
        
        st.markdown(f"""
        <div style='text-align: center; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;'>
            <h1 style='color: white; font-size: 48px; margin-bottom: 20px;'>🏆 Congratulations! 🏆</h1>
            <h2 style='color: white; font-size: 72px; margin: 20px 0;'>{st.session_state.score}</h2>
            <p style='color: white; font-size: 24px;'>Points Earned!</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col_final1, col_final2 = st.columns(2)
        with col_final1:
            if st.button("🔄 Start New Evaluation", use_container_width=True):
                st.session_state.word_index = 0
                st.session_state.letter_index = 0
                st.session_state.score = 0
                st.rerun()
        with col_final2:
            if st.button("🔀 Shuffle & Restart", use_container_width=True):
                random.shuffle(st.session_state.evaluation_words)
                st.session_state.word_index = 0
                st.session_state.letter_index = 0
                st.session_state.score = 0
                st.rerun()


def main():
    st.markdown("""
    <style>
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(100,100,255,0.3);
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #4CAF50;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(20, 25, 35, 0.95);
    }
    .stRadio > label {
        font-weight: 600;
        color: #64B5F6;
    }
    .stCheckbox > label {
        color: #81C784;
    }
    h1 {
        color: #64B5F6;
        font-weight: 700;
    }
    h3 {
        color: #81C784;
        font-weight: 600;
    }
    .stMarkdown {
        color: #E0E0E0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.sidebar.title("🤟 ASL System")
    st.sidebar.markdown("---")
    
    mode = st.sidebar.radio(
        "Mode",
        ["🔮 Prediction", "📚 Learning", "🎓 Evaluation"],
        key="mode_selection"
    )
    
    st.sidebar.markdown("---")
    
    model_path = "output/weights/best.pt"
    confidence = st.sidebar.slider(
        "Confidence",
        0.0, 1.0, 0.50, 0.05
    )
    
    if st.session_state.device is None:
        st.session_state.device = get_device()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Enhancements**")
    
    use_multiscale = st.sidebar.checkbox("🔍 Multi-Scale", value=False)
    use_landmarks = st.sidebar.checkbox("👋 Landmarks", value=False)
    
    if st.session_state.available_cameras is None:
        st.session_state.available_cameras = detect_cameras()
    
    selected_camera = st.sidebar.selectbox(
        "📹 Camera",
        st.session_state.available_cameras,
        format_func=lambda x: f"Cam {x}"
    )
    
    model_file = Path(model_path)
    if not model_file.is_absolute():
        model_file = (Path(__file__).resolve().parent / model_file).resolve()
    
    if not model_file.exists():
        st.error(f"❌ Model not found: {model_path}")
        st.info("💡 Please train a model first or check the path.")
        st.sidebar.markdown("---")
        st.sidebar.error("⚠️ Model not loaded")
        return
    
    try:
        model = load_model(str(model_file))
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return
    
    asl_images = load_asl_images()
    
    ml_settings = {
        'device': st.session_state.device,
        'use_multiscale': use_multiscale,
        'use_landmarks': use_landmarks,
        'camera_index': selected_camera
    }
    
    if mode == "🔮 Prediction":
        prediction_mode(model, confidence, ml_settings)
    elif mode == "📚 Learning":
        learning_mode(model, confidence, asl_images, ml_settings)
    elif mode == "🎓 Evaluation":
        evaluation_mode(model, confidence, asl_images, ml_settings)


if __name__ == "__main__":
    main()
