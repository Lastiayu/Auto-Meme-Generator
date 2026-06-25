import cv2
import os
import random
import time
import pygame
import streamlit as st
from deepface import DeepFace

# === KONFIGURASI HALAMAN STREAMLIT ===
st.set_page_config(page_title="AI Auto Meme", page_icon="🎭", layout="wide")
st.title("🎭 Auto Meme Generator")
st.markdown("Sistem Deteksi Emosi Real-time & Generator Meme Otomatis")
st.markdown("---")

# === INISIALISASI AUDIO & ASSETS ===
@st.cache_resource # Memori cache agar aset tidak di-load ulang setiap refresh web
def load_assets():
    pygame.mixer.init()
    memes = {"Happy": [], "Sad": [], "Neutral": []}
    sounds = {"Happy": [], "Sad": [], "Neutral": []}
    
    for emotion, folder in {"Happy": "assets/memes/happy", "Sad": "assets/memes/sad", "Neutral": "assets/memes/neutral"}.items():
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    memes[emotion].append(cv2.imread(os.path.join(folder, f)))
                    
    for emotion, folder in {"Happy": "assets/sounds/happy", "Sad": "assets/sounds/sad", "Neutral": "assets/sounds/neutral"}.items():
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(('.mp3', '.wav')):
                    sounds[emotion].append(os.path.join(folder, f))
    return memes, sounds

memes, sounds = load_assets()

def put_meme_left_of_face(frame, meme_img, x, y, w, h):
    if meme_img is None: return frame
    meme_h, meme_w = meme_img.shape[:2]
    new_w = int((h / meme_h) * meme_w)
    max_available_w = x - 20
    if max_available_w <= 10: return frame 
    if new_w > max_available_w: new_w = max_available_w
    resized_meme = cv2.resize(meme_img, (new_w, h))
    meme_x = x - new_w - 20
    frame[y:y+h, meme_x:meme_x+new_w] = resized_meme
    return frame

# === ANTARMUKA WEB ===
col1, col2 = st.columns([3, 1])

with col2:
    st.header("Kontrol Panel")
    run_camera = st.checkbox("Nyalakan Kamera 🎥", value=False)
    reset_btn = st.button("Reset Scanning 🔄")
    
    st.markdown("### Status Sistem")
    status_text = st.empty() # Placeholder untuk teks status

with col1:
    frame_window = st.image([]) # Placeholder untuk layar video utama

# === LOGIKA COMPUTER VISION ===
if run_camera:
    # Karena kita pakai Streamlit, model Cascade dimuat di sini
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    
    current_state = "SCANNING"
    current_emotion = "Neutral"
    active_meme = None
    processing_start_time = 0
    processing_duration = 2.0
    
    if reset_btn:
        current_state = "SCANNING"
        pygame.mixer.music.stop()

    while run_camera:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal mengakses kamera.")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        current_time = time.time()

        for (x, y, w, h) in faces:
            if current_state == "SCANNING":
                status_text.info("Memindai Wajah...")
                theme_color = (0, 255, 255) # Yellow di BGR, Cyan di RGB
                cv2.rectangle(frame, (x, y), (x+w, y+h), theme_color, 2)
                
                try:
                    face_roi = frame[y:y+h, x:x+w]
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    raw_emotion = result[0]['dominant_emotion']
                    
                    if raw_emotion == 'happy': current_emotion = "Happy"
                    elif raw_emotion in ['sad', 'angry', 'fear', 'disgust']: current_emotion = "Sad"
                    else: current_emotion = "Neutral"
                    
                    current_state = "PROCESSING"
                    processing_start_time = current_time
                except Exception:
                    pass

            elif current_state == "PROCESSING":
                status_text.warning("Memproses Emosi AI...")
                theme_color = (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), theme_color, 2)
                
                progress = (current_time - processing_start_time) / processing_duration
                bar_width = int(w * progress)
                cv2.rectangle(frame, (x, y-30), (x + bar_width, y-25), theme_color, -1)

                if current_time - processing_start_time >= processing_duration:
                    if len(memes[current_emotion]) > 0:
                        active_meme = random.choice(memes[current_emotion])
                    if len(sounds[current_emotion]) > 0:
                        active_sound = random.choice(sounds[current_emotion])
                        pygame.mixer.music.load(active_sound)
                        pygame.mixer.music.play()
                    current_state = "RESULT"

            elif current_state == "RESULT":
                if current_emotion == "Happy": 
                    theme_color = (0, 255, 0)
                    status_text.success(f"Hasil: {current_emotion} 😊")
                elif current_emotion == "Sad": 
                    theme_color = (0, 0, 255)
                    status_text.error(f"Hasil: {current_emotion} 😢")
                else: 
                    theme_color = (255, 255, 255)
                    status_text.info(f"Hasil: {current_emotion} 😐")

                cv2.rectangle(frame, (x, y), (x+w, y+h), theme_color, 2)
                frame = put_meme_left_of_face(frame, active_meme, x, y, w, h)

        # Konversi BGR (OpenCV) ke RGB (Streamlit/Web)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_window.image(frame_rgb)

    cap.release()
else:
    st.info("Silakan centang 'Nyalakan Kamera' di panel sebelah kanan untuk memulai.")
    pygame.mixer.music.stop()

    # env\Scripts\activate
    # streamlit run app.py