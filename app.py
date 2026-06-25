import cv2
import os
import random
import numpy as np
import streamlit as st
from deepface import DeepFace

# === KONFIGURASI HALAMAN STREAMLIT ===
st.set_page_config(page_title="AI Auto Meme", page_icon="🎭", layout="centered")
st.title("🎭 Auto Meme Generator")
st.markdown("Sistem Deteksi Emosi & Generator Meme (Cloud Version)")
st.markdown("---")

# === INISIALISASI ASSETS ===
@st.cache_resource 
def load_assets():
    memes = {"Happy": [], "Sad": [], "Neutral": []}
    sounds = {"Happy": [], "Sad": [], "Neutral": []}
    
    # Load Memes
    for emotion, folder in {"Happy": "assets/memes/happy", "Sad": "assets/memes/sad", "Neutral": "assets/memes/neutral"}.items():
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if f.endswith(('.png', '.jpg', '.jpeg')):
                    memes[emotion].append(cv2.imread(os.path.join(folder, f)))
                    
    # Load Sounds
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

# === ANTARMUKA WEB CLOUD ===
st.info("💡 Ambil foto wajahmu! Sistem akan mengunci ekspresimu dan men-generate meme.")

# Widget Kamera bawaan Streamlit (Aman untuk Cloud & Mobile)
picture = st.camera_input("Ambil Foto Reaksi")

if picture is not None:
    with st.spinner("Memproses Emosi AI..."):
        # Konversi gambar dari browser ke format OpenCV
        bytes_data = picture.getvalue()
        frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        
        current_emotion = "Neutral"
        active_meme = None
        active_sound = None

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                try:
                    face_roi = frame[y:y+h, x:x+w]
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    raw_emotion = result[0]['dominant_emotion']
                    
                    if raw_emotion == 'happy': current_emotion = "Happy"
                    elif raw_emotion in ['sad', 'angry', 'fear', 'disgust']: current_emotion = "Sad"
                    else: current_emotion = "Neutral"
                except Exception:
                    pass

                # Styling dan Overlay Meme
                if current_emotion == "Happy": theme_color = (0, 255, 0)
                elif current_emotion == "Sad": theme_color = (0, 0, 255)
                else: theme_color = (255, 255, 0)

                cv2.rectangle(frame, (x, y), (x+w, y+h), theme_color, 2)
                cv2.putText(frame, f"Hasil: {current_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, theme_color, 2)
                
                if len(memes[current_emotion]) > 0:
                    active_meme = random.choice(memes[current_emotion])
                if len(sounds[current_emotion]) > 0:
                    active_sound = random.choice(sounds[current_emotion])
                    
                frame = put_meme_left_of_face(frame, active_meme, x, y, w, h)
            
            # Konversi BGR (OpenCV) ke RGB (Streamlit/Web)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Tampilkan Hasil Visual
            st.success(f"Analisis Selesai! Ekspresi terdeteksi: **{current_emotion}**")
            st.image(frame_rgb, use_column_width=True)
            
            # Tampilkan dan Putar Audio (Aman untuk Browser)
            if active_sound:
                st.audio(active_sound, format="audio/mp3", autoplay=True)
        else:
            st.warning("Wajah tidak terdeteksi. Silakan ambil foto ulang di tempat yang lebih terang.")

    # env\Scripts\activate
    # streamlit run app.py