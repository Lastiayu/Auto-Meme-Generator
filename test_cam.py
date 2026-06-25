import cv2
import os
import random
import time
import pygame
from deepface import DeepFace

# 1. INISIALISASI AUDIO ENGINE
pygame.mixer.init()

# 2. DYNAMIC PRE-LOADING ASSETS (Meme & Audio)
meme_folders = {
    "Happy": "assets/memes/happy",
    "Sad": "assets/memes/sad",
    "Neutral": "assets/memes/neutral"
}
sound_folders = {
    "Happy": "assets/sounds/happy",
    "Sad": "assets/sounds/sad",
    "Neutral": "assets/sounds/neutral"
}

memes = {"Happy": [], "Sad": [], "Neutral": []}
sounds = {"Happy": [], "Sad": [], "Neutral": []}

print("Memuat aset multimedia ke memori...")

# Load Memes
for emotion, folder in meme_folders.items():
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    memes[emotion].append(img)
        print(f"-> {emotion} Memes: {len(memes[emotion])} dimuat.")

# Load Sounds
for emotion, folder in sound_folders.items():
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            if filename.endswith(('.mp3', '.wav')):
                sounds[emotion].append(os.path.join(folder, filename))
        print(f"-> {emotion} Sounds: {len(sounds[emotion])} dimuat.")

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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

current_state = "SCANNING" 
current_emotion = "Neutral"
active_meme = None
processing_start_time = 0
processing_duration = 2.0 

print("\nSistem Siap. Memulai dalam Mode SCANNING...")

while True:
    ret, frame = cap.read()
    if not ret: break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    current_time = time.time()

    for (x, y, w, h) in faces:
        if current_state == "SCANNING":
            theme_color = (255, 255, 0)
            cv2.rectangle(frame, (x, y), (x+w, y+h), theme_color, 2)
            cv2.putText(frame, "MEMINDAI EKSPRESI...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, theme_color, 2)
            
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
            theme_color = (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), theme_color, 2)
            dots = "." * (int(current_time * 5) % 4)
            cv2.putText(frame, f"MEMPROSES DATA{dots}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, theme_color, 2)
            
            progress = (current_time - processing_start_time) / processing_duration
            bar_width = int(w * progress)
            cv2.rectangle(frame, (x, y-30), (x + bar_width, y-25), theme_color, -1)

            if current_time - processing_start_time >= processing_duration:
                # Acak Gambar Meme
                if len(memes[current_emotion]) > 0:
                    active_meme = random.choice(memes[current_emotion])
                
                # Acak Audio
                if len(sounds[current_emotion]) > 0:
                    active_sound = random.choice(sounds[current_emotion])
                    pygame.mixer.music.load(active_sound)
                    pygame.mixer.music.play()

                current_state = "RESULT"

        elif current_state == "RESULT":
            if current_emotion == "Happy": theme_color = (0, 255, 0)
            elif current_emotion == "Sad": theme_color = (0, 0, 255)
            else: theme_color = (255, 255, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), theme_color, 2)
            cv2.putText(frame, f"Hasil: {current_emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, theme_color, 2)
            cv2.putText(frame, "Tekan 'R' untuk Scan Ulang", (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            frame = put_meme_left_of_face(frame, active_meme, x, y, w, h)

    cv2.imshow('AI Auto Meme Generator - FSM Mode', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r') or key == ord('R'):
        if current_state == "RESULT":
            pygame.mixer.music.stop() 
            current_state = "SCANNING"
            current_emotion = "Neutral"
            active_meme = None

cap.release()
cv2.destroyAllWindows()