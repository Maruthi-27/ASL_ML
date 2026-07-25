import cv2
import mediapipe as mp
import numpy as np
import pickle
import pyttsx3
import time
import threading
import customtkinter as ctk 
from PIL import Image, ImageTk
from collections import Counter

# --------------------------------------
# 1. SETTINGS (STRICT MODE)
# --------------------------------------
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

MODEL_PATH = "svm_mediapipe.pkl_1"
BUFFER_SIZE = 15             # Increased to 15 (Super Smooth)
CONFIDENCE_THRESHOLD = 0.80  
TYPING_COOLDOWN = 1.0        # Increased Cooldown to 1.0s
REPEAT_DELAY = 2.5           # Must hold for 2.5s to repeat a letter

# --------------------------------------
# 2. LOAD RESOURCES
# --------------------------------------
try:
    with open(MODEL_PATH, "rb") as f:
        svm_model = pickle.load(f)
    print("✅ Model loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: '{MODEL_PATH}' not found.")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# --------------------------------------
# 3. CORE FUNCTIONS
# --------------------------------------
def get_landmarks(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    if not results.multi_hand_landmarks:
        return None, frame
    hand_landmarks = results.multi_hand_landmarks[0]
    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    data = []
    for lm in hand_landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return np.array(data).reshape(1, -1), frame

def text_to_speech(text):
    def _speak():
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}") 
    threading.Thread(target=_speak, daemon=True).start()

# --------------------------------------
# 4. MODERN UI APP
# --------------------------------------
class ModernASLApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("NeuroSign | Strict Mode")
        self.geometry("1200x800")
        
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.cap = cv2.VideoCapture(0)
        self.curr_text = ""
        self.history = []
        
        # LOGIC VARIABLES
        self.last_valid_pred = "NONE" # Start with NONE so first letter always works
        self.gesture_start_time = 0
        self.last_typed_time = 0

        self.setup_ui()
        self.update_video()

        self.bind("<Return>", lambda event: self.trigger_speak())
        self.bind("<BackSpace>", lambda event: self.trigger_delete())
        self.bind("<space>", lambda event: self.trigger_space())
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # VIDEO SECTION
        self.frame_video = ctk.CTkFrame(self, corner_radius=20, fg_color="#101010")
        self.frame_video.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.lbl_video = ctk.CTkLabel(self.frame_video, text="")
        self.lbl_video.pack(expand=True, fill="both", padx=10, pady=10)

        # CONTROLS SECTION
        self.frame_controls = ctk.CTkFrame(self, corner_radius=20, fg_color="#2b2b2b")
        self.frame_controls.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")

        ctk.CTkLabel(self.frame_controls, text="DETECTED SIGN", 
                     font=("Segoe UI", 12, "bold"), text_color="#aaaaaa").pack(pady=(30, 5))
        
        self.lbl_pred = ctk.CTkLabel(self.frame_controls, text="-", 
                                     font=("Segoe UI", 80, "bold"), text_color="#4CAF50")
        self.lbl_pred.pack(pady=10)

        self.lbl_status = ctk.CTkLabel(self.frame_controls, text="System Ready", 
                                       font=("Segoe UI", 14), text_color="gray")
        self.lbl_status.pack(pady=(0, 20))

        ctk.CTkLabel(self.frame_controls, text="TRANSLATION", 
                     font=("Segoe UI", 12, "bold"), text_color="#aaaaaa").pack(pady=(10, 5))
        
        self.txt_output = ctk.CTkTextbox(self.frame_controls, height=150, font=("Consolas", 20))
        self.txt_output.pack(padx=20, pady=5, fill="x")

        self.frame_btns = ctk.CTkFrame(self.frame_controls, fg_color="transparent")
        self.frame_btns.pack(padx=20, pady=30, fill="x")

        self.btn_speak = ctk.CTkButton(self.frame_btns, text="🔊 SPEAK", command=self.trigger_speak,
                                       height=50, font=("Segoe UI", 14, "bold"),
                                       fg_color="#2CC985", hover_color="#229A65")
        self.btn_speak.pack(fill="x", pady=8)

        self.btn_space = ctk.CTkButton(self.frame_btns, text="␣ SPACE", command=self.trigger_space,
                                       height=40, font=("Segoe UI", 12, "bold"),
                                       fg_color="#3B8ED0", hover_color="#2C6E9F")
        self.btn_space.pack(fill="x", pady=8)

        self.btn_clear = ctk.CTkButton(self.frame_btns, text="⌫ DELETE", command=self.trigger_delete,
                                       height=40, font=("Segoe UI", 12, "bold"),
                                       fg_color="#C53A3A", hover_color="#942B2B")
        self.btn_clear.pack(fill="x", pady=8)

    # --------------------------------------
    # STRICT LOGIC TO PREVENT "CATT"
    # --------------------------------------
    def update_video(self):
        ret, frame = self.cap.read()
        if not ret:
            self.after(10, self.update_video)
            return

        features, frame = get_landmarks(frame)

        if features is not None:
            raw_pred = svm_model.predict(features)[0]
            
            # 1. Fill Buffer
            self.history.append(raw_pred)
            if len(self.history) > BUFFER_SIZE:
                self.history.pop(0)

            # 2. Check Consistency
            counts = Counter(self.history)
            top_pred, freq = counts.most_common(1)[0]
            
            try:
                probs = svm_model.predict_proba(features)[0]
                conf = np.max(probs)
            except:
                conf = 1.0

            self.lbl_status.configure(text=f"Seeing: {top_pred} ({int(conf*100)}%)")

            # 3. Only Process if SUPER STABLE
            if freq >= (BUFFER_SIZE - 2) and conf > CONFIDENCE_THRESHOLD:
                self.process_gesture(top_pred)
            else:
                self.lbl_pred.configure(text_color="gray")

        # UI Update
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(800, 600))
        self.lbl_video.configure(image=ctk_img)
        self.lbl_video.image = ctk_img

        self.after(10, self.update_video)

    def process_gesture(self, pred):
        now = time.time()
        self.lbl_pred.configure(text=pred, text_color="#4CAF50")

        # LOGIC CHANGE: 
        # If the detected sign is the SAME as the last valid one,
        # we IGNORE IT completely unless it's held for REPEAT_DELAY.
        # This prevents "CATT" because the second T is ignored.

        if pred == self.last_valid_pred:
            # ONLY type if held for 2.5 seconds (Intentional Repeat)
            if (now - self.gesture_start_time) > REPEAT_DELAY:
                self.type_character(pred)
                self.gesture_start_time = now # Reset timer
                self.lbl_pred.configure(text=f"{pred} (x2)", text_color="#FF9800")
            else:
                # Do nothing, just waiting
                pass

        elif pred != self.last_valid_pred:
            # It's a DIFFERENT letter. Check Cooldown.
            if (now - self.last_typed_time) > TYPING_COOLDOWN:
                self.type_character(pred)
                self.last_valid_pred = pred
                self.last_typed_time = now
                self.gesture_start_time = now

    def type_character(self, char):
        clean = char.strip().upper()
        if clean == "SPACE":
            self.curr_text += " "
        elif clean == "DELETE":
            self.curr_text = self.curr_text[:-1]
        else:
            self.curr_text += clean
        self.update_textbox()

    def update_textbox(self):
        self.txt_output.delete("0.0", "end")
        self.txt_output.insert("0.0", self.curr_text)

    def trigger_speak(self):
        if self.curr_text:
            self.lbl_pred.configure(text="🔊", text_color="#2196F3")
            text_to_speech(self.curr_text)

    def trigger_space(self):
        self.curr_text += " "
        self.update_textbox()

    def trigger_delete(self):
        self.curr_text = self.curr_text[:-1]
        self.update_textbox()

    def on_close(self):
        self.cap.release()
        self.destroy()
        import os; os._exit(0)

if __name__ == "__main__":
    app = ModernASLApp()
    app.mainloop()