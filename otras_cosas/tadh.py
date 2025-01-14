import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox, ttk
from threading import Thread
import numpy as np
import pyttsx3
from PIL import Image, ImageTk

# Configuración de Mediapipe para detección de pose y manos
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Configuración de texto a voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Velocidad del habla
engine.setProperty('volume', 1)  # Volumen del habla

# Variables globales para manejar la cámara
cap = None
running = False
selected_camera = 0
last_feedback = []
calibration_data = None

# Función para analizar la postura y la atención
def analyze_pose(frame, camera_position):
    global last_feedback, calibration_data
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(image_rgb)
    results_hands = hands.process(image_rgb)

    feedback = []

    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Verificar postura de la espalda (hombros y cadera)
        landmarks = results_pose.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        if calibration_data:
            ideal_nose_y = calibration_data['nose_y']
            shoulder_slope = abs(left_shoulder.y - right_shoulder.y)
            hip_slope = abs(left_hip.y - right_hip.y)

            if shoulder_slope > 0.05 or hip_slope > 0.05:
                feedback.append("Tu espalda está un poco torcida, trata de enderezarla.")

            if camera_position == "frontal" and abs(nose.y - ideal_nose_y) > 0.1:
                feedback.append("Parece que te inclinas mucho. Vuelve a tu posición ideal.")

    # Verificar si se sostiene un bolígrafo (manos)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Verificar proximidad del pulgar y el índice (como si sostuviera algo)
            distance_thumb_index = np.linalg.norm(
                np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_tip.x, index_tip.y])
            )
            if distance_thumb_index < 0.05:  # Umbral ajustable
                feedback.append("¡Muy bien! Parece que estás sosteniendo tu bolígrafo.")
            else:
                feedback.append("Recuerda sostener el bolígrafo correctamente.")

    # Comparar feedback nuevo con el último mostrado para evitar repetición
    if feedback != last_feedback:
        last_feedback = feedback
        for i, msg in enumerate(feedback):
            cv2.putText(frame, msg, (10, 30 * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            speak_message(msg)

    return frame

# Función para síntesis de voz
def speak_message(message):
    engine.say(message)
    engine.runAndWait()

# Función para capturar la cámara
def start_camera(camera_position):
    global cap, running
    cap = cv2.VideoCapture(selected_camera)

    if not cap.isOpened():
        messagebox.showerror("Error", "No se pudo acceder a la cámara.")
        return

    running = True

    while running:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "No se pudo leer el frame de la cámara.")
            break

        frame = cv2.resize(frame, (640, 480))
        frame = analyze_pose(frame, camera_position)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)

        canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        canvas.image = imgtk

    cap.release()

# Función para detener la cámara
def stop_camera():
    global running
    running = False

# Función para calibrar postura inicial
def calibrate_posture():
    global cap, calibration_data
    if not cap or not cap.isOpened():
        messagebox.showerror("Error", "Inicia la cámara antes de calibrar.")
        return

    messagebox.showinfo("Calibración", "Adopta tu postura ideal y presiona OK para calibrar.")

    ret, frame = cap.read()
    if not ret:
        messagebox.showerror("Error", "No se pudo leer el frame de la cámara para calibrar.")
        return

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(image_rgb)

    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark
        calibration_data = {
            'nose_y': landmarks[mp_pose.PoseLandmark.NOSE.value].y
        }
        messagebox.showinfo("Calibración", "Postura calibrada con éxito.")
    else:
        messagebox.showerror("Error", "No se detectó una postura. Intenta nuevamente.")

# Interfaz gráfica (Tkinter)
def start_interface():
    def on_start():
        if not running:
            Thread(target=start_camera, args=(camera_position.get(),)).start()

    def on_stop():
        stop_camera()
        messagebox.showinfo("Info", "Cámara detenida")

    def on_calibrate():
        calibrate_posture()

    def select_camera(event):
        global selected_camera
        selected_camera = int(camera_combo.get().split()[1])

    window = tk.Tk()
    window.title("TDAH Focus Helper")

    # Lista de cámaras disponibles
    cameras = []
    for i in range(5):
        cap_test = cv2.VideoCapture(i)
        if cap_test.isOpened():
            cameras.append(f"Cámara {i}")
            cap_test.release()

    tk.Label(window, text="Selecciona una cámara:", font=("Arial", 14)).pack(pady=5)
    camera_combo = ttk.Combobox(window, values=cameras, state="readonly", font=("Arial", 12))
    camera_combo.pack(pady=5)
    camera_combo.bind("<<ComboboxSelected>>", select_camera)
    camera_combo.current(0)  # Seleccionar la primera cámara por defecto

    tk.Label(window, text="Posición de la cámara:", font=("Arial", 14)).pack(pady=5)
    camera_position = ttk.Combobox(window, values=["frontal", "lateral"], state="readonly", font=("Arial", 12))
    camera_position.pack(pady=5)
    camera_position.current(0)  # Seleccionar "frontal" por defecto

    global canvas
    canvas = tk.Canvas(window, width=640, height=480, bg="black")
    canvas.pack(pady=10)

    start_button = tk.Button(window, text="Iniciar", command=on_start, width=20, height=2, bg="green", fg="white", font=("Arial", 12, "bold"))
    start_button.pack(pady=10)

    calibrate_button = tk.Button(window, text="Calibrar", command=on_calibrate, width=20, height=2, bg="blue", fg="white", font=("Arial", 12, "bold"))
    calibrate_button.pack(pady=10)

    stop_button = tk.Button(window, text="Detener", command=on_stop, width=20, height=2, bg="red", fg="white", font=("Arial", 12, "bold"))
    stop_button.pack(pady=10)

    window.mainloop()

if __name__ == "__main__":
    start_interface()
