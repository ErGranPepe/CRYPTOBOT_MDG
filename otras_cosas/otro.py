import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import Button, Label, filedialog, Frame, StringVar, OptionMenu, IntVar, Scale
from PIL import Image, ImageTk
from vpython import canvas, sphere, cylinder, vector, color, mag
import math
import numpy as np

class SkeletonTrackerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detección Avanzada de Esqueleto Humano y Análisis Anatómico")
        self.root.geometry("1200x900")

        # Frame para video
        self.video_frame = Frame(self.root, width=1000, height=700)
        self.video_frame.pack()

        self.label = Label(self.video_frame)
        self.label.pack()

        # Controles
        self.control_frame = Frame(self.root)
        self.control_frame.pack()

        self.start_button = Button(self.control_frame, text="Iniciar Detección", command=self.start_detection, width=20, bg="green", fg="white")
        self.start_button.grid(row=0, column=0, padx=10, pady=10)

        self.stop_button = Button(self.control_frame, text="Detener Detección", command=self.stop_detection, state=tk.DISABLED, width=20, bg="red", fg="white")
        self.stop_button.grid(row=0, column=1, padx=10, pady=10)

        self.save_button = Button(self.control_frame, text="Guardar Captura", command=self.save_frame, state=tk.DISABLED, width=20, bg="blue", fg="white")
        self.save_button.grid(row=0, column=2, padx=10, pady=10)

        # Selección de cámara
        self.camera_index = IntVar(value=0)
        self.camera_menu = OptionMenu(self.control_frame, self.camera_index, *range(5))
        self.camera_menu.grid(row=0, column=3, padx=10, pady=10)

        # Fuente de video
        self.source_var = StringVar(value="Webcam")
        self.source_menu = OptionMenu(self.control_frame, self.source_var, "Webcam", "Archivo de Video")
        self.source_menu.grid(row=0, column=4, padx=10, pady=10)

        # Control de sensibilidad
        self.sensitivity = Scale(self.control_frame, from_=0, to=100, orient=tk.HORIZONTAL, label="Sensibilidad")
        self.sensitivity.set(50)
        self.sensitivity.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # Checkbox para mostrar/ocultar ángulos
        self.show_angles = tk.BooleanVar()
        self.show_angles_checkbox = tk.Checkbutton(self.control_frame, text="Mostrar Ángulos", variable=self.show_angles)
        self.show_angles_checkbox.grid(row=1, column=2, padx=10, pady=10)

        self.cap = None
        self.running = False
        self.current_frame = None

        # Inicialización de MediaPipe Pose y Hands
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils

        # Inicialización de Visualización 3D
        self.canvas = canvas(title="Esqueleto en 3D", width=600, height=600, center=vector(0, 0, 0))
        self.joint_spheres = []
        self.bones = []

    def start_detection(self):
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.NORMAL)

        source = self.source_var.get()
        if source == "Webcam":
            self.cap = cv2.VideoCapture(self.camera_index.get())
        else:
            file_path = filedialog.askopenfilename(filetypes=[("Archivos de Video", "*.mp4;*.avi")])
            if file_path:
                self.cap = cv2.VideoCapture(file_path)
            else:
                self.stop_detection()
                return

        self.detect()

    def stop_detection(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        if self.cap:
            self.cap.release()
        self.label.config(image="")

    def save_frame(self):
        if self.current_frame is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")])
            if file_path:
                cv2.imwrite(file_path, cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2BGR))

    def calculate_angle(self, p1, p2, p3):
        a = np.array([p1.x, p1.y, p1.z])
        b = np.array([p2.x, p2.y, p2.z])
        c = np.array([p3.x, p3.y, p3.z])

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def update_3d_skeleton(self, landmarks):
        if not self.joint_spheres:
            for _ in landmarks:
                sphere_obj = sphere(radius=0.02, color=color.red, pos=vector(0, 0, 0))
                self.joint_spheres.append(sphere_obj)
            for _ in range(len(landmarks) - 1):
                bone = cylinder(radius=0.005, color=color.blue, pos=vector(0, 0, 0), axis=vector(0, 0, 0))
                self.bones.append(bone)

        for i, lm in enumerate(landmarks):
            self.joint_spheres[i].pos = vector(lm.x - 0.5, 0.5 - lm.y, -lm.z * 0.5)

        for i, bone in enumerate(self.bones):
            start = self.joint_spheres[i].pos
            end = self.joint_spheres[i + 1].pos
            bone.pos = start
            bone.axis = end - start

    def detect(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_detection()
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.current_frame = frame_rgb.copy()

        # Ajustar sensibilidad
        min_detection_confidence = self.sensitivity.get() / 100.0
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_detection_confidence)

        # Procesar con MediaPipe Pose
        pose_results = self.pose.process(frame_rgb)

        if pose_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            landmarks = pose_results.pose_landmarks.landmark
            self.update_3d_skeleton(landmarks)

            if self.show_angles.get():
                # Calcular y mostrar ángulos importantes
                angles = {
                    "Codo Izq": self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW],
                                                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]),
                    "Codo Der": self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER],
                                                     landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW],
                                                     landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]),
                    "Rodilla Izq": self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.LEFT_HIP],
                                                        landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE],
                                                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE]),
                    "Rodilla Der": self.calculate_angle(landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP],
                                                        landmarks[self.mp_pose.PoseLandmark.RIGHT_KNEE],
                                                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ANKLE])
                }

                for i, (name, angle) in enumerate(angles.items()):
                    cv2.putText(frame, f"{name}: {int(angle)}°", (10, 30 + i*30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        self.root.after(10, self.detect)

if __name__ == "__main__":
    root = tk.Tk()
    app = SkeletonTrackerApp(root)
    root.mainloop()
