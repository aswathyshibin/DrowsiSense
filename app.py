import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response, jsonify
import time
import threading

app = Flask(__name__)

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Constants for EAR & MAR
EYE_THRESHOLD = 0.21
YAWN_THRESHOLD = 0.55
CONSEC_FRAMES = 15

# Global status for polling
current_status = {"status": "Normal", "ear": 0, "mar": 0}

class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.counter = 0

    def __del__(self):
        self.video.release()

    def get_frame(self):
        global current_status
        success, frame = self.video.read()
        if not success:
            return None

        # Flip for mirrored effect
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        status = "Normal"
        ear = 0
        mar = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                landmarks = np.array([(lm.x * w, lm.y * h) for lm in face_landmarks.landmark])

                # Indices for eyes and mouth
                # Inner eye contours are more stable for EAR
                left_eye_idx = [33, 160, 158, 133, 153, 144]
                right_eye_idx = [362, 385, 387, 263, 373, 380]
                mouth_idx = [13, 14, 78, 308]

                def calculate_ear(eye_points):
                    p2_p6 = np.linalg.norm(eye_points[1] - eye_points[5])
                    p3_p5 = np.linalg.norm(eye_points[2] - eye_points[4])
                    p1_p4 = np.linalg.norm(eye_points[0] - eye_points[3])
                    return (p2_p6 + p3_p5) / (2.0 * p1_p4) if p1_p4 != 0 else 0

                def calculate_mar(mouth_points):
                    v_dist = np.linalg.norm(mouth_points[0] - mouth_points[1])
                    h_dist = np.linalg.norm(mouth_points[2] - mouth_points[3])
                    return v_dist / h_dist if h_dist != 0 else 0

                ear_left = calculate_ear(landmarks[left_eye_idx])
                ear_right = calculate_ear(landmarks[right_eye_idx])
                ear = (ear_left + ear_right) / 2.0
                mar = calculate_mar(landmarks[mouth_idx])

                # Logic Check
                if ear < EYE_THRESHOLD:
                    self.counter += 1
                    if self.counter >= CONSEC_FRAMES:
                        status = "Drowsy"
                elif mar > YAWN_THRESHOLD:
                    status = "Yawning"
                    self.counter = 0
                else:
                    self.counter = 0
                    status = "Normal"

                # Update global status
                current_status = {"status": status, "ear": round(ear, 3), "mar": round(mar, 3)}

                # Visual Feedback
                color = (0, 255, 0)
                if status == "Drowsy": color = (0, 0, 255)
                elif status == "Yawning": color = (0, 255, 255)

                cv2.putText(frame, f"STATUS: {status}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/detect")
def detect():
    return render_template("detect.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/status")
def get_status():
    return jsonify(current_status)

@app.route("/contact")
def contact():
    return render_template("contact.html")

if __name__ == "__main__":
    app.run(debug=True, threaded=True)