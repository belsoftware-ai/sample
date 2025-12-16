# face_recognition_utils.py

import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis

# Initialize face analysis model
face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
face_app.prepare(ctx_id=0)

def load_face_database(path="face_db.pkl"):
    try:
        with open(path, "rb") as f:
            face_db = pickle.load(f)
            if isinstance(face_db, list):
                # Convert list to dict
                face_db = {name: emb for name, emb in face_db}
            print("[INFO] Face database loaded.")
            return face_db
    except FileNotFoundError:
        print("[WARNING] face_db.pkl not found. Using empty database.")
        return {}

def save_face_database(face_db, path="face_db.pkl"):
    with open(path, "wb") as f:
        pickle.dump(face_db, f)

def get_embedding(face_img):
    faces = face_app.get(face_img)
    if not faces:
        return None
    return faces[0].embedding

def recognize_face(embedding, face_db, threshold=0.5):
    best_score = -1
    best_name = "Unknown"
    for name, db_emb in face_db.items():
        sim = np.dot(embedding, db_emb) / (np.linalg.norm(embedding) * np.linalg.norm(db_emb))
        if sim > best_score and sim > threshold:
            best_score = sim
            best_name = name
    return best_name, best_score

def process_frame(frame, face_db, camera_type="entry"):
    faces = face_app.get(frame)
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        face_img = frame[y1:y2, x1:x2]
        embedding = face.embedding
        name, score = recognize_face(embedding, face_db)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

