"""Face scanning, enrollment and recognition (see implementation plan sections 1-2).

CLI:
    python face.py enroll "Alice" [role]
    python face.py recognize
    python face.py compare a.jpg b.jpg     # cosine score between two photos, for threshold tuning
"""
import os
import sys
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import db

MODEL = "buffalo_l"  # RetinaFace detector + ArcFace 512-d embedding
THRESHOLD = 0.40  # cosine similarity; tune on your own photos with `compare`
MIN_FACE_PX = 80
MIN_SHARPNESS = 60.0  # variance of Laplacian on the face crop
BRIGHTNESS = (50, 210)  # mean gray level of the face crop
MIN_YAW_RANGE = 0.30  # liveness: how far the head must turn left<->right
FRAME_SKIP = 2  # process every Nth frame to stay fast on CPU
# An iPhone with Continuity Camera takes index 0 and moves the Mac camera to 1; set CAMERA_INDEX=1 then.
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", 0))


def load_model():
    app = FaceAnalysis(name=MODEL, allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def open_camera(index=CAMERA_INDEX):
    cam = cv2.VideoCapture(index)
    if not cam.isOpened() and index != 0:
        cam = cv2.VideoCapture(0)  # the iPhone isn't connected, so the Mac camera is back at 0
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cam.isOpened():
        raise RuntimeError("cannot open camera (check macOS camera permission for your terminal)")
    return cam


def largest_face(app, frame):
    faces = app.get(frame)
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), default=None)


def quality_ok(frame, face):
    x1, y1, x2, y2 = face.bbox.astype(int)
    if min(x2 - x1, y2 - y1) < MIN_FACE_PX or face.det_score < 0.6:
        return False
    crop = cv2.cvtColor(frame[max(y1, 0):y2, max(x1, 0):x2], cv2.COLOR_BGR2GRAY)
    if crop.size == 0:
        return False
    return cv2.Laplacian(crop, cv2.CV_64F).var() >= MIN_SHARPNESS and BRIGHTNESS[0] <= crop.mean() <= BRIGHTNESS[1]


def yaw(face):
    """Head-turn proxy from the 5 keypoints: nose offset from eye midpoint, relative to eye distance."""
    le, re, nose = face.kps[0], face.kps[1], face.kps[2]
    return (nose[0] - (le[0] + re[0]) / 2) / max(re[0] - le[0], 1e-6)


def mean_embedding(embs):
    v = np.mean(embs, axis=0)
    return v / np.linalg.norm(v)


def match(emb, matrix, threshold=THRESHOLD):
    """Best cosine match of a unit vector against unit-vector rows. Returns (row index or None, score)."""
    if len(matrix) == 0:
        return None, 0.0
    scores = matrix @ emb
    i = int(np.argmax(scores))
    return (i if scores[i] >= threshold else None), float(scores[i])


def scan(app, cam, want, need_liveness, timeout=10.0, show=None, space=0.0):
    """Read frames until `want` good embeddings (and liveness, if asked) are collected.

    Alignment happens inside insightface: the recognition model warps each face to the
    standard ArcFace pose using the eye/nose/mouth keypoints before embedding.
    """
    embs, yaws, n, last = [], [], 0, 0.0
    deadline = time.time() + timeout
    while time.time() < deadline:
        ok, frame = cam.read()
        if not ok:
            continue
        n += 1
        if n % FRAME_SKIP:
            continue
        face = largest_face(app, frame)
        good = face is not None and quality_ok(frame, face)
        if good:
            yaws.append(yaw(face))
            if time.time() - last >= space:
                embs.append(face.normed_embedding)
                last = time.time()
        live = not need_liveness or (len(yaws) > 1 and max(yaws) - min(yaws) >= MIN_YAW_RANGE)
        if show:
            if face is not None:
                x1, y1, x2, y2 = face.bbox.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0) if good else (0, 0, 255), 2)
            msg = f"{len(embs)}/{want} " + ("live" if live else "turn head left and right")
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            show(frame)
        if len(embs) >= want and live:
            return embs, True
    return embs, not need_liveness or (len(yaws) > 1 and max(yaws) - min(yaws) >= MIN_YAW_RANGE)


class FaceModule:
    def __init__(self, db_path="door.db"):
        self.app = load_model()
        self.conn = db.connect(db_path)
        self.reload()

    def reload(self):
        self.ids, self.names, self.matrix = db.load_embeddings(self.conn, MODEL)

    def enroll(self, name, role="user", shots=8, show=None):
        cam = open_camera()
        try:
            embs, _ = scan(self.app, cam, shots, need_liveness=False, timeout=30, show=show, space=0.4)
        finally:
            cam.release()
        if len(embs) < 5:
            raise RuntimeError(f"only {len(embs)} good face images captured, need at least 5")
        user_id = db.add_user(self.conn, name, role)
        db.add_embedding(self.conn, user_id, mean_embedding(embs), MODEL)
        self.reload()
        return user_id

    def recognize(self, show=None):
        cam = open_camera()
        try:
            embs, live = scan(self.app, cam, 3, need_liveness=True, show=show)
        finally:
            cam.release()
        if not embs:
            result = {"result": "no_face", "user_id": None, "name": None, "score": 0.0}
        elif not live:
            result = {"result": "spoof_suspected", "user_id": None, "name": None, "score": 0.0}
        else:
            i, score = match(mean_embedding(embs), self.matrix)
            result = {"result": "granted" if i is not None else "unknown",
                      "user_id": self.ids[i] if i is not None else None,
                      "name": self.names[i] if i is not None else None,
                      "score": round(score, 4)}
        db.log_access(self.conn, result["user_id"], result["result"], result["score"])
        return result


def window(frame):
    cv2.imshow("face", frame)
    cv2.waitKey(1)


def compare(app, a, b):
    fa, fb = (largest_face(app, cv2.imread(p)) for p in (a, b))
    if fa is None or fb is None:
        raise SystemExit("no face found in one of the images")
    return float(fa.normed_embedding @ fb.normed_embedding)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "enroll" and len(sys.argv) >= 3:
        print("user_id:", FaceModule().enroll(sys.argv[2], *sys.argv[3:4], show=window))
    elif cmd == "recognize":
        print(FaceModule().recognize(show=window))
    elif cmd == "compare" and len(sys.argv) == 4:
        print(f"{compare(load_model(), sys.argv[2], sys.argv[3]):.4f}  (threshold {THRESHOLD})")
    else:
        print(__doc__)
