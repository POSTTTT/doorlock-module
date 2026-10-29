"""Checks without a camera: matching, DB round-trip, enrollment poses, and real embeddings on insightface's sample photo."""
import numpy as np

import db
from face import MODEL, POSES, THRESHOLD, largest_face, load_model, match, mean_embedding, pose_of, scan_poses


def unit(v):
    return v / np.linalg.norm(v)


def test_match_and_db():
    rng = np.random.default_rng(0)
    alice, bob = unit(rng.normal(size=512)), unit(rng.normal(size=512))
    conn = db.connect(":memory:")
    db.add_embedding(conn, db.add_user(conn, "alice"), alice, MODEL)
    db.add_embedding(conn, db.add_user(conn, "bob"), bob, MODEL)
    ids, names, matrix = db.load_embeddings(conn, MODEL)

    i, score = match(unit(alice + 0.02 * rng.normal(size=512)), matrix)
    assert names[i] == "alice" and score > 0.9
    assert match(unit(rng.normal(size=512)), matrix)[0] is None  # stranger
    assert match(alice, np.empty((0, 512)))[0] is None  # empty DB

    conn.execute("UPDATE users SET is_active = 0 WHERE name = 'bob'")
    assert db.load_embeddings(conn, MODEL)[1] == ["alice"]

    db.log_access(conn, ids[0], "granted", score)
    assert conn.execute("SELECT method, result FROM access_logs").fetchone() == ("face", "granted")

    assert db.get_setting(conn, "door_pin") is None
    db.set_setting(conn, "door_pin", "a")
    db.set_setting(conn, "door_pin", "b")
    assert db.get_setting(conn, "door_pin") == "b"


def test_enrollment_poses():
    assert pose_of(0, 0) == "center"
    assert pose_of(0.3, 0) == "left" and pose_of(-0.3, 0.05) == "right"  # named as seen in the mirrored preview
    assert pose_of(0, 0.2) == "down" and pose_of(0.05, -0.2) == "up"
    assert pose_of(0.12, 0) is None  # between straight-on and turned

    class Face:  # keypoints: left eye, right eye, nose, mouth corners; the nose moves with the head
        def __init__(self, dx, dy):
            self.bbox, self.det_score = np.array([200, 140, 440, 380], float), 0.9
            nose = [320 + 80 * dx, 260 + 80 * dy]
            self.kps = np.array([[280, 220], [360, 220], nose, [290, 320], [350, 320]], float)
            self.normed_embedding = unit(np.random.default_rng(0).normal(size=512)).astype(np.float32)

    class App:
        def __init__(self, moves):
            self.moves = iter(moves)

        def get(self, frame):
            return [Face(*next(self.moves, (0, 0)))]

    class Cam:
        def read(self):
            return True, np.random.default_rng(1).integers(40, 220, (480, 640, 3), np.uint8)

    # FRAME_SKIP drops every other frame, so each pose is listed twice.
    still = scan_poses(App([(0, 0)] * 40), Cam(), timeout=0.5)
    assert list(still) == ["center"]  # a face that never moves can't fill the other poses
    circle = [(0, 0)] * 2 + [(0.3, 0)] * 2 + [(-0.3, 0)] * 2 + [(0, -0.2)] * 2 + [(0, 0.2)] * 2
    assert sorted(scan_poses(App(circle), Cam(), timeout=5)) == sorted(POSES)


def test_real_embeddings():
    from insightface.data import get_image

    app = load_model()
    img = get_image("t1")  # group photo shipped with insightface
    faces = app.get(img)
    assert len(faces) >= 2
    matrix = np.stack([f.normed_embedding for f in faces])
    assert matrix.shape[1] == 512
    # Every face in a shifted copy of the photo matches a distinct original face.
    shifted = app.get(np.roll(img, 5, axis=1))
    hits = [match(mean_embedding([f.normed_embedding]), matrix) for f in shifted]
    assert all(i is not None and score > 0.8 for i, score in hits)
    assert len({i for i, _ in hits}) == len(hits)
    assert largest_face(app, img) is not None
    # Different people in the photo stay below threshold.
    sims = matrix @ matrix.T
    assert (sims[~np.eye(len(faces), dtype=bool)] < THRESHOLD).all()


if __name__ == "__main__":
    test_match_and_db()
    test_enrollment_poses()
    test_real_embeddings()
    print("ok")
