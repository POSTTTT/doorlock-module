# Face recognition module

This module identifies a registered person from the laptop camera and tells the backend who they are. The backend then decides whether to open the door.
It implements `Face Recognition Module — Implementation Plan.pdf`.

The pipeline for each attempt:

1. OpenCV reads frames from the camera at 640×480.
2. InsightFace RetinaFace finds the face and its eye, nose and mouth points.
3. A quality check skips frames where the face is too small, blurry or badly lit.
4. A liveness check asks you to turn your head left and right, to block photos held up to the camera.
5. InsightFace aligns the face and ArcFace turns it into a 512-number embedding.
6. The embedding is compared to every stored one with cosine similarity.
7. The result is logged to SQLite and returned.

Only embeddings are stored, never photos.

## Setup

You need Python 3.10 or newer and a webcam.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first run downloads the `buffalo_l` model (about 280 MB) into `~/.insightface`.

On macOS, the first scan asks for camera permission for your terminal app (Terminal, iTerm or VS Code). Allow it. If you denied it before, turn it on in **System Settings → Privacy & Security → Camera** and restart the terminal.

## Using the web test page

```sh
.venv/bin/uvicorn api:app --port 8000
```

Open http://localhost:8000. There are two pages, linked at the top right. **Door** (`/`) has the door check and the door PIN. **Registration** (`/register`) has enrollment and the recent attempts. Each page shows the camera view on the left and a result banner under it. The camera only turns on while a scan is running.

**Enroll a person**

1. Open the **Registration** page.
2. Type a name. The role is optional and defaults to `user`.
3. Click **Enroll person** and look at the camera.
4. Move your head a little and change your expression while it takes 8 shots, about 0.4 s apart.
5. Watch the box around your face. It is green when the frame is good enough to use, and red when the face is too small, blurry, or too dark or bright.
6. On success the banner shows "Alice enrolled" with the new user number. If fewer than 5 good shots were captured within 30 s, it shows "Enrollment failed". Improve the lighting or move closer, then try again.

**Door check**

1. Open the **Door** page and click **Start door check**.
2. Face the camera and slowly turn your head left, then right.
3. The banner shows one of these results:

| Banner | API result | Meaning |
|---|---|---|
| Green: Access granted | `granted` | The face matched a registered person. The banner shows their name and match score. |
| Red: Access denied | `unknown` | A real face was seen, but no stored face scored above the threshold. |
| Amber: Liveness check failed | `spoof_suspected` | A face was seen but the head never turned enough. It could be a photo, or you didn't turn your head. |
| Amber: No face found | `no_face` | No usable face was seen within 10 s. |

**Door PIN**

If the face check fails, type the door PIN in the **Door PIN** panel on the Door page and click **Unlock with PIN**.

To set the PIN the first time, open the **Registration** page. In the **Door PIN** panel, type a new PIN of 4 to 12 digits twice and click **Set PIN**. To change it later, use the same panel: it then asks for the current PIN as well, and the button reads **Change PIN**.

Until a PIN is set, PIN unlock shows "PIN not checked". After 5 wrong PINs in a row, PIN checks are blocked for 30 s. This counts wrong current PINs on the change form too. The PIN is stored as a salted PBKDF2 hash in the `settings` table, never as plain text.

Every attempt appears under **Recent attempts** on the Registration page, with times in your local time zone.

The page loads its fonts from Google Fonts. Without internet access it falls back to system fonts and still works.

## Using the command line

These commands open an OpenCV preview window instead of the web page.

```sh
.venv/bin/python face.py enroll "Alice" admin   # enroll a person with a role
.venv/bin/python face.py recognize              # run one door check and print the result
.venv/bin/python face.py compare a.jpg b.jpg    # show the similarity score between two photos
```

## API for the team backend

The server listens on the port you gave to uvicorn. Interactive docs are at http://localhost:8000/docs.

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/enroll` | `{"name": "Alice", "role": "admin"}` | `{"user_id": 1}`, or 422 if not enough good shots were captured |
| POST | `/recognize` | none | `{"result": "granted", "user_id": 1, "name": "Alice", "score": 0.62}` |
| POST | `/unlock` | `{"pin": "4821"}` | `{"result": "granted"}` or `{"result": "denied"}`; 429 after 5 wrong PINs, 503 if no PIN is set |
| GET | `/pin` | none | `{"set": true}` if a door PIN is set |
| POST | `/pin` | `{"current": "4821", "new": "5930"}` | `{"set": true}`; `current` is needed only when a PIN is already set; 403 if it is wrong, 422 if `new` is not 4 to 12 digits |
| GET | `/logs` | none | The last 20 attempts, each with `method` (`face` or `password`) |
| GET | `/preview.jpg` | none | The latest camera frame during a scan, or 404 |

Only one scan runs at a time because there is one camera. Each call blocks until its scan finishes (up to 10 s for `/recognize` and 30 s for `/enroll`).

Example from the backend:

```python
import requests

r = requests.post("http://localhost:8000/recognize", timeout=15).json()
if r["result"] == "granted":
    open_door()  # your microcontroller / relay code
```

## Database

The database is `door.db` (SQLite). It is created automatically in the directory you start the program from.

| Table | Columns |
|---|---|
| `users` | id, name, role, is_active, created_at |
| `face_embeddings` | id, user_id, embedding (512 float32 values as a BLOB), model_name, created_at |
| `access_logs` | id, user_id, method, result, score, timestamp |
| `settings` | key, value (holds `door_pin`, the hashed door PIN) |

To disable a person without deleting them, run:

```sh
sqlite3 door.db "UPDATE users SET is_active = 0 WHERE name = 'Alice'"
```

Then restart the server, because it loads the embeddings into memory at startup.

## Tuning

All settings are at the top of `face.py`.

| Setting | Default | Effect |
|---|---|---|
| `THRESHOLD` | 0.40 | The minimum cosine score for a match. Raise it if the wrong people get accepted; lower it if registered people get rejected. |
| `MIN_FACE_PX` | 80 | The smallest face size in pixels. Lower it if people stand far from the camera. |
| `MIN_SHARPNESS` | 60 | The blur limit. Lower it for a soft or cheap webcam. |
| `BRIGHTNESS` | (50, 210) | The allowed brightness range of the face. |
| `MIN_YAW_RANGE` | 0.30 | How far the head must turn for liveness. Raise it to make spoofing harder. |
| `FRAME_SKIP` | 2 | Only every Nth frame is processed. Raise it if the CPU is slow. |

To tune `THRESHOLD`:

1. Run `face.py compare` on pairs of photos of the same person and on pairs of different people.
2. Set the threshold between the two groups of scores.
3. Test with people who are not registered.

## Tests

These tests don't need a camera:

```sh
.venv/bin/python test_face.py
```

They check matching, the database and real embeddings on a sample photo that comes with InsightFace.

## Troubleshooting

- **The scan uses your iPhone instead of the Mac camera:** Continuity Camera puts the iPhone at camera 0. Start the server with `CAMERA_INDEX=1 .venv/bin/uvicorn api:app --port 8000`. When the iPhone isn't connected, camera 1 doesn't exist and it falls back to camera 0, the Mac camera.
- **`cannot open camera`:** another app is using the camera, or the terminal doesn't have camera permission (see Setup).
- **Always `spoof_suspected`:** turn your head more, or lower `MIN_YAW_RANGE`.
- **Always `no_face` or red boxes:** there isn't enough light or the face is too far away. Add a lamp, move closer, or relax the quality settings.
- **Registered person gets `unknown`:** enroll again under the same lighting as the demo spot, or lower `THRESHOLD` slightly.
