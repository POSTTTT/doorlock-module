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
# This Mac only
.venv/bin/uvicorn api:app --port 8000

# Also reachable from a phone on the same Wi‑Fi, scanning with the phone's camera
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

The second command needs a self-signed certificate, because phones only let a page use the camera over HTTPS. Create it once:

```sh
openssl req -x509 -newkey rsa:2048 -nodes -days 365 -keyout key.pem -out cert.pem -subj "/CN=face-door"
```

The command prints the address to open on the phone, such as `https://172.16.35.187:8000`. The phone warns that the connection isn't private. Accept the warning once, then allow camera access. On the phone, scans use the phone's front camera. On the Mac, they still use the Mac's camera; open https://localhost:8000 there while the server runs with HTTPS. Anyone on the network can reach the page, so do this only on a network you trust. The certificate lasts 365 days.

Open http://localhost:8000. The home page has a **Start door unlock** button that opens the Door page. **Door** (`/door`) has the door check and the door PIN. **Registration** (`/register`) is a hub of cards, each opening its own page: **Face ID** (`/register/face`) enrolls people and lists them with a remove button for each; **Fingerprint ID** (`/register/fingerprint`) is a placeholder until a fingerprint reader is added; **Passcode** (`/register/passcode`) sets the door PIN; **Logs** (`/register/logs`) shows the recent attempts. The home page links to Registration with **Register a person**. The home icon at the top left returns to the home page, and **Registration** next to it returns to the hub. The Door and Face ID pages show the camera view with the result as a banner inside it. The camera only turns on while a scan is running.

**Registration lock**

Once a door PIN is set, the Registration page is locked with the same passcode. Opening it shows the passcode keypad first. A correct passcode opens Registration for 15 minutes. Click **Lock** at the top to close it sooner. Restarting the server also locks it. Until the first PIN is set, Registration stays open so that you can set one.

**Enroll a person**

1. Open **Registration**, then the **Face ID** card. Once a door PIN is set, Registration first asks for that passcode.
2. Type a name. People enrolled from the page get the role `user`. To set another role, use the command line or the API.
3. Click **Enroll person** and look straight at the camera. The first good frame is the straight-on shot.
4. Move your head slowly in a circle, like setting up Face ID on an iPhone. A ring around the camera view has four segments, for up, right, down and left. Each one lights up when a shot with the head turned that way is captured. Enrollment needs all five shots, so a face that doesn't move can't enroll.
5. Watch the box around your face. It is green when the frame is good enough to use, and red when the face is too small, blurry, or too dark or bright.
6. On success the banner shows "Alice enrolled" with the new user number. If a direction is still missing after 30 s, it shows "Enrollment failed" and names it. Turn your head further that way, or improve the lighting or move closer, then try again. If one direction never fills even when you turn well, lower `TURN` (left and right) or `TILT` (up and down) in `face.py`.

**Door check**

1. On the home page, click **Start door unlock**. The Door page opens and the face check starts right away. To run it again, click the home icon at the top left, then **Start door unlock**.
2. Face the camera and slowly turn your head left, then right.
3. The banner shows one of these results:

| Banner | API result | Meaning |
|---|---|---|
| Green: Access granted | `granted` | The face matched a registered person. The banner shows their name and match score. |
| Red: Access denied | `unknown` | A real face was seen, but no stored face scored above the threshold. |
| Amber: Liveness check failed | `spoof_suspected` | A face was seen but the head never turned enough. It could be a photo, or you didn't turn your head. |
| Amber: No face found | `no_face` | No usable face was seen within 10 s. |

**Door PIN**

If the face check fails, click **Enter passcode** at the top of the home page or the Door page. Tap the digits on the keypad, or type them. The keypad checks the passcode as soon as you enter the last digit. For a PIN saved before this feature, tap the check mark or press Enter the first time; after that it checks automatically. A correct passcode returns you to the home page after 2 s.

To set the PIN the first time, open **Registration**, then the **Passcode** card. In the **Door PIN** panel, type a new PIN of 4 to 12 digits twice and click **Set PIN**. To change it later, use the same panel: it then asks for the current PIN as well, and the button reads **Change PIN**.

Until a PIN is set, PIN unlock shows "PIN not checked". After 5 wrong PINs in a row, PIN checks are blocked for 30 s. This counts wrong current PINs on the change form too. The PIN is stored as a salted PBKDF2 hash in the `settings` table, never as plain text.

Every attempt appears under **Recent attempts** on the **Logs** page of Registration, with times in your local time zone.

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
| POST | `/login` | `{"pin": "4821"}` | `{"ok": true}` and a `session` cookie that unlocks the locked endpoints for 15 min; 403 if wrong |
| POST | `/logout` | none | Ends the session |
| POST | `/enroll` 🔒 | `{"name": "Alice", "role": "admin"}` | `{"user_id": 1}`, or 422 if the straight-on shot or any of the four head turns wasn't captured |
| POST | `/recognize` | none | `{"result": "granted", "user_id": 1, "name": "Alice", "score": 0.62}` |
| POST | `/unlock` | `{"pin": "4821"}` | `{"result": "granted"}` or `{"result": "denied"}`; 429 after 5 wrong PINs, 503 if no PIN is set |
| GET | `/pin` | none | `{"set": true, "length": 4}`; `set` is true if a door PIN is set, `length` is its digit count (`null` for an older PIN until it is entered correctly once) |
| POST | `/pin` 🔒 | `{"current": "4821", "new": "5930"}` | `{"set": true}`; `current` is needed only when a PIN is already set; 403 if it is wrong, 422 if `new` is not 4 to 12 digits |
| GET | `/logs` 🔒 | none | The last 20 attempts, each with `method` (`face` or `password`) |
| GET | `/users` 🔒 | none | Registered people, newest first, each with `id`, `name`, `role` and `created_at` |
| DELETE | `/users/{id}` 🔒 | none | `{"removed": id}`; deletes the person's face data so they can no longer be recognized, and keeps their name in old access logs; 404 if there is no such person |
| GET | `/preview.jpg` | none | The latest camera frame during a scan, or 404. The `X-Scan` header holds the scan progress as JSON: `{"shots": 2, "want": 3, "live": false, "face": true, "good": true}` |

🔒 Once a door PIN is set, these endpoints return 401 unless the request carries the `session` cookie from `/login`.

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
