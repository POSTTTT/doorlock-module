# Door lock module

Door lock module decides who may open a door. It recognizes registered people by face, accepts a door passcode as a fallback, and logs every attempt. Fingerprint unlock is planned, and the Registration page already has a placeholder for it.

The module only identifies the person. It tells your backend the result, and the backend decides whether to open the door. It implements `Face Recognition Module — Implementation Plan.pdf`.

## Features

- **Face ID unlock** with a head-turn liveness check, so a photo held up to the camera doesn't pass.
- **Face ID enrollment** like an iPhone: one straight-on shot, then one shot each with the head turned left, right, up and down.
- **Door passcode** of 4 to 12 digits on an on-screen keypad. It checks automatically after the last digit.
- **Registration area**, locked with the door passcode, for managing people, the passcode and the logs.
- **Works from a phone** on the same Wi‑Fi, using the phone's own front camera.
- **Privacy:** only face embeddings are stored, never photos. Removing a person deletes their face data.

## How face recognition works

For each scan:

1. OpenCV reads frames from the camera at 640×480. On a phone, the page sends frames from the phone's camera instead.
2. InsightFace RetinaFace finds the face and its eye, nose and mouth points.
3. A quality check skips frames where the face is too small, blurry or badly lit.
4. For a door check, a liveness check needs the head to turn left and right.
5. InsightFace aligns the face, and ArcFace turns it into a 512-number embedding.
6. The embedding is compared with every stored one using cosine similarity.
7. The result is logged to SQLite and returned.

## Setup

You need Python 3.10 or newer and a webcam.

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The first run downloads the `buffalo_l` model (about 280 MB) into `~/.insightface`.

On macOS, the first scan asks for camera permission for your terminal app (Terminal, iTerm or VS Code). Allow it. If you denied it before, turn it on in **System Settings → Privacy & Security → Camera**, then restart the terminal.

## Running the server

```sh
# This Mac only
.venv/bin/uvicorn api:app --port 8000

# Also reachable from a phone on the same Wi‑Fi, scanning with the phone's camera
.venv/bin/uvicorn api:app --host 0.0.0.0 --port 8000 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

With the first command, open http://localhost:8000 on the Mac.

The second command needs a self-signed certificate, because phones only let a web page use the camera over HTTPS. Create it once in the project folder:

```sh
openssl req -x509 -newkey rsa:2048 -nodes -days 365 -keyout key.pem -out cert.pem -subj "/CN=doorlock"
```

When the server starts, it prints the address to open on the phone, such as `https://172.16.35.187:8000`. The first time, the phone warns that the connection isn't private. Accept the warning, then allow camera access. On the Mac, open https://localhost:8000 while the server runs this way. The certificate lasts 365 days. Git ignores `key.pem` and `cert.pem`.

> **Security:** with `--host 0.0.0.0`, anyone on the same network can open the page and the API. Use it only on a network you trust.

uvicorn doesn't reload code changes on its own. Restart the server after updating the code, or add `--reload` while developing.

## Using the app

| Page | Path | What it does |
|---|---|---|
| Home | `/` | **Start door unlock** runs a face check. **Enter passcode** opens the keypad. **Register a person** opens Registration. |
| Door | `/door` | The face check, with the camera view and the result as a banner inside it. |
| Passcode | `/passcode` | The keypad for the door passcode. |
| Registration | `/register` | A hub with a card for each section below. |
| Face ID | `/register/face` | Enroll people, and see or remove the people registered. |
| Fingerprint ID | `/register/fingerprint` | Placeholder until a fingerprint reader is added. |
| Passcode | `/register/passcode` | Set or change the door passcode. |
| Logs | `/register/logs` | The 20 most recent attempts, in your local time. |

The camera only turns on while a scan is running. The camera view is mirrored, like a selfie camera.

### Registration lock

Once a door passcode is set, every Registration page asks for it first. A correct passcode opens Registration for 15 minutes. Click **Lock** at the top right to close it sooner. Restarting the server also locks it. Until the first passcode is set, Registration stays open so you can set one.

### Enroll a face

1. Open **Registration**, then **Face ID**.
2. Type the person's name and click **Enroll person**.
3. Look straight at the camera. The first good frame is the straight-on shot.
4. Move your head slowly in a circle. A ring around the camera view has four segments, for up, right, down and left. Each one lights up when that head turn is captured.
5. Enrollment finishes when all five shots are captured. If a direction is still missing after 30 s, the banner says "Enrollment failed" and names it. Turn further that way, add light or move closer, then try again.

The box around your face is green when the frame is good enough to use. It is red when the face is too small, blurry, or too dark or bright.

People enrolled from the page get the role `user`. To give someone another role, use the command line or the API.

To remove someone, click the trash button next to their name under **Registered people**. This deletes their face data, so they can no longer unlock the door by face. Their name stays in the logs.

### Door check

1. On the home page, click **Start door unlock**. The face check starts right away.
2. Face the camera, then slowly turn your head left and right.
3. The banner shows the result:

| Banner | API result | Meaning |
|---|---|---|
| Green: Access granted | `granted` | The face matched a registered person. The banner shows the match score as a percentage. |
| Red: Access denied | `unknown` | A real face was seen, but it didn't match anyone registered. |
| Amber: Liveness check failed | `spoof_suspected` | The head never turned enough. It could be a photo, or you didn't turn your head. |
| Amber: No face found | `no_face` | No usable face was seen within 10 s. |

To try again, go back to the home page and click **Start door unlock**.

### Door passcode

To unlock with the passcode, click **Enter passcode** on the home page or the Door page. Tap the digits, or type them. The keypad checks as soon as you enter the last digit. A correct passcode returns you to the home page after 2 s.

To set the passcode, open **Registration**, then **Passcode**. Type a new passcode of 4 to 12 digits twice and click **Set PIN**. To change it later, use the same form; it then asks for the current passcode too.

After 5 wrong passcodes in a row, passcode checks are blocked for 30 s. Wrong current passcodes on the change form count too. The passcode is stored as a salted PBKDF2 hash, never as plain text.

The page loads its fonts from Google Fonts. Without internet access, it falls back to system fonts and still works.

## Command line

These commands use the Mac camera and open an OpenCV preview window instead of the web page.

```sh
.venv/bin/python face.py enroll "Alice" admin   # enroll a person, with an optional role
.venv/bin/python face.py recognize              # run one door check and print the result
.venv/bin/python face.py compare a.jpg b.jpg    # print the similarity score between two photos
```

## API

Interactive docs are at `/docs` on the running server, for example http://localhost:8000/docs.

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/recognize` | none | `{"result": "granted", "user_id": 1, "name": "Alice", "score": 0.62}` |
| POST | `/unlock` | `{"pin": "4821"}` | `{"result": "granted"}` or `{"result": "denied"}`; 429 after 5 wrong passcodes; 503 if no passcode is set |
| POST | `/login` | `{"pin": "4821"}` | `{"ok": true}` and a `session` cookie that unlocks the 🔒 endpoints for 15 min; 403 if wrong |
| POST | `/logout` | none | Ends the session |
| POST | `/enroll` 🔒 | `{"name": "Alice", "role": "admin"}` | `{"user_id": 1}`; 422 if the straight-on shot or a head turn wasn't captured |
| GET | `/users` 🔒 | none | Registered people, newest first, each with `id`, `name`, `role` and `created_at` |
| DELETE | `/users/{id}` 🔒 | none | `{"removed": 1}`; deletes the person's face data and keeps their name in the logs; 404 if there is no such person |
| GET | `/pin` | none | `{"set": true, "length": 4}`; `length` is the passcode's digit count, or `null` for a passcode saved before lengths were stored, until it's entered once |
| POST | `/pin` 🔒 | `{"current": "4821", "new": "5930"}` | `{"set": true}`; `current` is needed only to change an existing passcode; 403 if it's wrong; 422 if `new` isn't 4 to 12 digits |
| GET | `/logs` 🔒 | none | The 20 most recent attempts, each with `method` (`face` or `password`) |
| GET | `/preview.jpg` | none | The latest mirrored camera frame during a scan, or 404. The `X-Scan` header holds the scan progress as JSON. |
| POST | `/frame` | a JPEG image | Feeds one camera frame to a running `source=browser` scan; 409 if none is running; 400 if not a JPEG; 413 if over 2 MB |

🔒 Once a door passcode is set, these return 401 without the `session` cookie from `/login`.

`/recognize` and `/enroll` take `?source=browser` to scan frames the page posts to `/frame`, instead of the Mac camera. The phone page uses this. For an enrollment, `/frame` also needs the `session` cookie.

Only one scan runs at a time, because there is one camera. Each scan call blocks until it finishes: up to 10 s for `/recognize` and 30 s for `/enroll`.

Example from the backend:

```python
import requests

r = requests.post("http://localhost:8000/recognize", timeout=15).json()
if r["result"] == "granted":
    open_door()  # your microcontroller or relay code
```

## Database

The database is `door.db` (SQLite). It is created automatically in the folder you start the server from.

| Table | Columns |
|---|---|
| `users` | id, name, role, is_active, created_at |
| `face_embeddings` | id, user_id, embedding (512 float32 values as a BLOB), model_name, created_at |
| `access_logs` | id, user_id, method, result, score, timestamp |
| `settings` | key, value: `door_pin` (the hashed passcode) and `door_pin_length` |

Removing a person sets `is_active` to 0 and deletes their rows in `face_embeddings`. The `users` row stays so old logs keep the name.

## Tuning

All settings are at the top of `face.py`.

| Setting | Default | Effect |
|---|---|---|
| `THRESHOLD` | 0.40 | The minimum cosine score for a match, shown as 40% in the app. Raise it if the wrong people get in; lower it if registered people get rejected. |
| `MIN_FACE_PX` | 80 | The smallest face size in pixels. Lower it if people stand far from the camera. |
| `MIN_SHARPNESS` | 60 | The blur limit. Lower it for a soft or cheap webcam. |
| `BRIGHTNESS` | (50, 210) | The allowed brightness range of the face. |
| `MIN_YAW_RANGE` | 0.30 | How far the head must turn for the liveness check. Raise it to make spoofing harder. |
| `TURN` | 0.18 | How far the head must turn left or right during enrollment. |
| `TILT` | 0.08 | How far the head must tilt up or down during enrollment. |
| `FRAME_SKIP` | 2 | Only every Nth frame is processed. Raise it if the CPU is slow. |

To tune `THRESHOLD`:

1. Run `face.py compare` on pairs of photos of the same person, and on pairs of different people.
2. Set the threshold between the two groups of scores.
3. Test with people who aren't registered.

## Tests

These tests don't need a camera:

```sh
.venv/bin/python test_face.py
```

They check matching, the database, the enrollment head turns with a simulated camera, and real embeddings on a sample photo that comes with InsightFace.

## Troubleshooting

- **The scan uses your iPhone instead of the Mac camera:** Continuity Camera puts the iPhone at camera 0. Start the server with `CAMERA_INDEX=1`, for example `CAMERA_INDEX=1 .venv/bin/uvicorn api:app --port 8000`. When the iPhone isn't connected, it falls back to camera 0, the Mac camera.
- **`cannot open camera`:** another app is using the camera, or the terminal doesn't have camera permission (see Setup).
- **"Camera not available" on the phone:** open the `https://` address, not `http://`, and allow camera access for the page.
- **An enrollment direction never lights up:** turn further, or lower `TURN` (left and right) or `TILT` (up and down).
- **Always `spoof_suspected`:** turn your head more, or lower `MIN_YAW_RANGE`.
- **Always `no_face`, or red boxes:** there isn't enough light, or the face is too far away. Add a lamp, move closer, or relax the quality settings.
- **A registered person gets `unknown`:** enroll them again under the same lighting as the door, or lower `THRESHOLD` slightly.
- **A code change has no effect:** restart the server.
