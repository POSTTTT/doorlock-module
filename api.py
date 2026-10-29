"""FastAPI surface for the team backend, plus a test page at /.  Run: uvicorn api:app --port 8000"""
import hashlib
import hmac
import json
import os
import secrets
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

import db
from face import FaceModule

app = FastAPI(title="Face recognition module")
face = FaceModule()
camera = threading.Lock()  # one camera, one scan at a time
latest = {"jpg": None, "info": {}}  # last annotated frame of the running scan and its progress, for the test page
pin_lock = threading.Lock()
pin_fails = {"count": 0, "until": 0.0}  # 5 wrong PINs in a row lock PIN checks for 30 s
# shortcut: sessions live in memory, so a server restart locks Registration again; fine for one kiosk.
sessions = {}  # session cookie token -> expiry (time.monotonic)
SESSION_S = 15 * 60


def network_url():
    """http://<LAN IP>:<port> when uvicorn was started with --host 0.0.0.0, else None."""
    # shortcut: reads uvicorn's flags in the "--host 0.0.0.0" form only; use env vars if the start command changes.
    flag = lambda name, default: sys.argv[sys.argv.index(name) + 1] if name in sys.argv[:-1] else default
    if flag("--host", "127.0.0.1") != "0.0.0.0":
        return None
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect(("8.8.8.8", 80))  # UDP connect sends nothing; it only picks the interface that faces the network
        except OSError:
            return None
        return f"http://{s.getsockname()[0]}:{flag('--port', '8000')}"


if url := network_url():
    print(f"On your phone, open {url}", flush=True)


def hash_pin(pin, salt=None):
    salt = salt or os.urandom(16)
    return salt.hex() + "$" + hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, 200_000).hex()


def check_pin(pin):
    """True if pin matches the stored door PIN. Counts failures toward the lockout."""
    stored = db.get_setting(face.conn, "door_pin")
    with pin_lock:
        wait = pin_fails["until"] - time.monotonic()
        if wait > 0:
            raise HTTPException(429, f"too many wrong PINs, try again in {int(wait) + 1} s")
        ok = hmac.compare_digest(hash_pin(pin, bytes.fromhex(stored.split("$")[0])), stored)
        pin_fails["count"] = 0 if ok else pin_fails["count"] + 1
        if pin_fails["count"] >= 5:
            pin_fails["count"], pin_fails["until"] = 0, time.monotonic() + 30
    return ok


def pin_set():
    return bool(db.get_setting(face.conn, "door_pin"))


def start_session(response):
    now = time.monotonic()
    for t in [t for t, exp in sessions.items() if exp <= now]:
        del sessions[t]
    token = secrets.token_urlsafe(32)
    sessions[token] = now + SESSION_S
    response.set_cookie("session", token, max_age=SESSION_S, httponly=True, samesite="strict")


def unlocked(request):
    """Registration is open until a door PIN exists; after that it needs a session from /login."""
    exp = sessions.get(request.cookies.get("session", ""))
    return not pin_set() or (exp is not None and exp > time.monotonic())


def require_unlocked(request: Request):
    if not unlocked(request):
        raise HTTPException(401, "registration is locked, enter the passcode first")


def fresh_show():
    latest["jpg"] = None  # don't let the page show the previous scan's last frame
    return show


def show(frame, info):
    latest["jpg"], latest["info"] = cv2.imencode(".jpg", frame)[1].tobytes(), info


class Enroll(BaseModel):
    name: str
    role: str = "user"


@app.post("/enroll", dependencies=[Depends(require_unlocked)])
def enroll(body: Enroll):
    with camera:
        try:
            return {"user_id": face.enroll(body.name, body.role, show=fresh_show())}
        except RuntimeError as e:
            raise HTTPException(422, str(e))


@app.post("/recognize")
def recognize():
    """result: granted | unknown | spoof_suspected | no_face. The backend decides whether to open the door."""
    with camera:
        try:
            return face.recognize(show=fresh_show())
        except RuntimeError as e:
            raise HTTPException(503, str(e))


class Unlock(BaseModel):
    pin: str


@app.post("/unlock")
def unlock(body: Unlock):
    """Door PIN fallback. result: granted | denied. Logged with method='password'."""
    if not pin_set():
        raise HTTPException(503, "no door PIN is set yet, set one on the Registration page")
    result = "granted" if check_pin(body.pin) else "denied"
    db.log_access(face.conn, None, result, None, method="password")
    return {"result": result}


class SetPin(BaseModel):
    current: str = ""
    new: str = Field(pattern=r"^\d{4,12}$")


@app.post("/login")
def login(body: Unlock, response: Response):
    """Unlocks Registration for SESSION_S with the door passcode. Not logged as a door attempt."""
    if not pin_set():
        raise HTTPException(503, "no door PIN is set yet, so Registration is open")
    if not check_pin(body.pin):
        raise HTTPException(403, "wrong passcode")
    start_session(response)
    return {"ok": True}


@app.post("/logout")
def logout(request: Request, response: Response):
    sessions.pop(request.cookies.get("session", ""), None)
    response.delete_cookie("session")
    return {"ok": True}


@app.get("/pin")
def pin_state():
    return {"set": pin_set()}


@app.post("/pin", dependencies=[Depends(require_unlocked)])
def set_pin(body: SetPin, response: Response):
    """Sets the door PIN. Changing an existing PIN needs the current one."""
    first = not pin_set()
    if not first and not check_pin(body.current):
        raise HTTPException(403, "the current PIN is wrong")
    db.set_setting(face.conn, "door_pin", hash_pin(body.new))
    if first:
        start_session(response)  # whoever set the first PIN stays on Registration
    return {"set": True}


@app.get("/logs", dependencies=[Depends(require_unlocked)])
def logs():
    rows = face.conn.execute(
        "SELECT l.timestamp, u.name, l.result, l.score, l.method FROM access_logs l "
        "LEFT JOIN users u ON u.id = l.user_id ORDER BY l.id DESC LIMIT 20"
    ).fetchall()
    return [dict(zip(("timestamp", "name", "result", "score", "method"), r)) for r in rows]


@app.get("/preview.jpg")
def preview():
    if latest["jpg"] is None:
        raise HTTPException(404)
    headers = {"Cache-Control": "no-store", "X-Scan": json.dumps(latest["info"])}
    return Response(latest["jpg"], media_type="image/jpeg", headers=headers)


@app.get("/", response_class=HTMLResponse)
@app.get("/door", response_class=HTMLResponse)
@app.get("/passcode", response_class=HTMLResponse)  # same page; JS shows the panels for the path
def page():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    if not unlocked(request):
        return RedirectResponse("/passcode?for=register", 303)
    return page()
