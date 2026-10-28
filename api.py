"""FastAPI surface for the team backend, plus a test page at /.  Run: uvicorn api:app --port 8000"""
import hashlib
import hmac
import os
import threading
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import db
from face import FaceModule

app = FastAPI(title="Face recognition module")
face = FaceModule()
camera = threading.Lock()  # one camera, one scan at a time
latest = {"jpg": None}  # last annotated frame of the running scan, for the test page
pin_lock = threading.Lock()
pin_fails = {"count": 0, "until": 0.0}  # 5 wrong PINs in a row lock PIN checks for 30 s


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


def fresh_show():
    latest["jpg"] = None  # don't let the page show the previous scan's last frame
    return show


def show(frame):
    latest["jpg"] = cv2.imencode(".jpg", frame)[1].tobytes()


class Enroll(BaseModel):
    name: str
    role: str = "user"


@app.post("/enroll")
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
    if not db.get_setting(face.conn, "door_pin"):
        raise HTTPException(503, "no door PIN is set yet, set one on the Registration page")
    result = "granted" if check_pin(body.pin) else "denied"
    db.log_access(face.conn, None, result, None, method="password")
    return {"result": result}


class SetPin(BaseModel):
    current: str = ""
    new: str = Field(pattern=r"^\d{4,12}$")


@app.get("/pin")
def pin_state():
    return {"set": bool(db.get_setting(face.conn, "door_pin"))}


@app.post("/pin")
def set_pin(body: SetPin):
    """Sets the door PIN. Changing an existing PIN needs the current one."""
    if db.get_setting(face.conn, "door_pin") and not check_pin(body.current):
        raise HTTPException(403, "the current PIN is wrong")
    db.set_setting(face.conn, "door_pin", hash_pin(body.new))
    return {"set": True}


@app.get("/logs")
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
    return Response(latest["jpg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/", response_class=HTMLResponse)
@app.get("/register", response_class=HTMLResponse)  # same page; JS shows the enroll panels
def page():
    return (Path(__file__).parent / "index.html").read_text()
