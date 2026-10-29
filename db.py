import sqlite3

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'user',
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    model_name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS access_logs (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    method TEXT NOT NULL,
    result TEXT NOT NULL,
    score REAL,
    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(path="door.db"):
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def add_user(conn, name, role="user"):
    with conn:
        return conn.execute("INSERT INTO users (name, role) VALUES (?, ?)", (name, role)).lastrowid


def remove_user(conn, user_id):
    """Deletes the user's face data and deactivates them; the row stays so old access logs keep the name.
    Returns False if there is no active user with that id."""
    with conn:
        if not conn.execute("UPDATE users SET is_active = 0 WHERE id = ? AND is_active = 1", (user_id,)).rowcount:
            return False
        conn.execute("DELETE FROM face_embeddings WHERE user_id = ?", (user_id,))
    return True


def add_embedding(conn, user_id, emb, model_name):
    with conn:
        conn.execute(
            "INSERT INTO face_embeddings (user_id, embedding, model_name) VALUES (?, ?, ?)",
            (user_id, np.asarray(emb, np.float32).tobytes(), model_name),
        )


def load_embeddings(conn, model_name):
    """Returns (ids, names, matrix) for active users; matrix rows are unit vectors."""
    rows = conn.execute(
        "SELECT u.id, u.name, e.embedding FROM face_embeddings e JOIN users u ON u.id = e.user_id "
        "WHERE u.is_active = 1 AND e.model_name = ?",
        (model_name,),
    ).fetchall()
    if not rows:
        return [], [], np.empty((0, 512), np.float32)
    ids, names, blobs = zip(*rows)
    return list(ids), list(names), np.stack([np.frombuffer(b, np.float32) for b in blobs])


def log_access(conn, user_id, result, score, method="face"):
    with conn:
        conn.execute(
            "INSERT INTO access_logs (user_id, method, result, score) VALUES (?, ?, ?, ?)",
            (user_id, method, result, score),
        )


def get_setting(conn, key):
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row and row[0]


def set_setting(conn, key, value):
    with conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))
