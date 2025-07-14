import sqlite3, os, time, json, threading

_DB_PATH = os.path.join('data', 'learning.db')
os.makedirs('data', exist_ok=True)
_lock = threading.RLock()

def _init():
    with _lock:
        conn = sqlite3.connect(_DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER,
            kind TEXT,
            query TEXT,
            result TEXT,
            good INTEGER,
            embedded INTEGER DEFAULT 0
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS artefact(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER,
            type TEXT,
            path TEXT,
            metadata_json TEXT
        )''')
        conn.commit(); conn.close()
_init()

def log_feedback(kind, query, result, good: bool):
    with _lock:
        conn = sqlite3.connect(_DB_PATH)
        conn.execute('INSERT INTO feedback(ts,kind,query,result,good) VALUES(?,?,?,?,?)',
                     (int(time.time()), kind, query, result, 1 if good else 0))
        conn.commit(); conn.close()

def fetch_unembedded(limit=100):
    conn = sqlite3.connect(_DB_PATH)
    rows = conn.execute('SELECT id, query, result FROM feedback WHERE good=1 AND embedded=0 LIMIT ?', (limit,)).fetchall()
    conn.close(); return rows

def mark_embedded(ids):
    if not ids: return
    conn = sqlite3.connect(_DB_PATH)
    q = 'UPDATE feedback SET embedded=1 WHERE id IN (%s)' % ','.join('?'*len(ids))
    conn.execute(q, ids); conn.commit(); conn.close()