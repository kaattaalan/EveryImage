import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from src.scanner import IndexingEngine
import src.database as db

conn = db.get_connection()
cur = conn.cursor()

# Pick 32 images from library 6
cur.execute("SELECT id, file_path FROM images WHERE library_id = 6 ORDER BY id DESC LIMIT 32")
rows = cur.fetchall()
img_ids = [r["id"] for r in rows]
print(f"Temporarily removing {len(img_ids)} images from DB to test live scanning...")

if img_ids:
    placeholders = ",".join("?" for _ in img_ids)
    cur.execute(f"SELECT id FROM faces WHERE image_id IN ({placeholders})", img_ids)
    face_ids = [r["id"] for r in cur.fetchall()]
    if face_ids:
        f_placeholders = ",".join("?" for _ in face_ids)
        cur.execute(f"DELETE FROM vec_faces WHERE face_id IN ({f_placeholders})", face_ids)
        cur.execute(f"DELETE FROM faces WHERE id IN ({f_placeholders})", face_ids)
    cur.execute(f"DELETE FROM vec_images WHERE image_id IN ({placeholders})", img_ids)
    cur.execute(f"DELETE FROM images WHERE id IN ({placeholders})", img_ids)
    conn.commit()
conn.close()

# Now start scanner
engine = IndexingEngine.get_instance()
print("Starting scanner on library 6...")
engine.start([6])

t_start = time.perf_counter()
while True:
    time.sleep(0.5)
    st = engine.get_status()
    print(f"State: {st['state']}, Tagged: {st['tagged_files']}/{st['total_files']}, Faces: {st['faces_found']}, Speed: {st['speed_fps']} fps, File: {st['current_file']}")
    if st["state"] in ("completed", "stopped", "error"):
        break

t_total = time.perf_counter() - t_start
print(f"Finished in {t_total:.2f}s!")
st = engine.get_status()
print(f"Final Tagged: {st['tagged_files']}/{st['total_files']}, Speed: {st['speed_fps']} fps")
