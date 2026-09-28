import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from src.scanner import IndexingEngine
import src.database as db

engine = IndexingEngine.get_instance()
print("Starting indexing on library 6 (F:\\Backup\\S21FE\\DCIM\\Camera)...")
engine.start([6])

for i in range(20):
    time.sleep(1.0)
    st = engine.get_status()
    print(f"[{i+1}s] State: {st['state']}, Scanned: {st['scanned_files']}/{st['total_files']}, Tagged: {st['tagged_files']}, Faces: {st['faces_found']}, Speed: {st['speed_fps']} fps, File: {st['current_file']}")
    if st['state'] in ('completed', 'stopped', 'error'):
        break

engine.stop()
print("Final status:", engine.get_status())
