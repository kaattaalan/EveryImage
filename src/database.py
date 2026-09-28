import sqlite3
import sqlite_vec
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import re

import src.config as cfg

DB_PATH = cfg.DATA_DIR / "everyimage.db"

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH), timeout=30.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA cache_size = -64000")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # 1. Libraries
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS libraries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT UNIQUE NOT NULL,
        volume_id TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Persons (Faces clusters)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        cover_face_id INTEGER,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 3. Images metadata
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        library_id INTEGER REFERENCES libraries(id) ON DELETE CASCADE,
        file_path TEXT UNIQUE NOT NULL,
        file_name TEXT NOT NULL,
        file_size INTEGER NOT NULL,
        mtime REAL NOT NULL,
        header_hash TEXT NOT NULL,
        width INTEGER,
        height INTEGER,
        has_embedding INTEGER DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 4. Faces detected in images
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_id INTEGER REFERENCES images(id) ON DELETE CASCADE,
        person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
        bbox_x1 REAL,
        bbox_y1 REAL,
        bbox_x2 REAL,
        bbox_y2 REAL,
        det_score REAL,
        crop_thumb TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Virtual tables for vector similarity (sqlite-vec)
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_images USING vec0(
        image_id INTEGER PRIMARY KEY,
        embedding FLOAT[768]
    );
    """)

    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_faces USING vec0(
        face_id INTEGER PRIMARY KEY,
        embedding FLOAT[512]
    );
    """)

    # Indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_hash ON images(header_hash);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_mtime ON images(file_path, file_size, mtime);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_image ON faces(image_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_faces_person ON faces(person_id);")

    # Ensure captured_at column exists in images
    try:
        cursor.execute("ALTER TABLE images ADD COLUMN captured_at DATETIME;")
    except sqlite3.OperationalError:
        pass
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_captured_at ON images(captured_at DESC);")
    
    # Ensure ocr_text column exists in images
    try:
        cursor.execute("ALTER TABLE images ADD COLUMN ocr_text TEXT;")
    except sqlite3.OperationalError:
        pass

    # Backfill captured_at from mtime if null
    cursor.execute("""
        UPDATE images 
        SET captured_at = datetime(mtime, 'unixepoch') 
        WHERE captured_at IS NULL;
    """)

    # 5. Full-Text Search (FTS5) for OCR text
    cursor.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
        image_id UNINDEXED,
        text_content,
        tokenize='unicode61 remove_diacritics 2'
    );
    """)

    # Indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_library ON images(library_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_no_ocr ON images(id) WHERE ocr_text IS NULL;")

    # Triggers for virtual tables (no FK cascade support)
    cursor.execute("""
    CREATE TRIGGER IF NOT EXISTS trg_delete_image AFTER DELETE ON images
    BEGIN
        DELETE FROM vec_images WHERE image_id = OLD.id;
        DELETE FROM images_fts WHERE image_id = OLD.id;
    END;
    """)

    cursor.execute("""
    CREATE TRIGGER IF NOT EXISTS trg_delete_face AFTER DELETE ON faces
    BEGIN
        DELETE FROM vec_faces WHERE face_id = OLD.id;
    END;
    """)

    conn.commit()
    conn.close()

def clear_database(keep_libraries: bool = True):
    """
    Clears all indexed images, vector embeddings, faces, person clusters, and OCR full-text search records.
    Optionally preserves configured libraries so the user does not need to re-add folders.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys = OFF;")
    cur.execute("DELETE FROM vec_faces;")
    cur.execute("DELETE FROM faces;")
    cur.execute("DELETE FROM persons;")
    cur.execute("DELETE FROM vec_images;")
    cur.execute("DELETE FROM images_fts;")
    cur.execute("DELETE FROM images;")
    if not keep_libraries:
        cur.execute("DELETE FROM libraries;")
    cur.execute("PRAGMA foreign_keys = ON;")
    conn.commit()
    try:
        cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    except Exception:
        pass
    conn.close()

# --- Library Operations ---
def add_library(path_str: str) -> Dict[str, Any]:
    norm_path = str(Path(path_str).resolve())
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO libraries (path) VALUES (?)", (norm_path,))
        lib_id = cur.lastrowid
        conn.commit()
        return {"id": lib_id, "path": norm_path}
    except sqlite3.IntegrityError:
        cur.execute("SELECT id, path, created_at FROM libraries WHERE path = ?", (norm_path,))
        row = cur.fetchone()
        return dict(row)
    finally:
        conn.close()

def get_libraries() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT l.id, l.path, l.created_at,
               COUNT(i.id) as image_count
        FROM libraries l
        LEFT JOIN images i ON l.id = i.library_id
        GROUP BY l.id
        ORDER BY l.id ASC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def remove_library(lib_id: int):
    conn = get_connection()
    try:
        cur = conn.cursor()
        # Images will cascade delete faces. 
        # The triggers we added will delete from vec_images, images_fts, and vec_faces.
        cur.execute("DELETE FROM images WHERE library_id = ?", (lib_id,))
        
        # Delete library record
        cur.execute("DELETE FROM libraries WHERE id = ?", (lib_id,))
        # Clean up orphaned persons
        cur.execute("DELETE FROM persons WHERE id NOT IN (SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL)")
        conn.commit()
    finally:
        conn.close()

def get_image_faces(image_id: int) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT f.id, f.person_id, f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
               f.det_score, f.crop_thumb,
               p.name as person_name
        FROM faces f
        LEFT JOIN persons p ON f.person_id = p.id
        WHERE f.image_id = ?
        ORDER BY f.det_score DESC
    """, (image_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def get_image_by_id(image_id: int) -> Optional[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM images WHERE id = ?", (image_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def get_image_stamp(file_path: str) -> Optional[sqlite3.Row]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, file_size, mtime, has_embedding FROM images WHERE file_path = ?", (file_path,))
    row = cur.fetchone()
    conn.close()
    return row

def get_library_stamps(library_id: int) -> Dict[str, Tuple[int, float, int]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT file_path, file_size, mtime, has_embedding FROM images WHERE library_id = ?", (library_id,))
    res = {row["file_path"]: (row["file_size"], row["mtime"], row["has_embedding"]) for row in cur.fetchall()}
    conn.close()
    return res

def upsert_image_record(
    library_id: int,
    file_path: str,
    file_name: str,
    file_size: int,
    mtime: float,
    header_hash: str,
    width: int,
    height: int,
    captured_at: Optional[str] = None,
    cur: Optional[sqlite3.Cursor] = None
) -> int:
    should_close = False
    if cur is None:
        conn = get_connection()
        cur = conn.cursor()
        should_close = True
    else:
        conn = None

    try:
        cur.execute("""
            INSERT INTO images (library_id, file_path, file_name, file_size, mtime, header_hash, width, height, captured_at, has_embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            ON CONFLICT(file_path) DO UPDATE SET
                file_size = excluded.file_size,
                mtime = excluded.mtime,
                header_hash = excluded.header_hash,
                width = excluded.width,
                height = excluded.height,
                captured_at = COALESCE(excluded.captured_at, images.captured_at),
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """, (library_id, file_path, file_name, file_size, mtime, header_hash, width, height, captured_at))
        
        img_id = cur.fetchone()["id"]
        
        if should_close and conn:
            conn.commit()
    finally:
        if should_close and conn:
            conn.close()
    return img_id

def save_image_embedding(image_id: int, embedding: np.ndarray, cur: Optional[sqlite3.Cursor] = None):
    should_close = False
    if cur is None:
        conn = get_connection()
        cur = conn.cursor()
        should_close = True
    else:
        conn = None

    try:
        emb_bytes = embedding.astype(np.float32).tobytes()
        cur.execute("DELETE FROM vec_images WHERE image_id = ?", (image_id,))
        cur.execute("INSERT INTO vec_images (image_id, embedding) VALUES (?, ?)", (image_id, emb_bytes))
        cur.execute("UPDATE images SET has_embedding = 1 WHERE id = ?", (image_id,))
        if should_close and conn:
            conn.commit()
    finally:
        if should_close and conn:
            conn.close()

def save_face(
    image_id: int,
    person_id: Optional[int],
    bbox: List[float],
    det_score: float,
    crop_thumb: str,
    embedding: np.ndarray,
    cur: Optional[sqlite3.Cursor] = None
) -> int:
    should_close = False
    if cur is None:
        conn = get_connection()
        cur = conn.cursor()
        should_close = True
    else:
        conn = None

    try:
        cur.execute("""
            INSERT INTO faces (image_id, person_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_score, crop_thumb)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (image_id, person_id, bbox[0], bbox[1], bbox[2], bbox[3], det_score, crop_thumb))
        face_id = cur.lastrowid
        
        emb_bytes = embedding.astype(np.float32).tobytes()
        cur.execute("INSERT INTO vec_faces (face_id, embedding) VALUES (?, ?)", (face_id, emb_bytes))

        if should_close and conn:
            conn.commit()
    finally:
        if should_close and conn:
            conn.close()
    return face_id

# --- Person & Cluster Operations ---
def create_person(name: str, cover_face_id: Optional[int] = None, cur: Optional[sqlite3.Cursor] = None) -> int:
    should_close = False
    if cur is None:
        conn = get_connection()
        cur = conn.cursor()
        should_close = True
    else:
        conn = None

    try:
        cur.execute("INSERT INTO persons (name, cover_face_id) VALUES (?, ?)", (name, cover_face_id))
        pid = cur.lastrowid

        if should_close and conn:
            conn.commit()
    finally:
        if should_close and conn:
            conn.close()
    return pid

def cleanup_orphan_persons():
    """Remove person records that have no associated faces. Call after merge/delete operations only."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM persons WHERE id NOT IN (SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL)")
    conn.commit()
    conn.close()

def get_persons() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT p.id, p.name, p.cover_face_id,
               COUNT(f.id) as face_count,
               f_cover.crop_thumb as cover_thumb
        FROM persons p
        JOIN faces f ON p.id = f.person_id
        LEFT JOIN faces f_cover ON p.cover_face_id = f_cover.id
        GROUP BY p.id
        HAVING face_count > 0
        ORDER BY face_count DESC, p.id ASC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def rename_person(person_id: int, new_name: str) -> Dict[str, Any]:
    cleaned_name = new_name.strip()
    conn = get_connection()
    cur = conn.cursor()
    
    # Check if another person already has this exact name (case-insensitive)
    cur.execute("SELECT id FROM persons WHERE LOWER(name) = LOWER(?) AND id != ?", (cleaned_name, person_id))
    existing = cur.fetchone()
    if existing:
        existing_id = existing["id"]
        # Auto-merge: reassign faces of person_id to existing_id
        cur.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (existing_id, person_id))
        cur.execute("DELETE FROM persons WHERE id = ?", (person_id,))
        conn.commit()
        conn.close()
        return {"status": "merged", "id": existing_id, "name": cleaned_name}
    else:
        cur.execute("UPDATE persons SET name = ? WHERE id = ?", (cleaned_name, person_id))
        conn.commit()
        conn.close()
        return {"status": "renamed", "id": person_id, "name": cleaned_name}

def merge_persons(target_person_id: int, source_person_ids: List[int]):
    if not source_person_ids:
        return
    conn = get_connection()
    try:
        cur = conn.cursor()
        chunk_size = 900
        for i in range(0, len(source_person_ids), chunk_size):
            chunk = source_person_ids[i:i + chunk_size]
            placeholders = ",".join("?" for _ in chunk)
            cur.execute(f"UPDATE faces SET person_id = ? WHERE person_id IN ({placeholders})", [target_person_id] + chunk)
            cur.execute(f"DELETE FROM persons WHERE id IN ({placeholders})", chunk)
        conn.commit()
    finally:
        conn.close()

def assign_face_to_person(face_id: int, person_id: int):
    conn = get_connection()
    conn.execute("UPDATE faces SET person_id = ? WHERE id = ?", (person_id, face_id))
    conn.commit()
    conn.close()

def find_closest_person_for_face(
    embedding: np.ndarray,
    distance_threshold: float = 0.50,
    cur: Optional[sqlite3.Cursor] = None
) -> Optional[int]:
    """Finds existing person whose known face is closest within distance threshold."""
    should_close = False
    if cur is None:
        conn = get_connection()
        cur = conn.cursor()
        should_close = True
    else:
        conn = None

    try:
        emb_bytes = embedding.astype(np.float32).tobytes()

        cur.execute("""
            SELECT vf.face_id, vf.distance, f.person_id
            FROM vec_faces vf
            JOIN faces f ON vf.face_id = f.id
            WHERE vf.embedding MATCH ? AND k = 10 AND f.person_id IS NOT NULL
            ORDER BY vf.distance ASC
            LIMIT 1
        """, (emb_bytes,))
        row = cur.fetchone()
        
        if row and row["distance"] <= distance_threshold:
            return row["person_id"]
        return None
    except sqlite3.OperationalError:
        return None
    finally:
        if should_close and conn:
            conn.close()

# --- Search Operations ---
def search_images_semantic(
    query_emb: np.ndarray,
    limit: int = 50,
    min_score: float = -1.0,
    filter_image_ids: Optional[List[int]] = None,
    library_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    min_size_kb: Optional[int] = None,
    max_size_kb: Optional[int] = None,
    adaptive_cutoff: bool = True
) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        q_bytes = query_emb.astype(np.float32).tobytes()
        
        # Fetch top candidates from vec_images
        k_val = min(2000, max(limit * 8, 300))
        cur.execute("""
            SELECT v.image_id, v.distance, i.library_id, i.file_path, i.file_name, i.width, i.height, i.header_hash, i.captured_at, i.file_size, i.ocr_text
            FROM vec_images v
            JOIN images i ON v.image_id = i.id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance ASC
        """, (q_bytes, k_val))
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()

    filter_set = set(filter_image_ids) if filter_image_ids is not None else None

    results = []
    top_sim = None

    for r in rows:
        dist = r["distance"]
        sim = max(-1.0, min(1.0, 1.0 - (dist * dist / 2.0)))

        # 1. Score threshold filter
        if sim < min_score:
            continue

        # 2. Face entity candidate filter
        if filter_set is not None and r["image_id"] not in filter_set:
            continue

        # 3. Album filter
        if library_id is not None and r["library_id"] != library_id:
            continue

        # 4. Date filter
        cap_at = r["captured_at"]
        if from_date and cap_at:
            if cap_at[:10] < from_date:
                continue
        if to_date and cap_at:
            if cap_at[:10] > to_date:
                continue

        # 5. File size filter
        fsize_kb = r["file_size"] / 1024.0
        if min_size_kb and fsize_kb < min_size_kb:
            continue
        if max_size_kb and fsize_kb > max_size_kb:
            continue

        if top_sim is None:
            top_sim = sim

        # 6. Adaptive knee cutoff: truncate false-positive trailing noise
        # If the leading candidate has strong similarity (>= 0.075), cut off when score drops by > 0.032 or hits floor
        if adaptive_cutoff and top_sim >= 0.075:
            if sim < max(0.065, top_sim - 0.032):
                break

        # Calibrated match confidence (stretching [0.05, 0.12] onto [0.0, 1.0])
        conf = max(0.0, min(1.0, (sim - 0.05) / 0.07))

        results.append({
            "id": r["image_id"],
            "image_id": r["image_id"],
            "file_path": r["file_path"],
            "file_name": r["file_name"],
            "width": r["width"],
            "height": r["height"],
            "header_hash": r["header_hash"],
            "captured_at": r["captured_at"],
            "file_size": r["file_size"],
            "distance": dist,
            "similarity": round(float(sim), 4),
            "confidence": round(float(conf), 3),
            "ocr_text": r["ocr_text"]
        })

        if len(results) >= limit:
            break

    return results

def get_image_ids_with_person(person_id: int) -> List[int]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT image_id FROM faces WHERE person_id = ?", (person_id,))
    ids = [r["image_id"] for r in cur.fetchall()]
    conn.close()
    return ids

# --- OCR Operations ---
def save_image_ocr(image_id: int, text_content: str, cur: Optional[sqlite3.Cursor] = None):
    should_close = False
    if cur is None:
        conn = get_connection()
        cur = conn.cursor()
        should_close = True
    else:
        conn = None

    cleaned_text = (text_content or "").strip()
    cur.execute("UPDATE images SET ocr_text = ? WHERE id = ?", (cleaned_text, image_id))
    cur.execute("DELETE FROM images_fts WHERE image_id = ?", (image_id,))
    if cleaned_text:
        cur.execute("INSERT INTO images_fts (image_id, text_content) VALUES (?, ?)", (image_id, cleaned_text))

    if should_close and conn:
        conn.commit()
        conn.close()

def search_images_ocr(
    query_str: Optional[str] = None,
    quoted_terms: Optional[List[str]] = None,
    limit: int = 50,
    library_id: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    min_size_kb: Optional[int] = None,
    max_size_kb: Optional[int] = None
) -> List[Dict[str, Any]]:
    # 1. Determine search terms
    terms_to_match = []
    if quoted_terms:
        terms_to_match = [t.strip() for t in quoted_terms if t and t.strip()]
    elif query_str:
        cleaned = query_str.strip()
        # Check if query_str itself has quotes
        in_quotes = re.findall(r'"([^"]+)"', cleaned)
        if in_quotes:
            terms_to_match = [t.strip() for t in in_quotes if t.strip()]
        else:
            terms_to_match = cleaned.split()

    if not terms_to_match:
        return []

    # 2. Build precision FTS5 query
    clauses = []
    for term in terms_to_match:
        # Sanitize while preserving spaces for phrase searches
        clean_term = re.sub(r'[^a-zA-Z0-9_\u00C0-\u017F\s]', '', term).strip()
        if not clean_term:
            continue
        words_in_term = clean_term.split()
        if len(words_in_term) > 1:
            # Multi-word phrase search: exact phrase match
            phrase = " ".join(words_in_term)
            clauses.append(f'"{phrase}"')
        else:
            w = words_in_term[0]
            # Exact word match for short words (<= 4 chars) to prevent "car" matching "card" or "care"
            if len(w) <= 4:
                clauses.append(f'"{w}"')
            else:
                # Prefix matching permitted for longer words
                clauses.append(f'"{w}"*')

    if not clauses:
        return []

    match_query = " AND ".join(clauses)

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT count(*) as cnt FROM images_fts")
    if cur.fetchone()["cnt"] == 0:
        conn.close()
        return []

    where_clauses = ["images_fts MATCH ?"]
    params: List[Any] = [match_query]

    if library_id is not None:
        where_clauses.append("i.library_id = ?")
        params.append(library_id)
    if from_date:
        where_clauses.append("i.captured_at >= ?")
        params.append(from_date)
    if to_date:
        where_clauses.append("i.captured_at <= ?")
        params.append(to_date + " 23:59:59")
    if min_size_kb:
        where_clauses.append("i.file_size >= ?")
        params.append(min_size_kb * 1024)
    if max_size_kb:
        where_clauses.append("i.file_size <= ?")
        params.append(max_size_kb * 1024)

    where_sql = " AND ".join(where_clauses)
    params.append(limit)

    sql = f"""
        SELECT fts.image_id, fts.text_content, fts.rank,
               i.library_id, i.file_path, i.file_name, i.width, i.height, i.header_hash, i.captured_at, i.file_size
        FROM images_fts fts
        JOIN images i ON fts.image_id = i.id
        WHERE {where_sql}
        ORDER BY fts.rank ASC
        LIMIT ?
    """

    try:
        cur.execute(sql, params)
        rows = cur.fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r["image_id"],
                "image_id": r["image_id"],
                "file_path": r["file_path"],
                "file_name": r["file_name"],
                "width": r["width"],
                "height": r["height"],
                "header_hash": r["header_hash"],
                "captured_at": r["captured_at"],
                "similarity": 1.0,
                "confidence": 1.0,
                "ocr_match": True,
                "ocr_text": r["text_content"]
            })
        return results
    except Exception:
        return []
    finally:
        conn.close()

def get_images_missing_ocr(limit: int = 500) -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT i.id, i.file_path, v.embedding
        FROM images i
        JOIN vec_images v ON i.id = v.image_id
        WHERE i.ocr_text IS NULL
        LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def get_gallery_images(library_id: Optional[int] = None, sort: str = "date_desc", limit: int = 100, offset: int = 0) -> Dict[str, Any]:
    conn = get_connection()
    cur = conn.cursor()
    where_clause = ""
    params = []
    if library_id:
        where_clause = "WHERE library_id = ?"
        params.append(library_id)

    order_by = "ORDER BY captured_at DESC, id DESC"
    if sort == "date_asc":
        order_by = "ORDER BY captured_at ASC, id ASC"
    elif sort == "name_asc":
        order_by = "ORDER BY file_name ASC"
    elif sort == "size_desc":
        order_by = "ORDER BY file_size DESC"

    cur.execute(f"SELECT COUNT(*) as count FROM images {where_clause}", params)
    total = cur.fetchone()["count"]

    cur.execute(f"""
        SELECT id, library_id, file_path, file_name, file_size, width, height, header_hash, captured_at, ocr_text
        FROM images
        {where_clause}
        {order_by}
        LIMIT ? OFFSET ?
    """, params + [limit, offset])
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return {"total": total, "images": rows}

def get_timeline_groups(library_id: Optional[int] = None, limit: int = 2000) -> List[Dict[str, Any]]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        where_clause = "WHERE captured_at IS NOT NULL"
        params = []
        if library_id:
            where_clause += " AND library_id = ?"
            params.append(library_id)

        # Added LIMIT to prevent massive memory spikes
        cur.execute(f"""
            SELECT strftime('%Y-%m', captured_at) as period,
                   strftime('%Y', captured_at) as year,
                   strftime('%m', captured_at) as month,
                   id, file_path, file_name, width, height, header_hash, captured_at
            FROM images
            {where_clause}
            ORDER BY captured_at DESC, id DESC
            LIMIT ?
        """, params + [limit])
        rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()

    from collections import OrderedDict
    groups = OrderedDict()
    for r in rows:
        p = r["period"] or "Unknown"
        if p not in groups:
            groups[p] = {
                "period": p,
                "year": r["year"] or "Unknown",
                "month": r["month"] or "Unknown",
                "images": []
            }
        groups[p]["images"].append(r)

    return list(groups.values())

def get_stats() -> Dict[str, Any]:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) as count FROM images")
        total_images = cur.fetchone()["count"]
        cur.execute("SELECT COUNT(*) as count FROM images WHERE has_embedding = 1")
        tagged_images = cur.fetchone()["count"]
        cur.execute("SELECT COUNT(*) as count FROM faces")
        total_faces = cur.fetchone()["count"]
        # Optimized: query persons table directly instead of distinct face scan
        cur.execute("SELECT COUNT(*) as count FROM persons")
        total_persons = cur.fetchone()["count"]
        cur.execute("SELECT COUNT(*) as count FROM libraries")
        total_libraries = cur.fetchone()["count"]
    finally:
        conn.close()
    return {
        "total_images": total_images,
        "tagged_images": tagged_images,
        "total_faces": total_faces,
        "total_persons": total_persons,
        "total_libraries": total_libraries
    }

def get_duplicate_groups(similarity_threshold: float = 0.96, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Returns groups of identical or near-duplicate images.
    Combines exact header/hash matches (100% identical) and SigLIP-2 vector similarity (bursts/re-encodes).
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        duplicate_groups = []
        seen_image_ids = set()

        # 1. Exact hash duplicates (100% duplicate content)
        cur.execute("""
            SELECT header_hash, COUNT(*) as cnt
            FROM images
            GROUP BY header_hash
            HAVING cnt > 1
            LIMIT ?
        """, (limit,))
        exact_hashes = [r["header_hash"] for r in cur.fetchall()]

        for hhash in exact_hashes:
            cur.execute("""
                SELECT id, library_id, file_path, file_name, file_size, width, height, header_hash, captured_at
                FROM images
                WHERE header_hash = ?
                ORDER BY id ASC
            """, (hhash,))
            imgs = [dict(r) for r in cur.fetchall()]
            if len(imgs) > 1:
                duplicate_groups.append({
                    "group_id": f"exact_{hhash[:12]}",
                    "type": "exact",
                    "similarity": 1.0,
                    "confidence_pct": 100,
                    "potential_savings_bytes": sum(img["file_size"] for img in imgs[1:]),
                    "images": imgs
                })
                for img in imgs:
                    seen_image_ids.add(img["id"])

        if len(duplicate_groups) >= limit:
            return duplicate_groups[:limit]

        # 2. Near-duplicates via SigLIP-2 vector distance (burst photos / crops / resized)
        max_dist = (2.0 * max(0.0, 1.0 - similarity_threshold)) ** 0.5

        cur.execute("""
            SELECT id, library_id, file_path, file_name, file_size, width, height, header_hash, captured_at
            FROM images
            WHERE has_embedding = 1
            ORDER BY captured_at DESC, id DESC
            LIMIT 400
        """)
        candidates = [dict(r) for r in cur.fetchall()]
        candidate_map = {img["id"]: img for img in candidates}

        for img in candidates:
            if len(duplicate_groups) >= limit:
                break
            iid = img["id"]
            if iid in seen_image_ids:
                continue

            cur.execute("""
                SELECT image_id, distance
                FROM vec_images
                WHERE embedding MATCH (SELECT embedding FROM vec_images WHERE image_id = ?)
                  AND k = 4
            """, (iid,))
            neighbors = cur.fetchall()

            cluster_imgs = [img]
            best_sim = 1.0
            for n in neighbors:
                nid = n["image_id"]
                dist = n["distance"]
                if nid != iid and dist <= max_dist and nid not in seen_image_ids:
                    if nid in candidate_map:
                        n_img = candidate_map[nid]
                    else:
                        cur.execute("""
                            SELECT id, library_id, file_path, file_name, file_size, width, height, header_hash, captured_at
                            FROM images WHERE id = ?
                        """, (nid,))
                        n_row = cur.fetchone()
                        if not n_row:
                            continue
                        n_img = dict(n_row)
                        candidate_map[nid] = n_img

                    sim = round(1.0 - (dist ** 2) / 2.0, 3)
                    best_sim = min(best_sim, sim)
                    cluster_imgs.append(n_img)
                    seen_image_ids.add(nid)

            if len(cluster_imgs) > 1:
                seen_image_ids.add(iid)
                cluster_imgs.sort(key=lambda x: x["file_size"], reverse=True)
                duplicate_groups.append({
                    "group_id": f"near_{iid}",
                    "type": "near_duplicate",
                    "similarity": round(best_sim, 3),
                    "confidence_pct": int(best_sim * 100),
                    "potential_savings_bytes": sum(x["file_size"] for x in cluster_imgs[1:]),
                    "images": cluster_imgs
                })

        return duplicate_groups[:limit]
    finally:
        conn.close()


def get_face_merge_suggestions(min_similarity: float = 0.65, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Legacy single-cover-face merge suggestion. Delegates to the centroid-based
    implementation with a floor of 0.82 to avoid low-precision matches.
    """
    return get_face_merge_suggestions_centroid(
        min_similarity=max(float(min_similarity), 0.82),
        limit=limit
    )


def get_face_merge_suggestions_centroid(min_similarity: float = 0.82, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Centroid-based face cluster merge suggestions.

    Algorithm:
    1. For every person cluster, load all ArcFace 512-D unit-norm embeddings from vec_faces.
    2. Compute the cluster centroid as the L2-normalized mean of all embeddings.
    3. Compute intra-cluster cohesion (mean pairwise cosine sim up to 10 faces) as a quality weight.
    4. For every unique pair of clusters, compute centroid-to-centroid cosine similarity.
    5. Effective confidence = centroid_sim * (0.5 + 0.5 * min_cluster_cohesion).
       This down-ranks suggestions where one cluster is internally noisy.
    6. Return pairs where centroid cosine sim >= min_similarity, ranked descending.

    Why centroid is better than single-cover-face KNN:
    - A single cover face can be a pose/lighting outlier that misrepresents the cluster.
    - Centroid averages out noise; two clusters of the same person consistently show
      centroid cosine sim > 0.85 for ArcFace w2_r100 (buffalo_l).
    - The threshold 0.82 on centroid similarity corresponds to very high precision on
      real-world personal photo library data with the InsightFace buffalo_l model.

    No re-indexing required -- uses the existing stored 512-D embeddings in vec_faces.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()

        # Load all person metadata. Cap at 300 for performance.
        cur.execute("""
            SELECT p.id as person_id, p.name,
                   COALESCE(p.cover_face_id, MIN(f.id)) as cover_face_id,
                   COUNT(f.id) as face_count,
                   MIN(i.header_hash) as cover_hash
            FROM persons p
            JOIN faces f ON p.id = f.person_id
            JOIN images i ON f.image_id = i.id
            GROUP BY p.id
            ORDER BY face_count DESC, p.id ASC
            LIMIT 300
        """)
        persons = [dict(r) for r in cur.fetchall()]
        if len(persons) < 2:
            return []

        # Build centroid for each cluster
        centroids: List[Dict] = []
        for p in persons:
            pid = p["person_id"]
            cur.execute("""
                SELECT vf.embedding
                FROM vec_faces vf
                JOIN faces f ON vf.face_id = f.id
                WHERE f.person_id = ?
            """, (pid,))
            rows = cur.fetchall()
            if not rows:
                continue

            embs = []
            for r in rows:
                raw = bytes(r["embedding"])
                emb = np.frombuffer(raw, dtype=np.float32).copy()
                if len(emb) == 512:
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        embs.append(emb / norm)

            if not embs:
                continue

            stack = np.stack(embs)  # (N, 512)
            mean_emb = stack.mean(axis=0)
            mean_norm = np.linalg.norm(mean_emb)
            if mean_norm < 1e-6:
                continue
            centroid = (mean_emb / mean_norm).astype(np.float32)

            # Intra-cluster cohesion: mean pairwise cosine similarity (up to first 10 faces)
            if len(embs) >= 2:
                sub = stack[:min(len(embs), 10)]
                gram_intra = sub @ sub.T
                iu = np.triu_indices(len(sub), k=1)
                cohesion = float(gram_intra[iu].mean())
            else:
                cohesion = 1.0  # singleton: treat as perfectly coherent

            centroids.append({
                "person_id": pid,
                "name": p["name"],
                "face_count": p["face_count"],
                "cover_hash": p["cover_hash"],
                "centroid": centroid,
                "cohesion": cohesion,
            })

        if len(centroids) < 2:
            return []

        # All-pairs centroid cosine similarity via one matrix multiply
        cmat = np.stack([c["centroid"] for c in centroids])  # (M, 512)
        gram = cmat @ cmat.T  # (M, M)

        suggestions = []
        seen_pairs: set = set()
        n_c = len(centroids)

        for i in range(n_c):
            for j in range(i + 1, n_c):
                sim = float(gram[i, j])
                if sim < min_similarity:
                    continue
                ci = centroids[i]
                cj = centroids[j]
                pair = (min(ci["person_id"], cj["person_id"]),
                        max(ci["person_id"], cj["person_id"]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)

                # Effective confidence discounts noisy clusters
                min_cohesion = min(ci["cohesion"], cj["cohesion"])
                effective_conf = sim * (0.5 + 0.5 * min_cohesion)

                # Primary = named or larger cluster
                ci_default = ci["name"].startswith("Person ")
                cj_default = cj["name"].startswith("Person ")
                if not ci_default and cj_default:
                    keep, merge = ci, cj
                elif ci_default and not cj_default:
                    keep, merge = cj, ci
                elif ci["face_count"] >= cj["face_count"]:
                    keep, merge = ci, cj
                else:
                    keep, merge = cj, ci

                suggestions.append({
                    "primary_person": {
                        "id": keep["person_id"],
                        "name": keep["name"],
                        "face_count": keep["face_count"],
                        "cover_thumb": f"faces/{keep['cover_hash']}_0.webp",
                    },
                    "candidate_person": {
                        "id": merge["person_id"],
                        "name": merge["name"],
                        "face_count": merge["face_count"],
                        "cover_thumb": f"faces/{merge['cover_hash']}_0.webp",
                    },
                    "centroid_similarity": round(sim, 4),
                    "similarity": round(sim, 4),
                    "confidence_pct": int(effective_conf * 100),
                    "primary_cohesion": round(keep["cohesion"], 3),
                    "candidate_cohesion": round(merge["cohesion"], 3),
                })

        suggestions.sort(key=lambda x: x["similarity"], reverse=True)
        return suggestions[:limit]
    finally:
        conn.close()

