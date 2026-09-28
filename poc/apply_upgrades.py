import re
from pathlib import Path

# =========================================================================
# 1. Update src/database.py
# =========================================================================
db_path = Path("src/database.py")
db_content = db_path.read_text(encoding="utf-8")

# Fix get_persons to filter HAVING face_count > 0 and clean orphans
old_get_persons = '''def get_persons() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT p.id, p.name, p.cover_face_id,
               COUNT(f.id) as face_count,
               f_cover.crop_thumb as cover_thumb
        FROM persons p
        LEFT JOIN faces f ON p.id = f.person_id
        LEFT JOIN faces f_cover ON p.cover_face_id = f_cover.id
        GROUP BY p.id
        ORDER BY face_count DESC, p.id ASC
    """)
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows'''

new_get_persons = '''def get_persons() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    # Clean up any zero-face orphan records first
    cur.execute("DELETE FROM persons WHERE id NOT IN (SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL)")
    conn.commit()
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
    return rows'''

db_content = db_content.replace(old_get_persons, new_get_persons)

# Fix rename_person to auto-merge if new name matches an existing person
old_rename = '''def rename_person(person_id: int, new_name: str):
    conn = get_connection()
    conn.execute("UPDATE persons SET name = ? WHERE id = ?", (new_name.strip(), person_id))
    conn.commit()
    conn.close()'''

new_rename = '''def rename_person(person_id: int, new_name: str) -> Dict[str, Any]:
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
        return {"status": "renamed", "id": person_id, "name": cleaned_name}'''

db_content = db_content.replace(old_rename, new_rename)
db_path.write_text(db_content, encoding="utf-8")
print("1. src/database.py updated successfully!")

# =========================================================================
# 2. Update src/api.py
# =========================================================================
api_path = Path("src/api.py")
api_content = api_path.read_text(encoding="utf-8")

# Fix open_image to be bulletproof on Windows
old_open_img = '''@app.post("/api/images/open")
def open_image(req: OpenImageRequest):
    p = Path(req.file_path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="File not found")
    if req.reveal:
        subprocess.run(["explorer.exe", f"/select,{str(p)}"])
    else:
        os.startfile(str(p))
    return {"status": "opened"}'''

new_open_img = '''@app.get("/api/tags/suggestions")
def get_tag_suggestions():
    persons = db.get_persons()
    person_names = [p["name"] for p in persons if not p["name"].startswith("Person ")]
    libs = db.get_libraries()
    album_names = [Path(l["path"]).name for l in libs]
    scene_tags = ["outdoor", "portrait", "nature", "car", "sunset", "dog", "city", "ocean", "document"]
    
    suggestions = []
    for name in person_names + album_names + scene_tags:
        if name and name not in suggestions:
            suggestions.append(name)
            if len(suggestions) >= 10:
                break
    return suggestions

@app.post("/api/images/open")
def open_image(req: OpenImageRequest):
    raw_path = req.file_path.strip().strip('"').strip("'")
    p = Path(raw_path).resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found on disk: {str(p)}")
    try:
        norm_path = os.path.normpath(str(p))
        if req.reveal:
            subprocess.Popen(f'explorer.exe /select,"{norm_path}"', shell=True)
        else:
            os.startfile(norm_path)
        return {"status": "opened", "path": norm_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not open file: {e}")'''

api_content = api_content.replace(old_open_img, new_open_img)

# Update rename endpoint in api.py
old_rename_ep = '''@app.post("/api/faces/persons/{person_id}/rename")
def rename_person(person_id: int, req: RenamePersonRequest):
    db.rename_person(person_id, req.name)
    return {"status": "updated", "id": person_id, "name": req.name}'''

new_rename_ep = '''@app.post("/api/faces/persons/{person_id}/rename")
def rename_person(person_id: int, req: RenamePersonRequest):
    res = db.rename_person(person_id, req.name)
    return res'''

api_content = api_content.replace(old_rename_ep, new_rename_ep)
api_path.write_text(api_content, encoding="utf-8")
print("2. src/api.py updated successfully!")
