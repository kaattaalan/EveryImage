import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src.config as cfg

import os
import subprocess
import io
import re
from typing import List, Optional
import asyncio
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
import pillow_heif
pillow_heif.register_heif_opener()

import src.database as db
from src.scanner import IndexingEngine
from src.models import SiglipEngine
import contextlib

app = FastAPI(title="EveryImage Desktop API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB on startup
@app.on_event("startup")
def startup_event():
    db.init_db()

# Mount thumbnails folder
app.mount("/thumbnails", StaticFiles(directory=str(cfg.CACHE_THUMBNAILS)), name="thumbnails")

# Models for request validation
class AddLibraryRequest(BaseModel):
    path: str

class RenamePersonRequest(BaseModel):
    name: str

class MergePersonsRequest(BaseModel):
    target_id: int
    source_ids: List[int]

class OpenImageRequest(BaseModel):
    file_path: str
    reveal: bool = False

# --- Endpoints ---

@app.get("/api/stats")
def get_stats():
    return db.get_stats()

@app.get("/api/scan/status")
def get_scan_status():
    engine = IndexingEngine.get_instance()
    return engine.get_status()

@app.post("/api/scan/start")
def start_scan(library_ids: Optional[List[int]] = None):
    engine = IndexingEngine.get_instance()
    success = engine.start(library_ids)
    if not success:
        raise HTTPException(status_code=400, detail="Indexing is already in progress")
    return {"status": "started"}

@app.post("/api/scan/stop")
def stop_scan():
    engine = IndexingEngine.get_instance()
    success = engine.stop()
    if not success:
        raise HTTPException(status_code=400, detail="Indexing is not currently running")
    return {"status": "stopping"}

class RebuildRequest(BaseModel):
    keep_libraries: bool = True

@app.post("/api/database/rebuild")
def rebuild_database(req: Optional[RebuildRequest] = None):
    keep_libs = req.keep_libraries if req else True
    engine = IndexingEngine.get_instance()
    if engine.state in ("crawling", "indexing"):
        engine.stop()
        if engine.worker_thread and engine.worker_thread.is_alive():
            engine.worker_thread.join(timeout=10.0)

    db.clear_database(keep_libraries=keep_libs)
    started = engine.start()
    return {"status": "rebuilding", "started": started}

@app.get("/api/libraries")
def list_libraries():
    return db.get_libraries()

@app.post("/api/libraries")
def add_library(req: AddLibraryRequest):
    p = Path(req.path.strip('"').strip("'"))
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail="Provided path does not exist or is not a directory")
    res = db.add_library(str(p))
    return res

@app.get("/api/gallery")
def get_gallery(
    page: int = Query(1, ge=1),
    page_size: int = Query(60, ge=1, le=500),
    library_id: Optional[int] = None,
    sort: str = Query("date_desc")
):
    offset = (page - 1) * page_size
    return db.get_gallery_images(library_id=library_id, sort=sort, limit=page_size, offset=offset)

@app.get("/api/timeline")
def get_timeline(library_id: Optional[int] = None):
    return db.get_timeline_groups(library_id=library_id)

@app.delete("/api/libraries/{lib_id}")
def delete_library(lib_id: int):
    db.remove_library(lib_id)
    db.cleanup_orphan_persons()
    return {"status": "deleted"}

@app.get("/api/faces/persons")
def list_persons():
    return db.get_persons()

@app.get("/api/faces/persons/{person_id}/images")
def get_person_images(person_id: int):
    # Retrieve base images, avoiding injecting huge IN clauses
    with contextlib.closing(db.get_connection()) as conn:
        cur = conn.cursor()
        cur.execute(f"""
            SELECT i.id, i.library_id, i.file_path, i.file_name, i.file_size, i.width, i.height, i.header_hash, i.captured_at
            FROM images i
            JOIN faces f ON i.id = f.image_id
            WHERE f.person_id = ?
            GROUP BY i.id
            ORDER BY i.captured_at DESC, i.id DESC
        """, (person_id,))
        rows = [dict(r) for r in cur.fetchall()]

    for r in rows:
        r["similarity"] = 1.0
        r["distance"] = 0.0
    return rows

@app.post("/api/faces/persons/{person_id}/rename")
def rename_person(person_id: int, req: RenamePersonRequest):
    res = db.rename_person(person_id, req.name)
    return res

@app.post("/api/faces/merge")
def merge_persons(req: MergePersonsRequest):
    if req.target_id in req.source_ids:
        req.source_ids.remove(req.target_id)
    if not req.source_ids:
        raise HTTPException(status_code=400, detail="No distinct source persons to merge")
    db.merge_persons(req.target_id, req.source_ids)
    db.cleanup_orphan_persons()
    return {"status": "merged", "target_id": req.target_id}

@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(60, ge=1, le=500),
    min_score: float = Query(-1.0),
    negative_prompt: Optional[str] = Query(None),
    library_id: Optional[int] = Query(None),
    person_id: Optional[int] = Query(None),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    min_size_kb: Optional[int] = Query(None)
):
    raw_query = q.strip()
    
    # 1. Parse negative quoted terms: e.g. -"shawaya"
    neg_quoted_terms = [t.strip() for t in re.findall(r'-"([^"]+)"', raw_query) if t.strip()]

    # 2. Parse positive quoted terms (not preceded by -): e.g. "shawaya"
    temp_query = re.sub(r'-"([^"]+)"', '', raw_query)
    quoted_terms = [t.strip() for t in re.findall(r'"([^"]+)"', temp_query) if t.strip()]

    # 3. Remove all quoted blocks to get unquoted query
    unquoted_query = re.sub(r'-?"[^"]*"', '', raw_query).strip()

    # 4. Check for inline negative syntax (e.g. "car -red -truck")
    inline_negatives = []
    positive_parts = []
    for token in unquoted_query.split():
        if token.startswith("-") and len(token) > 1:
            inline_negatives.append(token[1:])
        else:
            positive_parts.append(token)
    
    positive_query = " ".join(positive_parts).strip().lower()
    combined_negative = " ".join(inline_negatives).strip().lower()
    if negative_prompt and negative_prompt.strip():
        combined_negative = (combined_negative + " " + negative_prompt.strip().lower()).strip()

    # Exclude IDs containing negative OCR terms if specified
    excluded_ocr_ids = set()
    if neg_quoted_terms:
        neg_ocr_matches = db.search_images_ocr(quoted_terms=neg_quoted_terms, limit=2000)
        excluded_ocr_ids = {m["image_id"] for m in neg_ocr_matches}

    # Case 1: Pure Quoted OCR Search (e.g. "shawaya", "Club Shawaya")
    if quoted_terms and not positive_query and not person_id:
        ocr_results = db.search_images_ocr(
            quoted_terms=quoted_terms,
            limit=limit,
            library_id=library_id,
            from_date=from_date,
            to_date=to_date,
            min_size_kb=min_size_kb
        )
        if excluded_ocr_ids:
            ocr_results = [r for r in ocr_results if r["image_id"] not in excluded_ocr_ids]
        return {
            "query": q,
            "matched_person": None,
            "scene_prompt": None,
            "negative_prompt": None,
            "is_person_only": False,
            "ocr_only": True,
            "ocr_matches_count": len(ocr_results),
            "total_matches": len(ocr_results),
            "results": ocr_results
        }

    # 3. Entity Recognition: check if query references a recognized Person
    all_persons = db.get_persons()
    matched_person = None
    remaining_scene_prompt = positive_query

    # Direct person ID provided
    if person_id:
        matched_person = next((p for p in all_persons if p["id"] == person_id), None)
        if matched_person:
            p_name = matched_person["name"].lower()
            remaining_scene_prompt = positive_query.replace(p_name, "").strip()

    # Fuzzy match person name if not supplied by ID
    if not matched_person:
        sorted_persons = sorted([p for p in all_persons if len(p["name"]) >= 2], key=lambda x: len(x["name"]), reverse=True)
        for p in sorted_persons:
            p_name = p["name"].lower()
            if p_name in positive_query:
                matched_person = p
                remaining_scene_prompt = positive_query.replace(p_name, "").strip()
                break

    # Clean up connector words like "near a", "near", "with a", "with", "in a", "in", "at a", "at", "by a", "by"
    if remaining_scene_prompt:
        for connector in ["near a", "near", "with a", "with", "in a", "in", "at a", "at", "by a", "by"]:
            if remaining_scene_prompt.startswith(connector):
                remaining_scene_prompt = remaining_scene_prompt[len(connector):].strip()

    filter_image_ids = None
    if matched_person:
        filter_image_ids = db.get_image_ids_with_person(matched_person["id"])
        if not filter_image_ids:
            return {
                "query": q,
                "matched_person": matched_person["name"],
                "scene_prompt": remaining_scene_prompt,
                "negative_prompt": combined_negative if combined_negative else None,
                "is_person_only": not bool(remaining_scene_prompt or quoted_terms),
                "total_matches": 0,
                "results": []
            }

        # If quoted OCR terms are also present with a person (e.g. Alice "certificate")
        if quoted_terms:
            ocr_person_matches = db.search_images_ocr(
                quoted_terms=quoted_terms,
                limit=limit,
                library_id=library_id,
                from_date=from_date,
                to_date=to_date,
                min_size_kb=min_size_kb
            )
            person_set = set(filter_image_ids)
            ocr_candidate_ids = [m["image_id"] for m in ocr_person_matches if m["image_id"] in person_set]
            if not ocr_candidate_ids:
                return {
                    "query": q,
                    "matched_person": matched_person["name"],
                    "scene_prompt": remaining_scene_prompt,
                    "negative_prompt": combined_negative if combined_negative else None,
                    "is_person_only": False,
                    "ocr_matches_count": 0,
                    "total_matches": 0,
                    "results": []
                }
            filter_image_ids = ocr_candidate_ids

        # Pure Person Search (no remaining scene text after matching person + optional quoted OCR)
        if not remaining_scene_prompt:
            with contextlib.closing(db.get_connection()) as conn:
                cur = conn.cursor()
                # Use a JOIN directly in SQLite if there's no OCR limit or we use filter_image_ids
                if filter_image_ids is not None:
                    # We still have filter_image_ids, but it could be > 999.
                    # Since it could be from OCR, we need to pass them.
                    # Wait, if we use temp table or just chunking, we can avoid the limit.
                    # Alternatively, if there's no OCR, we just join person_id.
                    if quoted_terms:
                        # Chunking since it's an intersection with OCR results
                        rows = []
                        chunk_size = 900
                        for i in range(0, len(filter_image_ids), chunk_size):
                            chunk = filter_image_ids[i:i + chunk_size]
                            placeholders = ",".join("?" for _ in chunk)
                            query_sql = f"""
                                SELECT id, library_id, file_path, file_name, file_size, width, height, header_hash, captured_at, ocr_text
                                FROM images
                                WHERE id IN ({placeholders})
                            """
                            params = list(chunk)
                            if library_id:
                                query_sql += " AND library_id = ?"
                                params.append(library_id)
                            if from_date:
                                query_sql += " AND captured_at >= ?"
                                params.append(from_date)
                            if to_date:
                                query_sql += " AND captured_at <= ?"
                                params.append(to_date)
                            if min_size_kb:
                                query_sql += " AND file_size >= ?"
                                params.append(min_size_kb * 1024)
                            query_sql += " ORDER BY captured_at DESC, id DESC LIMIT ?"
                            params.append(limit)
                            cur.execute(query_sql, params)
                            rows.extend([dict(r) for r in cur.fetchall()])
                            if len(rows) >= limit:
                                rows = rows[:limit]
                                break
                    else:
                        # Pure person search, no OCR quotes
                        query_sql = f"""
                            SELECT i.id, i.library_id, i.file_path, i.file_name, i.file_size, i.width, i.height, i.header_hash, i.captured_at, i.ocr_text
                            FROM images i
                            JOIN faces f ON i.id = f.image_id
                            WHERE f.person_id = ?
                        """
                        params = [matched_person["id"]]
                        if library_id:
                            query_sql += " AND i.library_id = ?"
                            params.append(library_id)
                        if from_date:
                            query_sql += " AND i.captured_at >= ?"
                            params.append(from_date)
                        if to_date:
                            query_sql += " AND i.captured_at <= ?"
                            params.append(to_date)
                        if min_size_kb:
                            query_sql += " AND i.file_size >= ?"
                            params.append(min_size_kb * 1024)
                        query_sql += " GROUP BY i.id ORDER BY i.captured_at DESC, i.id DESC LIMIT ?"
                        params.append(limit)
                        cur.execute(query_sql, params)
                        rows = [dict(r) for r in cur.fetchall()]

            if excluded_ocr_ids:
                rows = [r for r in rows if r["id"] not in excluded_ocr_ids]
            for r in rows:
                r["similarity"] = 1.0
                r["confidence"] = 1.0
                r["distance"] = 0.0
                if quoted_terms:
                    r["ocr_match"] = True

            return {
                "query": q,
                "matched_person": matched_person["name"],
                "scene_prompt": None,
                "negative_prompt": None,
                "is_person_only": not bool(quoted_terms),
                "total_matches": len(rows),
                "results": rows
            }

    # Case 2: Hybrid Multimodal (visual query + quoted OCR filter)
    # e.g. dinner "shawaya" -> search images with OCR "shawaya", then visually rank by "dinner"
    if quoted_terms and positive_query:
        ocr_matches = db.search_images_ocr(
            quoted_terms=quoted_terms,
            limit=limit,
            library_id=library_id,
            from_date=from_date,
            to_date=to_date,
            min_size_kb=min_size_kb
        )
        ocr_id_set = {m["image_id"] for m in ocr_matches}
        if filter_image_ids is not None:
            filter_image_ids = [i for i in filter_image_ids if i in ocr_id_set]
        else:
            filter_image_ids = list(ocr_id_set)

        if not filter_image_ids:
            return {
                "query": q,
                "matched_person": matched_person["name"] if matched_person else None,
                "scene_prompt": remaining_scene_prompt,
                "negative_prompt": combined_negative if combined_negative else None,
                "is_person_only": False,
                "ocr_matches_count": 0,
                "total_matches": 0,
                "results": []
            }

    # Case 3: Visual Semantic Search (SigLIP Vector Index)
    # Notice: If NO quoted terms are in query, OCR is NOT called at all! No accidental "car" -> "card" noise.
    siglip = SiglipEngine.get_instance()
    text_to_encode = remaining_scene_prompt if remaining_scene_prompt else positive_query
    q_emb = siglip.encode_text(text_to_encode)

    # Apply Negative Vector Projection if negative prompt is present
    if combined_negative:
        neg_emb = siglip.encode_text(combined_negative)
        q_emb = q_emb - 0.45 * neg_emb
        import numpy as np
        q_emb = q_emb / np.maximum(np.linalg.norm(q_emb), 1e-12)

    # Vector Search with all filters and adaptive knee cutoff
    matches = db.search_images_semantic(
        q_emb,
        limit=limit,
        min_score=min_score,
        filter_image_ids=filter_image_ids,
        library_id=library_id,
        from_date=from_date,
        to_date=to_date,
        min_size_kb=min_size_kb,
        adaptive_cutoff=True
    )

    if excluded_ocr_ids:
        matches = [item for item in matches if item["image_id"] not in excluded_ocr_ids]

    if quoted_terms:
        for item in matches:
            item["ocr_match"] = True

    return {
        "query": q,
        "matched_person": matched_person["name"] if matched_person else None,
        "scene_prompt": remaining_scene_prompt,
        "negative_prompt": combined_negative if combined_negative else None,
        "is_person_only": False,
        "ocr_matches_count": len(matches) if quoted_terms else 0,
        "total_matches": len(matches),
        "results": matches
    }

@app.get("/api/tags/suggestions")
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

async def decode_heic(p: Path) -> bytes:
    def _convert():
        with Image.open(p) as im:
            im_oriented = ImageOps.exif_transpose(im).convert("RGB")
            buf = io.BytesIO()
            im_oriented.save(buf, format="JPEG", quality=92)
            # Make sure to close the converted image as well
            im_oriented.close()
            return buf.getvalue()
    return await asyncio.to_thread(_convert)

@app.get("/api/images/{image_id}/file")
async def get_image_file(image_id: int):
    img = db.get_image_by_id(image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image record not found in database")
    p = Path(img["file_path"])
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Image file not found on disk: {p}")

    ext = p.suffix.lower()
    if ext in [".heic", ".heif"]:
        try:
            # We could cache to disk, but returning directly from RAM is fine if offloaded
            data = await decode_heic(p)
            return Response(content=data, media_type="image/jpeg")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not decode HEIC image: {e}")

    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp"
    }
    media_type = media_types.get(ext, "application/octet-stream")
    return FileResponse(str(p), media_type=media_type)

@app.get("/api/images/{image_id}/faces")
def get_image_faces(image_id: int):
    return db.get_image_faces(image_id)

@app.post("/api/images/open")
def open_image(req: OpenImageRequest):
    raw_path = req.file_path.strip().strip('"').strip("'")
    p = Path(raw_path).resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"File not found on disk: {str(p)}")
    try:
        norm_path = os.path.normpath(str(p))
        if req.reveal:
            # We must use proper arguments array for safety
            subprocess.Popen(["explorer.exe", f"/select,{norm_path}"])
        else:
            try:
                os.startfile(norm_path)
            except Exception:
                subprocess.Popen(["explorer.exe", norm_path])
        return {"status": "opened", "path": norm_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not open file: {e}")

# Serve frontend HTML
@app.get("/")
def serve_index():
    index_file = cfg.WORKSPACE_ROOT / "src" / "web" / "index.html"
    return FileResponse(index_file)
