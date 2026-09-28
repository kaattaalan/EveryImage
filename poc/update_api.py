from pathlib import Path

api_path = Path("src/api.py")
content = api_path.read_text(encoding="utf-8")

start_marker = '@app.get("/api/search")'
end_marker = '@app.post("/api/images/open")'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    raise ValueError(f"Could not find markers: {start_idx}, {end_idx}")

new_search_code = '''@app.get("/api/search")
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(60, ge=1, le=500),
    min_score: float = Query(-1.0),
    negative_prompt: Optional[str] = Query(None),
    library_id: Optional[int] = Query(None),
    from_date: Optional[str] = Query(None),
    to_date: Optional[str] = Query(None),
    min_size_kb: Optional[int] = Query(None)
):
    raw_query = q.strip()
    
    # Check for inline negative syntax (e.g. "car -red -truck")
    inline_negatives = []
    positive_parts = []
    for token in raw_query.split():
        if token.startswith("-") and len(token) > 1:
            inline_negatives.append(token[1:])
        else:
            positive_parts.append(token)
    
    positive_query = " ".join(positive_parts).strip().lower()
    combined_negative = " ".join(inline_negatives).strip().lower()
    if negative_prompt and negative_prompt.strip():
        combined_negative = (combined_negative + " " + negative_prompt.strip().lower()).strip()

    # 1. Entity Recognition: check if query contains any known person name
    all_persons = db.get_persons()
    matched_person = None
    remaining_scene_prompt = positive_query

    # Match longest person name first
    sorted_persons = sorted([p for p in all_persons if len(p["name"]) > 2], key=lambda x: len(x["name"]), reverse=True)
    for p in sorted_persons:
        p_name = p["name"].lower()
        if p_name in positive_query:
            matched_person = p
            # Strip person name to form scene prompt
            remaining_scene_prompt = positive_query.replace(p_name, "").strip()
            # Clean up connector words like "near a", "near", "with a", "with", "in a", "in", "at a", "at", "by a", "by"
            for connector in ["near a", "near", "with a", "with", "in a", "in", "at a", "at", "by a", "by"]:
                if remaining_scene_prompt.startswith(connector):
                    remaining_scene_prompt = remaining_scene_prompt[len(connector):].strip()
            break

    filter_image_ids = None
    if matched_person:
        filter_image_ids = db.get_image_ids_with_person(matched_person["id"])
        if not filter_image_ids:
            return {
                "query": q,
                "matched_person": matched_person["name"],
                "scene_prompt": remaining_scene_prompt,
                "negative_prompt": combined_negative if combined_negative else None,
                "total_matches": 0,
                "results": []
            }

    # 2. Encode query text with SigLIP Text ONNX
    siglip = SiglipEngine.get_instance()
    text_to_encode = remaining_scene_prompt if remaining_scene_prompt else positive_query
    q_emb = siglip.encode_text(text_to_encode)

    # 3. Apply Negative Vector Projection if negative prompt is present
    if combined_negative:
        neg_emb = siglip.encode_text(combined_negative)
        # Vector subtraction in semantic latent space
        q_emb = q_emb - 0.45 * neg_emb
        import numpy as np
        q_emb = q_emb / np.maximum(np.linalg.norm(q_emb), 1e-12)

    # 4. Vector Search with all filters
    matches = db.search_images_semantic(
        q_emb,
        limit=limit,
        min_score=min_score,
        filter_image_ids=filter_image_ids,
        library_id=library_id,
        from_date=from_date,
        to_date=to_date,
        min_size_kb=min_size_kb
    )

    return {
        "query": q,
        "matched_person": matched_person["name"] if matched_person else None,
        "scene_prompt": remaining_scene_prompt,
        "negative_prompt": combined_negative if combined_negative else None,
        "total_matches": len(matches),
        "results": matches
    }
'''

new_content = content[:start_idx] + new_search_code + "\n" + content[end_idx:]
api_path.write_text(new_content, encoding="utf-8")
print("Updated src/api.py successfully!")
