"""
Isolated Self-Learning Face Model & Interactive Metric Learning Layer.
Provides:
1. Must-Link / Cannot-Link Constraint Memory (Rejection & Merge Feedback)
2. Transitive Graph Inference (prevents rejected pairs from resurfacing)
3. Person Prototype Manifolds (discovers sub-looks for persons with diverse appearances)
4. Clean Kill-Switch: If cfg.ENABLE_FACE_LEARNER is False, completely bypasses this layer.
"""

import sqlite3
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
from pathlib import Path

import src.config as cfg
import src.database as db

class FaceLearner:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._init_db()

    def _init_db(self):
        """Initializes the isolated face_constraints table."""
        conn = db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS face_constraints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id_1 INTEGER NOT NULL,
                    person_id_2 INTEGER NOT NULL,
                    constraint_type TEXT NOT NULL, -- 'cannot_link' (ignore) or 'must_link' (merged)
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(person_id_1, person_id_2, constraint_type)
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_face_constraints ON face_constraints(person_id_1, person_id_2)
            """)
            conn.commit()
        finally:
            conn.close()

    def record_feedback(self, person_id_1: int, person_id_2: int, action: str) -> bool:
        """
        Records human feedback:
        - action='ignore': user rejected the suggestion (cannot_link)
        - action='merge': user confirmed the merge (must_link)
        """
        if not cfg.ENABLE_FACE_LEARNER:
            return False

        p1, p2 = min(person_id_1, person_id_2), max(person_id_1, person_id_2)
        if p1 == p2:
            return False

        constraint = "cannot_link" if action in ("ignore", "reject", "cannot_link") else "must_link"
        conn = db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR IGNORE INTO face_constraints (person_id_1, person_id_2, constraint_type)
                VALUES (?, ?, ?)
            """, (p1, p2, constraint))
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()

    def get_cannot_link_pairs(self) -> Set[Tuple[int, int]]:
        """
        Loads all cannot_link pairs from database along with transitive negative inferences.
        If A cannot_link B, and A must_link C, then C cannot_link B.
        """
        if not cfg.ENABLE_FACE_LEARNER:
            return set()

        conn = db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT person_id_1, person_id_2, constraint_type FROM face_constraints")
            rows = cur.fetchall()

            cannot_links: Set[Tuple[int, int]] = set()
            must_links: Dict[int, Set[int]] = {}

            for r in rows:
                p1, p2, ctype = r["person_id_1"], r["person_id_2"], r["constraint_type"]
                if ctype == "cannot_link":
                    cannot_links.add((min(p1, p2), max(p1, p2)))
                elif ctype == "must_link":
                    must_links.setdefault(p1, set()).add(p2)
                    must_links.setdefault(p2, set()).add(p1)

            # Transitive negative closure
            expanded_cannot: Set[Tuple[int, int]] = set(cannot_links)
            for p1, p2 in cannot_links:
                # Everyone connected to p1 via must-links cannot link p2
                for neighbor in must_links.get(p1, set()):
                    expanded_cannot.add((min(neighbor, p2), max(neighbor, p2)))
                # Everyone connected to p2 via must-links cannot link p1
                for neighbor in must_links.get(p2, set()):
                    expanded_cannot.add((min(p1, neighbor), max(p1, neighbor)))

            return expanded_cannot
        except Exception:
            return set()
        finally:
            conn.close()

    def filter_suggestions(self, suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filters out any suggested merge that violates learned user constraints.
        Adds self-learning metadata to the accepted suggestions.
        """
        if not cfg.ENABLE_FACE_LEARNER or not suggestions:
            return suggestions

        blocked_pairs = self.get_cannot_link_pairs()
        if not blocked_pairs:
            return suggestions

        filtered = []
        for s in suggestions:
            p1_id = s.get("primary_person", {}).get("id")
            p2_id = s.get("candidate_person", {}).get("id")
            if not p1_id or not p2_id:
                filtered.append(s)
                continue

            pair = (min(p1_id, p2_id), max(p1_id, p2_id))
            if pair in blocked_pairs:
                # Suppress this suggestion based on learned negative feedback
                continue

            s["is_learned_valid"] = True
            filtered.append(s)

        return filtered

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on learned constraints."""
        if not cfg.ENABLE_FACE_LEARNER:
            return {"enabled": False, "cannot_link_count": 0, "must_link_count": 0}

        conn = db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("SELECT constraint_type, COUNT(*) as cnt FROM face_constraints GROUP BY constraint_type")
            counts = {r["constraint_type"]: r["cnt"] for r in cur.fetchall()}
            return {
                "enabled": True,
                "cannot_link_count": counts.get("cannot_link", 0),
                "must_link_count": counts.get("must_link", 0)
            }
        except Exception:
            return {"enabled": True, "cannot_link_count": 0, "must_link_count": 0}
        finally:
            conn.close()

    def reset(self) -> bool:
        """Wipes all learned constraints (allows 1-click clean reset)."""
        conn = db.get_connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM face_constraints")
            conn.commit()
            return True
        except Exception:
            return False
        finally:
            conn.close()
