from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
import time
from dataclasses import dataclass

_DB_LOCK = threading.Lock()


@dataclass(frozen=True)
class RewriteArtifact:
    artifact_key: str
    prompt_id: str
    model: str
    request_url: str
    source_text: str
    rewritten_text: str
    created_at: float
    updated_at: float
    hit_count: int


def _resolve_cache_db_path() -> str:
    raw = os.getenv("COMPRESSION_LLM_CACHE_DB_PATH", "").strip()
    if raw:
        return raw
    return "/tmp/compression_llm_rewrites.sqlite3"


def _connect() -> sqlite3.Connection:
    path = _resolve_cache_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_rewrite_artifacts (
            artifact_key TEXT PRIMARY KEY,
            prompt_id TEXT NOT NULL,
            model TEXT NOT NULL,
            request_url TEXT NOT NULL,
            source_text TEXT NOT NULL,
            rewritten_text TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.commit()
    return conn


def build_artifact_key(*, request_url: str, model: str, prompt_id: str, source_text: str) -> str:
    payload = "\n".join((request_url.strip(), model.strip(), prompt_id.strip(), source_text))
    return f"rw_{hashlib.sha256(payload.encode('utf-8')).hexdigest()}"


def _row_to_artifact(row: sqlite3.Row | None) -> RewriteArtifact | None:
    if row is None:
        return None
    return RewriteArtifact(**dict(row))


def get_rewrite_artifact(artifact_key: str, *, increment_hit: bool = False) -> RewriteArtifact | None:
    normalized = artifact_key.strip()
    if not normalized:
        return None
    with _DB_LOCK:
        conn = _connect()
        try:
            if increment_hit:
                conn.execute(
                    """
                    UPDATE llm_rewrite_artifacts
                    SET hit_count = hit_count + 1,
                        updated_at = ?
                    WHERE artifact_key = ?
                    """,
                    (time.time(), normalized),
                )
                conn.commit()
            row = conn.execute(
                """
                SELECT artifact_key, prompt_id, model, request_url, source_text, rewritten_text, created_at, updated_at, hit_count
                FROM llm_rewrite_artifacts
                WHERE artifact_key = ?
                """,
                (normalized,),
            ).fetchone()
        finally:
            conn.close()
    return _row_to_artifact(row)


def lookup_rewrite(*, request_url: str, model: str, prompt_id: str, source_text: str) -> RewriteArtifact | None:
    artifact_key = build_artifact_key(
        request_url=request_url,
        model=model,
        prompt_id=prompt_id,
        source_text=source_text,
    )
    return get_rewrite_artifact(artifact_key, increment_hit=True)


def store_rewrite(
    *,
    request_url: str,
    model: str,
    prompt_id: str,
    source_text: str,
    rewritten_text: str,
) -> RewriteArtifact:
    now = time.time()
    artifact_key = build_artifact_key(
        request_url=request_url,
        model=model,
        prompt_id=prompt_id,
        source_text=source_text,
    )
    with _DB_LOCK:
        conn = _connect()
        try:
            conn.execute(
                """
                INSERT INTO llm_rewrite_artifacts (
                    artifact_key, prompt_id, model, request_url, source_text, rewritten_text, created_at, updated_at, hit_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(artifact_key) DO UPDATE SET
                    rewritten_text = excluded.rewritten_text,
                    updated_at = excluded.updated_at
                """,
                (artifact_key, prompt_id, model, request_url, source_text, rewritten_text, now, now, 0),
            )
            conn.commit()
            row = conn.execute(
                """
                SELECT artifact_key, prompt_id, model, request_url, source_text, rewritten_text, created_at, updated_at, hit_count
                FROM llm_rewrite_artifacts
                WHERE artifact_key = ?
                """,
                (artifact_key,),
            ).fetchone()
        finally:
            conn.close()
    artifact = _row_to_artifact(row)
    if artifact is None:
        raise RuntimeError("failed to store rewrite artifact")
    return artifact


def list_recent_rewrites(*, limit: int = 50, prompt_id: str | None = None) -> list[RewriteArtifact]:
    safe_limit = max(1, min(limit, 500))
    with _DB_LOCK:
        conn = _connect()
        try:
            if prompt_id and prompt_id.strip():
                rows = conn.execute(
                    """
                    SELECT artifact_key, prompt_id, model, request_url, source_text, rewritten_text, created_at, updated_at, hit_count
                    FROM llm_rewrite_artifacts
                    WHERE prompt_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (prompt_id.strip(), safe_limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT artifact_key, prompt_id, model, request_url, source_text, rewritten_text, created_at, updated_at, hit_count
                    FROM llm_rewrite_artifacts
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (safe_limit,),
                ).fetchall()
        finally:
            conn.close()
    return [_row_to_artifact(row) for row in rows if _row_to_artifact(row) is not None]


def select_candidate_artifacts(
    *,
    source_texts: list[str],
    request_url: str,
    model: str,
    limit: int = 40,
) -> list[RewriteArtifact]:
    normalized_source_texts = [text for text in source_texts if isinstance(text, str) and text.strip()]
    recent_limit = max(1, min(limit, 200))
    exact_matches: list[RewriteArtifact] = []
    seen_keys: set[str] = set()
    for text in normalized_source_texts:
        with _DB_LOCK:
            conn = _connect()
            try:
                rows = conn.execute(
                    """
                    SELECT artifact_key, prompt_id, model, request_url, source_text, rewritten_text, created_at, updated_at, hit_count
                    FROM llm_rewrite_artifacts
                    WHERE request_url = ? AND model = ? AND source_text = ?
                    ORDER BY updated_at DESC
                    LIMIT 10
                    """,
                    (request_url, model, text),
                ).fetchall()
            finally:
                conn.close()
        for row in rows:
            artifact = _row_to_artifact(row)
            if artifact is None or artifact.artifact_key in seen_keys:
                continue
            seen_keys.add(artifact.artifact_key)
            exact_matches.append(artifact)
            if len(exact_matches) >= recent_limit:
                return exact_matches

    recent = list_recent_rewrites(limit=recent_limit)
    candidates = list(exact_matches)
    for artifact in recent:
        if artifact.artifact_key in seen_keys:
            continue
        if artifact.request_url == request_url or artifact.model == model:
            candidates.append(artifact)
            seen_keys.add(artifact.artifact_key)
        if len(candidates) >= recent_limit:
            break
    return candidates
