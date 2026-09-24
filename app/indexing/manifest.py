"""SQLite 文档清单：文件 hash/版本/块数/时效，支撑增量索引。"""
import os
import sqlite3
from datetime import UTC, datetime

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id       TEXT PRIMARY KEY,
    path         TEXT NOT NULL,
    md5          TEXT NOT NULL,
    chunk_count  INTEGER NOT NULL,
    version      TEXT NOT NULL,
    valid_from   TEXT,
    valid_until  TEXT,
    indexed_at   TEXT NOT NULL
);
"""


class Manifest:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        # check_same_thread=False：TestClient 在工作线程跑 ASGI app；
        # 实际写入均在事件循环内串行（documents 写操作共用 index_lock）
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def upsert(self, doc_id: str, **fields) -> None:
        self._conn.execute(
            """
            INSERT INTO documents(doc_id, path, md5, chunk_count, version,
                                  valid_from, valid_until, indexed_at)
            VALUES (:doc_id, :path, :md5, :chunk_count, :version,
                    :valid_from, :valid_until, :indexed_at)
            ON CONFLICT(doc_id) DO UPDATE SET
                path=excluded.path, md5=excluded.md5, chunk_count=excluded.chunk_count,
                version=excluded.version, valid_from=excluded.valid_from,
                valid_until=excluded.valid_until, indexed_at=excluded.indexed_at
            """,
            {
                "doc_id": doc_id,
                "path": fields["path"],
                "md5": fields["md5"],
                "chunk_count": fields["chunk_count"],
                "version": fields["version"],
                "valid_from": fields.get("valid_from"),
                "valid_until": fields.get("valid_until"),
                "indexed_at": datetime.now(UTC).isoformat(),
            },
        )
        self._conn.commit()

    def get(self, doc_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return dict(row) if row else None

    def all(self) -> dict[str, dict]:
        rows = self._conn.execute("SELECT * FROM documents").fetchall()
        return {r["doc_id"]: dict(r) for r in rows}

    def remove(self, doc_id: str) -> None:
        self._conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
