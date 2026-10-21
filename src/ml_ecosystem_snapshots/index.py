"""Ephemeral local SQLite search index for fast symbol lookups.

Provides on-demand SQLite FTS5 indexing over bundled and locally cached
snapshot JSON files, enabling sub-millisecond lookups and fuzzy searches
without loading multi-megabyte JSON trees into memory.
"""

from __future__ import annotations

import contextlib
import gzip
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import typing

import pydantic

from .exceptions import SnapshotSchemaError
from .models import JsonValue
from .network_env import get_custom_snapshots_paths


class IndexedSymbolRecord(pydantic.BaseModel):
    """Typed record for an indexed symbol returned by `search_index` or `lookup_symbol`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    api_path: str = ""
    framework: str = ""
    kind: str = "function"
    category: str = ""
    data: dict[str, JsonValue] = pydantic.Field(default_factory=dict)

    @classmethod
    def from_raw_dict(
        cls, raw: typing.Mapping[str, JsonValue], framework: str = ""
    ) -> IndexedSymbolRecord:
        """Construct an `IndexedSymbolRecord` from a raw JSON symbol dictionary.

        Args:
            raw: Deserialized JSON symbol dictionary.
            framework: Optional framework identifier.

        Returns:
            Populated `IndexedSymbolRecord` instance.
        """
        api_path = str(
            raw.get("api_path") or raw.get("name") or raw.get("mnemonic") or ""
        )
        kind = str(raw.get("kind") or "function")
        category = str(raw.get("category") or "")
        return cls(
            api_path=api_path,
            framework=framework,
            kind=kind,
            category=category,
            data=dict(raw),
        )

    def __getitem__(self, key: str) -> JsonValue:
        """Retrieve a symbol data field by dictionary key.

        Args:
            key: Field name inside `self.data`.

        Returns:
            Field value.
        """
        return self.data[key]

    def get(self, key: str, default: JsonValue = None) -> JsonValue:
        """Retrieve a symbol data field by key with an optional default.

        Args:
            key: Field name inside `self.data`.
            default: Default value when absent.

        Returns:
            Field value or `default`.
        """
        return self.data.get(key, default)


def get_cache_dir() -> str:
    """Retrieve the platform-specific cache directory for ml-ecosystem-snapshots.

    Returns:
        Absolute path to the user's local cache directory.
    """
    base = os.environ.get("ML_ECOSYSTEM_SNAPSHOTS_CACHE_DIR")
    if not base:
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg:
            base = os.path.join(xdg, "ml_ecosystem_snapshots")
        else:
            base = os.path.join(
                os.path.expanduser("~"), ".cache", "ml_ecosystem_snapshots"
            )
    try:
        os.makedirs(base, exist_ok=True)
    except FileNotFoundError:
        base = os.path.join(tempfile.gettempdir(), "ml_ecosystem_cache")
        try:
            os.makedirs(base, exist_ok=True)
        except (FileExistsError, OSError):
            pass
    except FileExistsError:
        pass
    return base


def get_index_db_path() -> str:
    """Retrieve the path to the ephemeral SQLite index database.

    Returns:
        Absolute path to the index database file.
    """
    return os.path.join(get_cache_dir(), "index.db")


def compute_file_sha256(file_path: str) -> str:
    """Compute the SHA256 checksum of a file.

    Args:
        file_path: Path to the target file.

    Returns:
        Hexadecimal SHA256 checksum string.
    """
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Initialize the SQLite index database connection and schema.

    Args:
        db_path: Optional path to the database file (defaults to get_index_db_path()).

    Returns:
        Open sqlite3 connection with schema initialized.
    """
    path = db_path or get_index_db_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        conn = sqlite3.connect(path)
    except (OSError, sqlite3.OperationalError):
        conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        conn.execute("PRAGMA synchronous=NORMAL")
    except sqlite3.OperationalError:
        pass

    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS indexed_files (
                file_path TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                framework TEXT NOT NULL,
                version TEXT NOT NULL,
                symbol_count INTEGER NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                framework TEXT NOT NULL,
                version TEXT NOT NULL,
                api_path TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                docstring TEXT,
                json_data TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_symbols_lookup
            ON symbols(framework, api_path)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_symbols_name
            ON symbols(framework, name)
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
                api_path,
                name,
                docstring,
                content='symbols',
                content_rowid='id'
            )
            """
        )

    return conn


def get_readonly_connection(db_path: str | None = None) -> sqlite3.Connection:
    """Open a read-only SQLite connection to the index database with concurrency timeout.

    Args:
        db_path: Optional path to SQLite database.

    Returns:
        Configured read-only sqlite3.Connection.
    """
    path = db_path or get_index_db_path()
    if not os.path.exists(path):
        return init_db(path)
    uri = f"file:{os.path.abspath(path)}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout=10000")
        return conn
    except sqlite3.OperationalError:
        return init_db(path)


def clean_index_cache(db_path: str | None = None) -> int:
    """Remove local ephemeral index database and auxiliary journal/WAL files.

    Args:
        db_path: Optional path to SQLite database file.

    Returns:
        Number of removed cache files.
    """
    path = db_path or get_index_db_path()
    count = 0
    for suffix in ("", "-wal", "-shm", "-journal"):
        p = f"{path}{suffix}"
        if os.path.exists(p):
            try:
                os.remove(p)
                count += 1
            except OSError:
                pass
    return count


def extract_framework_and_version(file_name: str) -> tuple[str, str]:
    """Extract framework identifier and version string from snapshot filename.

    Args:
        file_name: Filename of the snapshot (e.g. 'torch_v2.2.0.json').

    Returns:
        Tuple of (framework, version).
    """
    base = file_name
    if base.endswith(".json.gz"):
        base = base[:-8]
    elif base.endswith(".json"):
        base = base[:-5]

    if "_v" in base:
        parts = base.split("_v", 1)
        return parts[0], parts[1]
    if "_exhaustive" in base:
        fw = base.replace("_exhaustive", "")
        return fw, "latest"

    multiword_frameworks = [
        "nvidia_sass",
        "nvidia_ptx",
        "amd_rdna",
        "html_dsl",
        "latex_dsl",
        "flax_nnx",
        "optax",
        "orbax_checkpoint",
    ]
    for mw in multiword_frameworks:
        if base.startswith(mw + "_"):
            return mw, base[len(mw) + 1 :]

    if "_" in base:
        parts = base.split("_", 1)
        return parts[0], parts[1]

    return base, "latest"


def index_snapshot_file(json_path: str, conn: sqlite3.Connection) -> int:
    """Index a single snapshot JSON file into the database.

    Args:
        json_path: Path to the snapshot JSON file.
        conn: Open SQLite connection.

    Returns:
        Number of symbols indexed.

    Raises:
        SnapshotSchemaError: If the snapshot JSON file cannot be parsed or is not a valid object or array.
    """
    if not os.path.isfile(json_path):
        return 0

    file_hash = compute_file_sha256(json_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT file_hash FROM indexed_files WHERE file_path = ?",
        (json_path,),
    )
    row = cur.fetchone()
    if row and row["file_hash"] == file_hash:
        return 0

    try:
        if json_path.endswith(".gz"):
            with gzip.open(json_path, "rt", encoding="utf-8") as gf:
                data: JsonValue = json.load(gf)
        else:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
    except (json.JSONDecodeError, OSError, EOFError) as exc:
        raise SnapshotSchemaError(
            f"Invalid snapshot JSON payload in '{json_path}': {exc}"
        ) from exc

    if not isinstance(data, (dict, list)):
        raise SnapshotSchemaError(
            f"Snapshot payload in '{json_path}' must be a JSON object or array, got {type(data).__name__}."
        )

    file_name = os.path.basename(json_path)
    framework, version = extract_framework_and_version(file_name)
    if isinstance(data, dict):
        if data.get("target"):
            framework = str(data["target"])
        if data.get("version"):
            version = str(data["version"])

    items: list[dict[str, JsonValue]] = []
    if isinstance(data, dict):
        raw_cats = data.get("categories")
        raw_ops = data.get("operations")
        raw_insts = data.get("instructions")
        raw_items = data.get("items")
        if isinstance(raw_cats, dict):
            for _cat, cat_items in raw_cats.items():
                if isinstance(cat_items, list):
                    items.extend([dict(x) for x in cat_items if isinstance(x, dict)])
        elif isinstance(raw_ops, list):
            items.extend([dict(x) for x in raw_ops if isinstance(x, dict)])
        elif isinstance(raw_insts, list):
            items.extend([dict(x) for x in raw_insts if isinstance(x, dict)])
        elif isinstance(raw_items, list):
            items.extend([dict(x) for x in raw_items if isinstance(x, dict)])
    else:
        for raw_elem in data:
            assert isinstance(raw_elem, dict)
            items.append(dict(raw_elem))

    with conn:
        cur.execute(
            "SELECT id, api_path, name, docstring FROM symbols WHERE file_path = ?",
            (json_path,),
        )
        old_rows = cur.fetchall()
        if old_rows:
            for r in old_rows:
                cur.execute(
                    "INSERT INTO symbols_fts(symbols_fts, rowid, api_path, name, docstring) VALUES('delete', ?, ?, ?, ?)",
                    (r["id"], r["api_path"], r["name"], r["docstring"]),
                )
            cur.execute("DELETE FROM symbols WHERE file_path = ?", (json_path,))

        inserted = 0
        for item in items:
            api_path = str(
                item.get("api_path") or item.get("name") or item.get("mnemonic") or ""
            )
            name = str(
                item.get("name") or item.get("mnemonic") or api_path.split(".")[-1]
            )
            kind = str(item.get("kind", "function"))
            doc = str(item.get("docstring") or "")
            raw_json = json.dumps(item)

            cur.execute(
                """
                INSERT INTO symbols (file_path, framework, version, api_path, name, kind, docstring, json_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (json_path, framework, version, api_path, name, kind, doc, raw_json),
            )
            sym_id = cur.lastrowid
            cur.execute(
                """
                INSERT INTO symbols_fts (rowid, api_path, name, docstring)
                VALUES (?, ?, ?, ?)
                """,
                (sym_id, api_path, name, doc),
            )
            inserted += 1

        cur.execute(
            """
            INSERT OR REPLACE INTO indexed_files (file_path, file_hash, framework, version, symbol_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (json_path, file_hash, framework, version, inserted),
        )

    return inserted


def get_available_snapshot_files() -> list[str]:
    """Discover all bundled and locally cached snapshot JSON files.

    Returns:
        List of absolute file paths to discovered JSON files.
    """
    found: list[str] = []
    base_dir = os.path.dirname(__file__)
    search_dirs = get_custom_snapshots_paths() + [
        os.path.join(base_dir, "snapshots"),
        os.path.join(base_dir, "frameworks"),
        os.path.join(get_cache_dir(), "snapshots"),
    ]
    for sdir in search_dirs:
        if os.path.isdir(sdir):
            for fname in sorted(os.listdir(sdir)):
                if fname.endswith(".json") or fname.endswith(".json.gz"):
                    found.append(os.path.join(sdir, fname))
    return found


def ensure_index(db_path: str | None = None) -> sqlite3.Connection:
    """Ensure the local SQLite index is initialized and up to date.

    Args:
        db_path: Optional path to the database file.

    Returns:
        Active sqlite3 connection ready for querying.
    """
    conn = init_db(db_path)
    snapshot_files = get_available_snapshot_files()
    for sfile in snapshot_files:
        try:
            index_snapshot_file(sfile, conn)
        except Exception:
            pass
    return conn


def search_index(
    query: str,
    framework: str | None = None,
    version: str | None = None,
    limit: int = 10,
    db_path: str | None = None,
) -> list[IndexedSymbolRecord]:
    """Search for operations using the SQLite FTS5 index.

    Args:
        query: Full-text search term or mnemonic.
        framework: Optional framework filter.
        version: Optional version filter.
        limit: Maximum results to return.
        db_path: Optional custom database path.

    Returns:
        List of `IndexedSymbolRecord` instances.
    """
    clean_q = re.sub(r"[^\w\.]+", " ", query).strip()
    if not clean_q:
        return []

    tokens = [
        t for t in clean_q.split() if t.upper() not in ("AND", "OR", "NOT", "NEAR")
    ]
    if not tokens:
        return []

    with contextlib.closing(ensure_index(db_path)) as conn:
        cur = conn.cursor()

        fts_term = " ".join(f'"{t}"' for t in tokens[:-1] + [tokens[-1]]) + "*"

        conditions = ["symbols_fts MATCH ?"]
        params: list[str | int] = [fts_term]

        if framework:
            conditions.append("s.framework = ?")
            params.append(framework.lower().strip())
        if version:
            conditions.append("s.version = ?")
            params.append(version.strip())

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT s.framework, s.json_data
            FROM symbols_fts f
            JOIN symbols s ON f.rowid = s.id
            WHERE {where_clause}
            ORDER BY rank
            LIMIT ?
        """
        params.append(limit)

        cur.execute(sql, params)
        rows = cur.fetchall()
        results: list[IndexedSymbolRecord] = []
        for r in rows:
            try:
                parsed: JsonValue = json.loads(r["json_data"])
                assert isinstance(parsed, dict)
                results.append(
                    IndexedSymbolRecord.from_raw_dict(
                        parsed, framework=str(r["framework"] or "")
                    )
                )
            except Exception:
                pass
        return results


def lookup_symbol(
    framework: str,
    api_path: str,
    version: str | None = None,
    db_path: str | None = None,
) -> IndexedSymbolRecord | None:
    """Look up exact symbol definition by framework and API path.

    Args:
        framework: Target framework name.
        api_path: Canonical API path or mnemonic.
        version: Optional framework version.
        db_path: Optional database path.

    Returns:
        `IndexedSymbolRecord` instance, or None if not found.
    """
    with contextlib.closing(ensure_index(db_path)) as conn:
        cur = conn.cursor()

        clean_fw = framework.lower().strip()
        conditions = ["framework = ?", "(api_path = ? OR name = ?)"]
        params: list[str | int] = [clean_fw, api_path, api_path]

        if version:
            conditions.append("version = ?")
            params.append(version.strip())

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT framework, json_data
            FROM symbols
            WHERE {where_clause}
            ORDER BY id DESC
            LIMIT 1
        """
        cur.execute(sql, params)
        row = cur.fetchone()
        if row:
            parsed_sym: JsonValue = json.loads(row["json_data"])
            assert isinstance(parsed_sym, dict)
            return IndexedSymbolRecord.from_raw_dict(
                parsed_sym, framework=str(row["framework"] or clean_fw)
            )
        return None


def clear_index(db_path: str | None = None) -> bool:
    """Clear and delete the local ephemeral index database file.

    Args:
        db_path: Optional database path.

    Returns:
        True if cleared or absent, False if removal failed.
    """
    path = db_path or get_index_db_path()
    if os.path.exists(path):
        try:
            os.remove(path)
            return True
        except OSError:
            return False
    return True
