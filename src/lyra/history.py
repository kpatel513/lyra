from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_BACKUP_EXTENSIONS = {
    ".py",
    ".pyw",
    ".sh",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
}


def _iter_files(repo: Path) -> Iterable[Path]:
    repo = repo.resolve()
    for p in repo.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(repo)
        # Skip known junk/large dirs
        if rel.parts and rel.parts[0] in {".git", ".venv", "venv", "build", "dist", ".lyra", "__pycache__"}:
            continue
        yield p


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _looks_binary(path: Path) -> bool:
    try:
        data = path.read_bytes()[:4096]
    except OSError:
        return True
    return b"\x00" in data


@dataclass(frozen=True)
class ManifestEntry:
    rel_path: str
    size: int
    sha256: str

    def to_dict(self) -> dict:
        return {"rel_path": self.rel_path, "size": self.size, "sha256": self.sha256}


def build_manifest(repo: Path) -> dict[str, ManifestEntry]:
    repo = repo.resolve()
    out: dict[str, ManifestEntry] = {}
    for p in _iter_files(repo):
        try:
            rel = str(p.relative_to(repo))
            st = p.stat()
            out[rel] = ManifestEntry(rel_path=rel, size=st.st_size, sha256=_sha256(p))
        except OSError:
            continue
    return out


@dataclass(frozen=True)
class HistoryEntry:
    repo: Path
    run_id: str
    root: Path
    before_manifest_path: Path
    after_manifest_path: Path
    changes_path: Path
    backup_root: Path
    meta_path: Path

    def to_dict(self) -> dict:
        return {
            "repo": str(self.repo),
            "run_id": self.run_id,
            "root": str(self.root),
        }


def history_root(repo: Path) -> Path:
    return repo.resolve() / ".lyra" / "history"


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_history_entry(
    *,
    repo: Path,
    command: str,
    backup_extensions: Optional[set[str]] = None,
    max_backup_bytes: int = 5 * 1024 * 1024,
) -> HistoryEntry:
    """
    Create a history entry with a BEFORE manifest and a backup copy of selected files.

    We intentionally back up a conservative set of likely-to-be-edited text/code files
    to keep this lightweight while still being useful.
    """
    repo = repo.resolve()
    run_id = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    root = history_root(repo) / run_id
    backup_root = root / "before"
    meta_path = root / "meta.json"
    before_manifest_path = root / "before_manifest.json"
    after_manifest_path = root / "after_manifest.json"
    changes_path = root / "changes.json"

    exts = backup_extensions or DEFAULT_BACKUP_EXTENSIONS

    before_manifest = build_manifest(repo)
    _write_json(before_manifest_path, {k: v.to_dict() for k, v in before_manifest.items()})

    backed_up: list[str] = []
    skipped: list[str] = []
    for rel, entry in before_manifest.items():
        p = repo / rel
        if p.suffix.lower() not in exts:
            continue
        if entry.size > max_backup_bytes:
            skipped.append(rel)
            continue
        if _looks_binary(p):
            skipped.append(rel)
            continue
        dest = backup_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(p.read_bytes())
            backed_up.append(rel)
        except OSError:
            skipped.append(rel)

    _write_json(
        meta_path,
        {
            "repo": str(repo),
            "run_id": run_id,
            "created_at_utc": run_id,
            "command": command,
            "backed_up_files": backed_up,
            "skipped_files": skipped,
            "note": "Undo can only restore files that were backed up.",
        },
    )

    return HistoryEntry(
        repo=repo,
        run_id=run_id,
        root=root,
        before_manifest_path=before_manifest_path,
        after_manifest_path=after_manifest_path,
        changes_path=changes_path,
        backup_root=backup_root,
        meta_path=meta_path,
    )


def finalize_history_entry(entry: HistoryEntry) -> dict:
    """
    Record AFTER manifest and compute changes vs BEFORE.
    """
    before = json.loads(entry.before_manifest_path.read_text(encoding="utf-8"))
    before_keys = set(before.keys())

    after_manifest = build_manifest(entry.repo)
    _write_json(entry.after_manifest_path, {k: v.to_dict() for k, v in after_manifest.items()})

    after_keys = set(after_manifest.keys())
    added = sorted(after_keys - before_keys)
    deleted = sorted(before_keys - after_keys)

    modified: list[str] = []
    for k in sorted(before_keys & after_keys):
        if before[k]["sha256"] != after_manifest[k].sha256:
            modified.append(k)

    changes = {
        "run_id": entry.run_id,
        "added": added,
        "deleted": deleted,
        "modified": modified,
    }
    _write_json(entry.changes_path, changes)
    return changes


def list_history(repo: Path) -> list[dict]:
    root = history_root(repo)
    if not root.exists():
        return []
    items: list[dict] = []
    for p in sorted(root.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        meta = p / "meta.json"
        if not meta.exists():
            continue
        try:
            items.append(json.loads(meta.read_text(encoding="utf-8")))
        except Exception:
            continue
    return items


def _load_manifest(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def undo_history(
    *,
    repo: Path,
    run_id: str,
    force: bool = False,
) -> None:
    """
    Restore backed-up files for a given run_id and remove added files.

    Safety:
    - By default, refuses to overwrite files that have diverged since the run (requires --force).
    """
    repo = repo.resolve()
    root = history_root(repo) / run_id
    before_manifest_path = root / "before_manifest.json"
    after_manifest_path = root / "after_manifest.json"
    changes_path = root / "changes.json"
    backup_root = root / "before"

    if not changes_path.exists() or not after_manifest_path.exists() or not before_manifest_path.exists():
        raise FileNotFoundError(f"History entry incomplete or missing: {root}")

    changes = json.loads(changes_path.read_text(encoding="utf-8"))
    after = _load_manifest(after_manifest_path)

    # Check for divergence since "after" state.
    divergent: list[str] = []
    for rel in changes["modified"] + changes["deleted"]:
        p = repo / rel
        if not p.exists():
            continue
        try:
            current = _sha256(p)
        except OSError:
            continue
        expected_after = after.get(rel, {}).get("sha256")
        if expected_after and current != expected_after:
            divergent.append(rel)

    if divergent and not force:
        raise RuntimeError(
            "Refusing to undo because these files changed since the optimization run:\n"
            + "\n".join(f"  - {p}" for p in divergent)
            + "\nRe-run with --force to overwrite."
        )

    # Remove added files
    for rel in changes["added"]:
        p = repo / rel
        if p.exists():
            try:
                p.unlink()
            except OSError:
                pass

    # Restore modified/deleted files (only if backed up)
    for rel in changes["modified"] + changes["deleted"]:
        src = backup_root / rel
        dst = repo / rel
        if not src.exists():
            # Can't restore what we didn't back up.
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())


