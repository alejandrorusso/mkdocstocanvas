import json
from pathlib import Path


def load_cache(cache: Path) -> dict:
    """Load cache, creating it if missing, and handling empty/corrupt files."""
    if not cache.exists():
        # Create parent directories if they don't exist, then create the file
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text("{}", encoding="utf-8")
        return {}

    with open(cache, "r", encoding="utf-8") as f:
        # .strip() handles files that are just whitespace
        content = f.read().strip()
        return json.loads(content) if content else {}


def save_cache(cache: Path, metadata: dict):
    """Save upload metadata to file (atomic write)."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp_cache = cache.with_name(f"{cache.name}.tmp")

    with open(tmp_cache, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # Atomic replace: either the old cache remains, or the new one fully lands.
    tmp_cache.replace(cache)
