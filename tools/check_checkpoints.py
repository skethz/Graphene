"""Verify reference checkpoint files against the release's SHA-256 manifest."""

import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "checkpoints"
    manifest = json.loads((root / "manifest.json").read_text())
    for item in manifest["checkpoints"]:
        path = root / item["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; download the reference-checkpoints release assets first.")
        digest = hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Checkpoint hash/size mismatch: {path}")
        print(f"OK {item['file']}")


if __name__ == "__main__":
    main()
