"""Snapshot versionado de la caché de datos (hashes SHA-256).

Genera ``data_cache_hashes.json`` con el hash de cada archivo de ``data/cache``
y ``data/ff1`` para poder verificar que una corrida usa exactamente los mismos
datos. Útil para reproducibilidad y para el CI.

Uso:
  .venv/bin/python scripts/snapshot.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRS = [ROOT / "data" / "cache", ROOT / "data" / "ff1"]
OUT = ROOT / "data_cache_hashes.json"


def main() -> None:
    hashes: dict[str, str] = {}
    for directory in DIRS:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    OUT.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(f"{len(hashes)} archivos -> {OUT}")


if __name__ == "__main__":
    main()
