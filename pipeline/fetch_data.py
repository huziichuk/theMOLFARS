"""Download the intact splat (Equipment Room.ply, ~395 MB) from the team's Google Drive when it is not on disk yet.

    python pipeline/fetch_data.py [dest]          default dest: data/equipment_room.ply

build_scene.py and server.py call fetch_ply() themselves when the file is missing, so a fresh clone only needs
the Excel pack in data/. Set SPLAT_URL to use another public link (any URL that serves the raw .ply bytes).
"""
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLY = ROOT / "data" / "equipment_room.ply"
DRIVE_ID = "1MrwNjHxFqk5IfHpJQ24KlTn5nAAn40rd"
# usercontent + confirm=t skips Drive's "can't scan this large file for viruses" page for public files
URL = os.environ.get("SPLAT_URL", f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download&confirm=t")


def fetch_ply(dest: Path = DEFAULT_PLY, url: str = URL) -> Path:
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    print(f"[fetch] {dest.name} not found, downloading from {url.split('?')[0]} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "theMOLFARS-demo"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        head = r.read(3)
        if head != b"ply":  # Drive answers with an HTML page when the link is not public or the quota is used up
            raise SystemExit(f"[fetch] the link did not return a .ply file (got {head!r}); download it by hand into {dest}")
        f.write(head)
        total, done, step = int(r.headers.get("Content-Length") or 0), 3, 50 << 20
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if done // step != (done - len(chunk)) // step:
                print(f"[fetch] {done >> 20} MB" + (f" / {total >> 20} MB" if total else ""), flush=True)
    part.replace(dest)  # only a complete file gets the real name, so an interrupted download is retried next time
    print(f"[fetch] saved {dest} ({dest.stat().st_size >> 20} MB)")
    return dest


if __name__ == "__main__":
    fetch_ply(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PLY)
