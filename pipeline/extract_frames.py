"""Sample the walk-through video into JPEG frames the browser can match 3D frames against.

Usage: python pipeline/extract_frames.py data/boiler_room_360.mp4 web/frames [fps]
Writes NNNN.jpg + index.json ({"fps": .., "source": <video hash>, "files": [..]}). Needs ffmpeg on PATH.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


def main(video, outdir, fps="1"):
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found on PATH")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.jpg"):
        old.unlink()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-vf", f"fps={fps},scale=960:-2",
                    "-q:v", "3", str(out / "%04d.jpg")], check=True)
    files = sorted(p.name for p in out.glob("*.jpg"))
    assert files, "ffmpeg produced no frames"
    # the browser caches tile embeddings under this hash, so frames from another video never reuse them
    h = hashlib.sha1(f"{fps}".encode())
    with open(video, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    (out / "index.json").write_text(json.dumps({"fps": float(fps), "source": h.hexdigest()[:12], "files": files}))
    print(f"[frames] {len(files)} frames at {fps} fps -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
