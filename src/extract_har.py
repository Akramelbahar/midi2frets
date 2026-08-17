"""Extract Songsterr track JSONs from a browser HAR archive.

Usage:
    python src/extract_har.py path/to/www.songsterr.com.har --out data/raw

The script scans HTTP responses for URLs ending in /<N>.json served from
Songsterr's CDN and saves each track as <artist> - <title> - <track_name>.json.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse


def sanitize(name: str) -> str:
    return re.sub(r"[^\w\-_.]", "_", name).strip("_") or "track"


def extract_tracks(har_path: str, out_dir: str):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(har_path, "r", encoding="utf-8") as f:
        har = json.load(f)

    entries = har.get("log", {}).get("entries", [])
    saved = 0
    seen = set()

    for entry in entries:
        url = entry.get("request", {}).get("url", "")
        # Songsterr CDN track JSONs end with /<index>.json
        if not re.search(r"/\d+\.json$", url):
            continue
        if url in seen:
            continue
        seen.add(url)

        content = entry.get("response", {}).get("content", {})
        text = content.get("text")
        if not text:
            continue

        try:
            track = json.loads(text)
        except json.JSONDecodeError:
            continue

        if "measures" not in track or "tuning" not in track:
            continue

        # Try to get artist/title from HAR page title or URL
        pages = har.get("log", {}).get("pages", [{}])
        page_title = pages[0].get("title", "")
        m = re.search(r"a/wsa/([^/]+)-tab-s\d+", url) or re.search(r"a/wsa/([^/]+)-tab-s\d+", page_title)
        slug = m.group(1).replace("-", " ") if m else "unknown"

        track_name = track.get("name", f"track_{saved}")
        filename = f"{sanitize(slug)} - {sanitize(track_name)}.json"
        dest = out_path / filename
        counter = 1
        while dest.exists():
            dest = out_path / f"{sanitize(slug)} - {sanitize(track_name)}_{counter}.json"
            counter += 1

        with dest.open("w", encoding="utf-8") as f:
            json.dump(track, f, indent=2)
        saved += 1
        print(f"Saved {saved}: {dest} ({track.get('instrument', 'unknown')})")

    print(f"\nExtracted {saved} track JSONs to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("har", help="Path to HAR file")
    parser.add_argument("--out", default="data/raw", help="Output directory")
    args = parser.parse_args()
    extract_tracks(args.har, args.out)
