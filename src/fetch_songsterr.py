"""Fetch Songsterr metadata and extract track JSONs from HAR files.

Songsterr track data is served from a CDN with a per-revision access hash.
That hash is only available while browsing a tab in the browser. Therefore:

1. Use this script to discover song IDs / revision IDs via the public API.
2. Open the song in a browser, let all tracks load, then save the network log as HAR.
3. Run extract_har.py on the saved HAR to dump the actual track JSON files.

The script also saves the HAR if you pass --save-har, which you can then feed
into extract_har.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from urllib.parse import quote_plus

import urllib.request

API_BASE = "https://www.songsterr.com/api"
SEARCH_URL = f"{API_BASE}/search"
REVISIONS_URL = f"{API_BASE}/meta/{{song_id}}/revisions"


def _get_json(url: str, retries: int = 3) -> dict | list:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            print(f"  attempt {attempt + 1}/{retries} failed: {e}")
            time.sleep(1)
    raise RuntimeError(f"Failed to fetch {url}")


def search_songs(pattern: str, size: int = 50) -> list[dict]:
    url = f"{SEARCH_URL}?pattern={quote_plus(pattern)}&size={size}&from=0&more=true"
    return _get_json(url)


def get_revisions(song_id: int, translate: str = "en") -> list[dict]:
    url = REVISIONS_URL.format(song_id=song_id) + f"?translateTo={translate}"
    return _get_json(url)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", help="Search pattern (e.g. 'nirvana teen spirit')")
    parser.add_argument("--song-id", type=int, help="Songsterr song ID")
    parser.add_argument("--out", default="data/processed", help="Directory for metadata JSON")
    parser.add_argument("--size", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.search:
        results = search_songs(args.search, args.size)
        print(f"Found {len(results)} songs for '{args.search}'")
        for r in results:
            print(f"  {r.get('songId')}: {r.get('artist')} - {r.get('title')}")
        with (out_dir / f"search_{args.search.replace(' ', '_')}.json").open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    if args.song_id:
        revisions = get_revisions(args.song_id)
        print(f"\nSong {args.song_id}: {len(revisions)} revisions")
        for rev in revisions:
            print(
                f"  revisionId={rev.get('revisionId')} tracks={rev.get('tracksCount')} "
                f"blocked={rev.get('isBlocked')} by {rev.get('person')}"
            )
        with (out_dir / f"revisions_{args.song_id}.json").open("w", encoding="utf-8") as f:
            json.dump(revisions, f, indent=2)

        print(
            "\nTo obtain the actual track JSONs: open the tab in a browser, "
            "save the network log as HAR, then run:\n"
            f"  python src/extract_har.py <har_file> --out data/raw"
        )


if __name__ == "__main__":
    main()
