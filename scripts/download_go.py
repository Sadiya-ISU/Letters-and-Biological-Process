"""Download a pinned go-basic.obo and record its provenance.

Default URL is the current release; to reproduce a *specific* historical release
(see Task 2), pass --url http://release.geneontology.org/<YYYY-MM-DD>/ontology/go-basic.obo
"""
import argparse, hashlib, shutil, sys, urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
DEFAULT_URL = "http://current.geneontology.org/ontology/go-basic.obo"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--out", default=str(DATA / "go-basic.obo"))
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.url} ...", file=sys.stderr)
    req = urllib.request.Request(args.url, headers={"User-Agent": "Mozilla/5.0 (compatible; go_lingdepth/0.1)"})
    with urllib.request.urlopen(req) as resp, open(args.out, "wb") as out:
        shutil.copyfileobj(resp, out)
    raw = Path(args.out).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    data_version = "UNKNOWN"
    for line in raw[:4000].decode("utf-8", "replace").splitlines():
        if line.startswith("data-version:"):
            data_version = line.split("data-version:", 1)[1].strip()
            break
    (DATA / "DATA_VERSION.txt").write_text(
        f"url: {args.url}\ndata-version: {data_version}\nsha256: {sha}\nbytes: {len(raw)}\n")
    print(f"Saved {args.out}\n  data-version: {data_version}\n  sha256: {sha}\n  bytes: {len(raw)}")


if __name__ == "__main__":
    main()
