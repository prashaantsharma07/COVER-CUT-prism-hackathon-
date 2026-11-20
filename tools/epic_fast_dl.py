"""Parallel EPIC-KITCHENS video downloader (stdlib only).

EPIC's own downloader uses ONE connection per file (~0.5 MB/s observed from Kaggle). This one splits a
file into byte ranges fetched by several connections, retries/resumes each range, and checks the size.
URLs follow epic_downloader.py: EPIC-55 videos -> <base55>/videos/<split>/<P>/<vid>.MP4,
extension videos -> <base100>/<P>/videos/<vid>.MP4.
usage (standalone test): python tools/epic_fast_dl.py <out_dir> <video_id> [--conns 8]"""
import argparse, csv, os, ssl, threading, time, urllib.request

ssl._create_default_https_context = ssl._create_unverified_context      # same as the official script
B55 = "https://data.bris.ac.uk/datasets/3h91syskeag572hl6tvuovwv4d"
B100 = "https://data.bris.ac.uk/datasets/2g1n6qdydwa9u22shpxqzp0t8m"
H = os.path.dirname(os.path.abspath(__file__))
_S55 = None


def url_for(vid, scripts_dir=None):
    global _S55
    if _S55 is None:
        p = os.path.join(scripts_dir or os.path.join(H, "..", "ext", "epic-kitchens-download-scripts"), "data", "epic_55_splits.csv")
        _S55 = {r["video_id"]: r["split"] for r in csv.DictReader(open(p))}
    part = vid.split("_")[0]
    if vid in _S55:
        return f"{B55}/videos/{_S55[vid]}/{part}/{vid}.MP4"
    return f"{B100}/{part}/videos/{vid}.MP4"


def _probe(url, timeout=60):
    """(total_size, supports_ranges) using a 1-byte range request."""
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        cr = r.headers.get("Content-Range")
        if r.status == 206 and cr and "/" in cr:
            return int(cr.rsplit("/", 1)[1]), True
        return int(r.headers.get("Content-Length") or 0), False


def _fetch(url, path, start, end, prog, lock, retries=20):
    pos = start
    for attempt in range(retries):
        try:
            hdr = {"User-Agent": "Mozilla/5.0"}
            if end is not None:
                hdr["Range"] = f"bytes={pos}-{end}"
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=60) as r, open(path, "r+b") as f:
                f.seek(pos)
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b); pos += len(b)
                    with lock:
                        prog[0] += len(b)
            if end is None or pos > end:
                return
        except Exception:
            time.sleep(min(30, 2 * (attempt + 1)))
    raise RuntimeError(f"range {start}-{end} failed after {retries} attempts")


def download(url, path, conns=8):
    """Returns (path, MB, MB/s). Writes to path + '.part' and renames on success."""
    size, ranges = _probe(url)
    tmp = path + ".part"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(tmp, "wb") as f:
        if size:
            f.truncate(size)
    prog, lock, t0 = [0], threading.Lock(), time.time()
    if ranges and size > 4 << 20 and conns > 1:
        step = -(-size // conns)
        jobs = [(s, min(size, s + step) - 1) for s in range(0, size, step)]
    else:
        jobs = [(0, None)]
    errs = []

    def run(s, e):
        try:
            _fetch(url, tmp, s, e, prog, lock)
        except Exception as ex:
            errs.append(ex)

    ts = [threading.Thread(target=run, args=j, daemon=True) for j in jobs]
    [t.start() for t in ts]; [t.join() for t in ts]
    if errs:
        raise errs[0]
    got = os.path.getsize(tmp)
    if size and (got != size or prog[0] < size):
        raise RuntimeError(f"size mismatch {got} / {prog[0]} vs {size}")
    os.replace(tmp, path)
    dt = max(time.time() - t0, 1e-6)
    return path, got / 1e6, got / 1e6 / dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("vid"); ap.add_argument("--conns", type=int, default=8)
    a = ap.parse_args()
    u = url_for(a.vid); print(u)
    print(download(u, os.path.join(a.out, a.vid + ".MP4"), a.conns))
