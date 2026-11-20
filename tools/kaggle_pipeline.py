"""End-to-end GPU run (built for a Kaggle notebook, works on any Linux GPU box).

For each manifest video: download the ORIGINAL (EPIC from data.bris, shortest first) -> run the real
run.sh pipeline in-process (features cached) -> delete the video. Prefetches the next download while
the GPU works. Checks the action model's accuracy on the first EPIC videos and stops if it is broken.
Stops starting new videos after --max-hours so results are saved before Kaggle's 12 h limit, and
resumes from a previous run attached as input. Everything lands in <work>/prism_results(.zip).

usage: python tools/kaggle_pipeline.py [--work /kaggle/working] [--max-hours 11] [--only epic|fpvsum|all]
                                       [--shard 0/2] [--fpvsum-dir DIR] [--skip-gate]
testing: --local-videos DIR --fake-models"""
import argparse, collections, concurrent.futures as cf, csv, glob, json, os, re, shutil, subprocess, sys, time

H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(H, ".."))
T0 = time.time()


def log(*a):
    msg = f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:5.1f}m] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if LOGF:
        LOGF.write(msg + "\n"); LOGF.flush()


LOGF = None


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), **kw)


def ensure_env(work, fake):
    if not shutil.which("ffmpeg"):
        log("ffmpeg not found: installing a static build via imageio-ffmpeg")
        sh([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"], check=True)
        import imageio_ffmpeg
        b = os.path.join(work, "bin"); os.makedirs(b, exist_ok=True)
        link = os.path.join(b, "ffmpeg")
        if not os.path.exists(link):
            os.symlink(imageio_ffmpeg.get_ffmpeg_exe(), link)
        os.environ["PATH"] = b + os.pathsep + os.environ["PATH"]
    need = []
    for mod, pkg in [("einops", "einops"), ("timm", "timm"), ("transformers", "transformers"),
                     ("sentencepiece", "sentencepiece"), ("scipy", "scipy"), ("yt_dlp", "yt-dlp")]:
        try:
            __import__(mod)
        except ImportError:
            need.append(pkg)
    if need and os.environ.get("PRISM_SKIP_PIP") != "1":
        log("pip install", need); sh([sys.executable, "-m", "pip", "install", "-q", *need], check=True)
    os.makedirs(os.path.join(ROOT, "ext"), exist_ok=True)
    for url, d in [("https://github.com/epic-kitchens/epic-kitchens-100-annotations.git", "epic-kitchens-100-annotations"),
                   ("https://github.com/azuxmioy/fpvsum.git", "fpvsum"),
                   ("https://github.com/epic-kitchens/epic-kitchens-download-scripts.git", "epic-kitchens-download-scripts")]:
        if not os.path.isdir(os.path.join(ROOT, "ext", d)):
            sh(["git", "clone", "-q", "--depth", "1", url, os.path.join(ROOT, "ext", d)], check=True)
    if not fake:
        if sh(["bash", os.path.join(H, "fetch_weights.sh")]).returncode != 0:
            sys.exit("weight download failed")
        import torch
        log("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO GPU: enable a GPU accelerator!")


def resume(res):
    n = 0
    for d in glob.glob("/kaggle/input/**/prism_results", recursive=True):
        for sub in ["outputs", "cache"]:
            for f in glob.glob(os.path.join(d, sub, "*")):
                dst = os.path.join(res, sub, os.path.basename(f))
                if not os.path.exists(dst):
                    shutil.copy2(f, dst); n += 1
    if n:
        log(f"resumed {n} files from a previous run")


def find_video(root, vid):
    for p in glob.glob(os.path.join(root, "**", vid + ".*"), recursive=True):
        if p.lower().endswith(".mp4") and ".part" not in p:
            return p
    return None


def main():
    global LOGF
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working"); ap.add_argument("--raw", default="/tmp/prism_raw")
    ap.add_argument("--max-hours", type=float, default=11.0); ap.add_argument("--only", default="all", choices=["all", "epic", "fpvsum"])
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--fpvsum-dir"); ap.add_argument("--skip-gate", action="store_true")
    ap.add_argument("--gate-videos", type=int, default=6); ap.add_argument("--batch", default="8")
    ap.add_argument("--dl-workers", type=int, default=4); ap.add_argument("--conns", type=int, default=8)
    ap.add_argument("--min-free-gb", type=float, default=25)
    ap.add_argument("--local-videos"); ap.add_argument("--fake-models", action="store_true")
    a = ap.parse_args()
    res = os.path.join(a.work, "prism_results")
    out, cache = os.path.join(res, "outputs"), os.path.join(res, "cache")
    for d in [out, cache, a.raw]:
        os.makedirs(d, exist_ok=True)
    LOGF = open(os.path.join(res, "pipeline.log"), "a")
    ensure_env(a.work, a.fake_models)
    resume(res)
    os.environ.update(PRISM_CACHE=cache, PRISM_BATCH=a.batch, PRISM_STRICT="1")
    sys.path.insert(0, os.path.join(ROOT, "TEAM_ID", "code"))
    if a.fake_models:
        sys.path.insert(0, H); import fake_models; fake_models.install(); log("FAKE MODELS: plumbing test only")
    from prism import main as M

    man = list(csv.DictReader(open(os.path.join(ROOT, "data", "manifest.csv"))))
    k, n = map(int, a.shard.split("/"))
    epic = sorted([r for r in man if r["dataset"] == "epic_kitchens_100"], key=lambda r: float(r["duration_sec"]))
    epic = [r["video_id"] for i, r in enumerate(epic) if i % n == k]
    fpv = [r["video_id"] for i, r in enumerate(r for r in man if r["dataset"] == "fpvsum") if i % n == k]
    done = lambda v: os.path.exists(os.path.join(out, v + ".json"))
    deadline = T0 + a.max_hours * 3600
    failed, stats = [], collections.Counter()

    def process(vid, path):
        t = time.time()
        try:
            M.run(path, os.path.join(out, vid + ".json"))
            o = json.load(open(os.path.join(out, vid + ".json")))
            stats[o["label"]["name"]] += 1
            log(f"done {vid}: {o['label']['name']} {len(o['actions'])} actions {len(o['highlights'])} highlights "
                f"in {time.time() - t:.0f}s")
        except Exception as ex:
            failed.append(vid); log(f"FAILED {vid}: {type(ex).__name__}: {ex}")

    # ---------------- EPIC
    if a.only in ("all", "epic"):
        todo = [v for v in epic if not done(v)]
        log(f"EPIC: {len(epic) - len(todo)} already done, {len(todo)} to go")

        sys.path.insert(0, H)
        import epic_fast_dl

        def download(vid):
            if a.local_videos:
                return find_video(a.local_videos, vid)
            dst = os.path.join(a.raw, vid + ".MP4")
            for attempt in range(2):
                try:
                    _, mb, rate = epic_fast_dl.download(epic_fast_dl.url_for(vid), dst, conns=a.conns)
                    log(f"downloaded {vid}: {mb:.0f} MB at {rate:.1f} MB/s")
                    return dst
                except Exception as ex:
                    log(f"fast download failed for {vid} ({type(ex).__name__}: {ex}); attempt {attempt + 1}")
            log(f"falling back to the official downloader for {vid}")
            sh([sys.executable, "epic_downloader.py", "--videos", "--specific-videos", vid, "--output-path", a.raw],
               cwd=os.path.join(ROOT, "ext", "epic-kitchens-download-scripts"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return find_video(a.raw, vid)

        def disk_ok():
            return shutil.disk_usage(a.raw).free > a.min_free_gb * 1e9

        gate_pending = not (a.skip_gate or a.fake_models or len(epic) - len(todo) >= a.gate_videos)
        held = []
        t_start, bytes_done = time.time(), 0
        with cf.ThreadPoolExecutor(a.dl_workers) as pool:
            futs, nxt = {}, 0
            for i, vid in enumerate(todo):
                while nxt < len(todo) and nxt <= i + a.dl_workers and (nxt == i or disk_ok()):
                    futs[nxt] = pool.submit(download, todo[nxt]); nxt += 1
                if i not in futs:
                    futs[i] = pool.submit(download, vid); nxt = max(nxt, i + 1)
                path = futs.pop(i).result()
                if time.time() > deadline:
                    log("time budget reached: stopping (re-run with this output attached to resume)"); break
                if not path:
                    failed.append(vid); log(f"FAILED download {vid}"); continue
                mb = os.path.getsize(path) / 1e6; bytes_done += mb
                el = (time.time() - t_start) / 3600
                log(f"[{i + 1}/{len(todo)}] {vid} ({mb:.0f} MB) | overall {bytes_done / max(el, 1e-6) / 3600:.1f} MB/s")
                process(vid, path)
                if gate_pending:
                    held.append(path)
                    if len(held) >= a.gate_videos:
                        r = sh([sys.executable, os.path.join(H, "check_action_model.py"), "--max-clips", "120", a.raw],
                               capture_output=True, text=True)
                        log("ACCURACY GATE:", r.stdout.strip().replace("\n", " | "), r.stderr.strip()[-500:])
                        m = re.search(r"verb top-1 ([\d.]+)%", r.stdout)
                        if not m or float(m.group(1)) < 35:
                            log("STOP: action model accuracy is far below expected (verb 65-70%). Send pipeline.log to Claude.")
                            for f in futs.values():
                                f.cancel()
                            break
                        gate_pending = False
                        for p in held:
                            os.remove(p)
                        held = []
                elif not a.local_videos:
                    os.remove(path)
            for f in futs.values():
                f.cancel()

    # ---------------- FPVSum
    if a.only in ("all", "fpvsum") and time.time() < deadline:
        src = a.fpvsum_dir
        if not src:
            hits = [d for d in glob.glob("/kaggle/input/**/", recursive=True)
                    if any(find_video(d, v) for v in fpv[:3])]
            src = hits[0] if hits else None
        if not src and a.local_videos:
            src = a.local_videos
        todo = [v for v in fpv if not done(v)]
        log(f"FPVSum: {len(fpv) - len(todo)} already done, {len(todo)} to go, videos from {src or 'YouTube (yt-dlp)'}")
        for vid in todo:
            if time.time() > deadline:
                log("time budget reached"); break
            path = find_video(src, vid) if src else None
            if not path and not src:
                sh([sys.executable, os.path.join(H, "download_fpvsum.py"), os.path.join(a.raw, "fpvsum")],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) if vid == todo[0] else None
                path = find_video(os.path.join(a.raw, "fpvsum"), vid)
            if not path:
                failed.append(vid); log(f"FAILED: no video for {vid}"); continue
            process(vid, path)

    # ---------------- summary
    n_out = len(glob.glob(os.path.join(out, "*.json")))
    log(f"FINISHED: {n_out}/187 outputs; {len(failed)} not processed this run: {failed[:12]}{' ...' if len(failed) > 12 else ''}")
    log("labels predicted this run:", dict(stats))
    for cmd in [[sys.executable, os.path.join(H, "validate_strict.py"), "--dir", out],
                [sys.executable, os.path.join(H, "score.py"), "--pred", out]]:
        r = sh(cmd, capture_output=True, text=True)
        log("\n" + r.stdout.strip() + ("\n" + r.stderr.strip()[-800:] if r.returncode else ""))
    LOGF.close()
    shutil.make_archive(os.path.join(a.work, "prism_results"), "zip", a.work, "prism_results")
    print(f"\nResults: {os.path.join(a.work, 'prism_results.zip')}  (download it from the notebook's Output tab)")


if __name__ == "__main__":
    main()
