"""Sets team_id in every output JSON. usage: python tools/set_team_id.py <GOLDEN_DAWN> [outputs_dir]"""
import json, os, sys
tid = sys.argv[1]; d = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GOLDEN_DAWN", "outputs")
for f in os.listdir(d):
    if f.endswith(".json"):
        p = os.path.join(d, f); o = json.load(open(p)); o["team_id"] = tid; json.dump(o, open(p, "w"), indent=1)
cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GOLDEN_DAWN", "code", "prism", "config.py")
s = open(cfg).read().replace('"TEAM_ID_PLACEHOLDER"', json.dumps(tid)); open(cfg, "w").write(s)
print("team_id set in", len(os.listdir(d)), "files and in config.py")
