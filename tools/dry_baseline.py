"""Write baseline JSONs for all manifest videos WITHOUT the videos, using manifest durations.
For testing the writer, validator and scorer only. Never submit these outputs."""
import csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "TEAM_ID", "code"))
from prism import baseline, writer

man, out = sys.argv[1], sys.argv[2]
for r in csv.DictReader(open(man)):
    d = float(r["duration_sec"])
    lab = baseline.label_head(None, d)
    acts = baseline.action_head(None, d)
    hl, pr = baseline.highlight_head(None, d, lab, acts)
    writer.write(os.path.join(out, r["video_id"] + ".json"), writer.build(r["video_id"], d, lab, acts, hl, pr))
print("done")
