"""Builds GOLDEN_DAWN/code/prism/pair_prior.json from the EPIC-100 TRAIN annotations only."""
import collections, csv, json, os
H = os.path.dirname(os.path.abspath(__file__)); E = os.path.join(H, "..", "ext", "epic-kitchens-100-annotations")
vk = [r["key"] for r in csv.DictReader(open(os.path.join(E, "EPIC_100_verb_classes.csv")))]
nk = [r["key"] for r in csv.DictReader(open(os.path.join(E, "EPIC_100_noun_classes.csv")))]
c = collections.Counter((int(r["verb_class"]), int(r["noun_class"])) for r in csv.DictReader(open(os.path.join(E, "EPIC_100_train.csv"))))
json.dump({"verbs": vk, "nouns": nk, "pairs": [[v, n, k] for (v, n), k in sorted(c.items())]},
          open(os.path.join(H, "..", "GOLDEN_DAWN", "code", "prism", "pair_prior.json"), "w"))
print(len(c), "train pairs")
