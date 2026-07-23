"""One-off, idempotent: stamp the pre-Day-14 result JSONs with the new run-metadata fields so the
schema is rectangular before the first joint row is written. Adds ONLY missing keys; re-running is a
no-op. Does NOT recompute anything -- the per-origin artifact needs raw predictions the old runs did
not persist, so single runs get per-origin only by re-running `train.loop --all` (which also writes
these fields natively, making this backfill unnecessary if you re-run).

  encoder runs -> training_regime=single, sampler=None, gate_mode=learned, topo_aug=none
  naive floors -> training_regime=single, sampler=None, gate_mode=None,    topo_aug=None

Run from the repo root:  python backfill_schema.py
"""
import glob
import json
import os

R = "results"
ENC_META = dict(training_regime="single", sampler=None, gate_mode="learned", topo_aug="none")
NAIVE_META = dict(training_regime="single", sampler=None, gate_mode=None, topo_aug=None)


def backfill(pattern, meta):
    for f in sorted(glob.glob(os.path.join(R, pattern))):
        if "smoke" in f:
            continue
        recs = json.load(open(f))
        added = 0
        for r in recs:
            for k, v in meta.items():
                if k not in r:
                    r[k] = v; added += 1
        with open(f, "w") as fh:
            json.dump(recs, fh, indent=2)
        tag = f"+{added} fields" if added else "already current"
        print(f"  {os.path.basename(f):45s} {tag}")


if __name__ == "__main__":
    print("encoder runs:")
    backfill("encoder__*.json", ENC_META)
    print("naive floors:")
    backfill("naive__*.json", NAIVE_META)
    print("done. re-run is a no-op (idempotent).")
