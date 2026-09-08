"""Generate the 4B-3 detection probe configs.

Only three things differ from the committed phase4b_r2u_z16 baseline:
  model.detection.anchors   (used by the head at decode time)
  train.anchors             (used by YOLOLoss.build_targets)
  model.detection.p2        (dp2a only; needs the matching code change)

Both anchor fields are always written together: the model decodes with the
former and assigns with the latter, and letting them diverge would silently
train one prior while evaluating another.
"""

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "configs", "phase4b_r2u_z16.yaml")

KM9 = [[[9, 8], [18, 15], [32, 24]],
       [[49, 38], [80, 52], [65, 102]],
       [[124, 82], [166, 136], [237, 214]]]
KM12 = [[[8, 7], [15, 12], [20, 21]],
        [[35, 19], [38, 32], [64, 41]],
        [[41, 72], [87, 63], [135, 83]],
        [[98, 128], [180, 139], [241, 221]]]


def main():
    with open(BASE, encoding="utf-8") as f:
        base = yaml.safe_load(f)

    def make(cell, anchors, p2=None):
        cfg = yaml.safe_load(yaml.safe_dump(base))     # deep copy
        det = cfg["model"]["detection"]
        det["anchors"] = anchors
        if p2 is not None:
            det["p2"] = p2
        cfg["train"]["anchors"] = anchors
        path = os.path.join(ROOT, "configs", f"phase4b_{cell}.yaml")
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
        return path

    made = [make("danc_z16", KM9), make("dp2a_z16", KM12, p2="up")]

    # audit: print exactly what differs from the baseline
    print("diff vs configs/phase4b_r2u_z16.yaml")
    for p in made:
        with open(p, encoding="utf-8") as f:
            new = yaml.safe_load(f)
        diffs = []
        if new["model"]["detection"].get("anchors") != base["model"]["detection"].get("anchors"):
            diffs.append("model.detection.anchors")
        if new["model"]["detection"].get("p2") != base["model"]["detection"].get("p2"):
            diffs.append(f"model.detection.p2={new['model']['detection'].get('p2')}")
        if new["train"].get("anchors") != base["train"].get("anchors"):
            diffs.append("train.anchors")
        for key in ("encoder", "representation", "segmentation"):
            if new["model"].get(key) != base["model"].get(key):
                diffs.append(f"UNEXPECTED model.{key}")
        for key in base["train"]:
            if key == "anchors":
                continue
            if new["train"].get(key) != base["train"].get(key):
                diffs.append(f"UNEXPECTED train.{key}={new['train'].get(key)}")
        print(f"  {os.path.basename(p)}: {', '.join(diffs)}")


if __name__ == "__main__":
    main()
