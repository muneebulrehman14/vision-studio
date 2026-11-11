"""Headless batch runner: analyze a folder (or one image), write a CSV report.

No GUI needed. The pixel-level analysis (edges, quality, color, resolution)
always works; model predictions are included when TensorFlow is installed.

Usage:
    python src/batch.py photos/                 # CSV to stdout
    python src/batch.py photos/ --csv out.csv    # write a file
    python src/batch.py photo.jpg --models EfficientNetB0
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import IMG_EXTS, analyze_image, list_images  # noqa: E402
from caption import make_caption  # noqa: E402
from classifier import Classifier, TFNotAvailable  # noqa: E402


def list_images_recursive(folder):
    paths = []
    for root, _dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(IMG_EXTS):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def run(folder=None, image=None, models=None, out=None, recursive=False):
    clf = Classifier()
    paths = []
    if folder:
        if not os.path.isdir(folder):
            print(f"Folder not found: {folder}")
            return 1
        paths = list_images_recursive(folder) if recursive else list_images(folder)
    elif image:
        paths = [image]
    else:
        print("Give a folder (and/or --recursive) or a single image path.")
        return 1

    if not paths:
        print(f"No images found in {folder}")
        return 1

    models = models or clf.available_models()
    print(f"Analyzing {len(paths)} image(s) with {', '.join(models)} ...")

    rows = []
    for i, p in enumerate(paths, 1):
        print(f"[{i}/{len(paths)}] {os.path.basename(p)}", end="  ", flush=True)
        try:
            res = analyze_image(p, light=True)
        except Exception as e:
            rows.append({"file": os.path.basename(p), "error": str(e)})
            print(f"FAILED ({e})")
            continue
        row = dict(res["flat"])
        tf_missing = False
        for m in models:
            try:
                from PIL import Image
                img = Image.open(p).convert("RGB")
                preds, secs = clf.predict(img, m)
                row[f"{m}_top1"] = preds[0][0]
                row[f"{m}_conf_pct"] = round(preds[0][1], 1)
                row[f"{m}_caption"] = make_caption(preds)
                row[f"{m}_ms"] = int(secs * 1000)
            except TFNotAvailable:
                tf_missing = True
                break
            except Exception as e:
                row[f"{m}_top1"] = f"error: {type(e).__name__}"
        tag = " (TF not installed, analysis-only)" if tf_missing else ""
        print(f"OK{tag}")
        rows.append(row)

    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    cols.sort(key=lambda k: (k not in ("file", "error"), k))
    if out:
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nSaved {len(rows)} row(s) to {out}")
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Headless Vision batch runner")
    ap.add_argument("folder", nargs="?", help="folder of images")
    ap.add_argument("-i", "--image", help="a single image instead of a folder")
    ap.add_argument("-o", "--csv", help="write results to this CSV file")
    ap.add_argument("--models", nargs="*", default=None,
                    help="models to run (default: all three)")
    ap.add_argument("--recursive", action="store_true",
                    help="walk subfolders when using a folder")
    a = ap.parse_args()
    sys.exit(run(a.folder, a.image, a.models, a.csv, a.recursive) or 0)
