"""Unified entry point for the Vision Project.

Usage:
    python src/main.py                      # auto: web if gradio, else desktop
    python src/main.py --ui web            # Gradio 3D web UI
    python src/main.py --ui desktop        # Tkinter desktop studio
    python src/main.py photo.jpg           # headless: analyze and print
    python src/main.py photo.jpg --csv out.csv
    python src/main.py --folder photos/    # headless batch over a folder
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _try_web():
    try:
        import gradio  # noqa: F401
        return True
    except ImportError:
        return False


def _try_desktop():
    try:
        import tkinter  # noqa: F401
        return True
    except ImportError:
        return False


def main():
    ap = argparse.ArgumentParser(
        prog="vision",
        description="Vision Reader / Studio — pretrained image analysis.")
    ap.add_argument("image", nargs="?",
                   help="path to an image (used by --csv / headless)")
    ap.add_argument("--ui", choices=["web", "desktop", "auto"], default="auto",
                   help="which interface to launch (default: auto)")
    ap.add_argument("--folder", help="run headless over a folder of images")
    ap.add_argument("--csv", help="write a CSV report (implies headless)")
    ap.add_argument("--model", default="VGG16",
                   choices=["VGG16", "ResNet50", "EfficientNetB0"],
                   help="single-model mode (headless only)")
    ap.add_argument("--share", action="store_true",
                   help="publish a public Gradio link (web only)")
    ap.add_argument("--no-browser", action="store_true",
                   help="don't auto-open the web UI in a browser")
    args = ap.parse_args()

    if args.csv or (args.folder and not args.ui) or (args.image and not args.ui):
        return _run_headless(args)

    ui = args.ui
    if ui == "auto":
        ui = "web" if _try_web() else ("desktop" if _try_desktop() else "none")
    if ui == "web":
        from vision_web import launch_ui
        launch_ui(share=args.share)
    elif ui == "desktop":
        from vision_desktop import App
        App(args.image).run()
    else:
        print("No UI is available in this environment.")
        print("  pip install gradio        # for the web UI")
        print("  # or use tkinter on Linux: sudo apt install python3-tk")
        print("Headless still works:  python src/main.py photo.jpg --csv out.csv")
        return 1


def _run_headless(args):
    import csv as _csv
    from analysis import analyze_image, list_images
    from caption import make_caption
    from classifier import Classifier, TFNotAvailable
    from PIL import Image

    clf = Classifier()
    model = args.model or "VGG16"

    def process_one(path):
        res = analyze_image(path, light=False)
        row = dict(res["flat"])
        img = Image.open(path).convert("RGB")
        try:
            preds, secs = clf.predict(img, model)
            row[f"{model}_top1"] = preds[0][0]
            row[f"{model}_conf_pct"] = round(preds[0][1], 1)
            row[f"{model}_caption"] = make_caption(preds)
            row[f"{model}_ms"] = int(secs * 1000)
        except TFNotAvailable as e:
            row[f"{model}_top1"] = "TF not installed"
        row[f"{model}_caption"] = "(object prediction needs TensorFlow)"
        print(f"{os.path.basename(path):<30} -> analysis OK (TF not installed)")
        return row

    if args.folder:
        paths = list_images(args.folder)
        if not paths:
            print(f"No images found in {args.folder}")
            return 1
        rows = [process_one(p) for p in paths]
    else:
        if not args.image:
            print("No input image given.")
            return 1
        rows = [process_one(args.image)]

    if args.csv:
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV report saved: {args.csv}  ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
