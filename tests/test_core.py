"""Tests for the TF-free parts of the vision core.

Run:  python -m pytest tests/ -q     (or: python tests/test_core.py)

These verify resolution / edge / quality / color analysis on a synthetic
image, plus captioning and the (lazy) classifier error path. No TF needed.
"""

import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analysis import (  # noqa: E402
    analyze_image, aspect_text, color_name, hex_of, human_size,
    label_for, list_images, resolution_class, sobel,
)
from caption import make_caption  # noqa: E402
from classifier import Classifier, TFNotAvailable  # noqa: E402

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")


def _temp_image():
    """A synthetic 640x480 image: red half / green half with a hard edge."""
    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr[:, :320] = (200, 30, 30)     # red
    arr[:, 320:] = (30, 180, 60)     # green
    f = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    Image.fromarray(arr).save(f.name)
    return f.name


def test_units():
    print("\nUnit checks:")
    check("human_size", human_size(2048) == "2.0 KB")
    check("aspect 4:3", aspect_text(800, 600) == "4:3")
    check("aspect about 16:9", "16:9" in aspect_text(1920, 1080))
    check("resolution class 4K", resolution_class(3840, 2160) == "4K UHD")
    check("resolution class Full HD", resolution_class(1920, 1080) == "Full HD")
    check("label_for band", label_for(5, [(3, "low"), (10, "mid"), (99, "high")]) == "mid")
    check("color_name red", color_name((210, 40, 40)) == "Red")
    check("hex_of", hex_of((255, 0, 128)) == "#ff0080")


def test_sobel():
    print("\nSobel:")
    g = np.zeros((32, 32), dtype=np.float32)
    g[:, 16:] = 1.0  # hard vertical edge
    gx, gy = sobel(g)
    check("vertical edge -> horizontal gradient", np.abs(gx).max() > np.abs(gy).max() * 3)


def test_analyze_image():
    print("\nanalyze_image:")
    p = _temp_image()
    res = analyze_image(p)
    check("has views", "Original" in res["views"] and "Sobel edges" in res["views"])
    check("flat has width", res["flat"]["width"] == 640)
    check("resolution class", res["flat"]["resolution_class"] == "SD" or
          res["flat"]["resolution_class"] == "Low (thumbnail size)")
    # hard red/green boundary => meaningful edge density
    check("edge density > 0", res["flat"]["edge_density_pct"] > 0)
    check("palette has 6 entries", len(res["palette"]) == 6)
    check("sections present", len(res["sections"]) >= 5)
    flat_light = analyze_image(p, light=True)
    check("light mode omits views", "views" not in flat_light)
    os.unlink(p)


def test_list_images():
    print("\nlist_images:")
    with tempfile.TemporaryDirectory() as d:
        for n in ("a.png", "b.jpg", "c.jpeg", "d.txt"):
            open(os.path.join(d, n), "w").close()
        found = [os.path.basename(x) for x in list_images(d)]
        check("finds png/jpg/jpeg", set(found) >= {"a.png", "b.jpg", "c.jpeg"})
        check("skips non-images", "d.txt" not in found)


def test_caption():
    print("\ncaption:")
    check("high conf", "contains" in make_caption([("Dog", 88.0), ("Cat", 3.0)]))
    check("mid conf", "possibly" in make_caption([("Dog", 40.0), ("Cat", 20.0)]))
    check("low conf", "could be" in make_caption([("Dog", 10.0), ("Cat", 9.0)]))


def test_classifier_lazy():
    print("\nclassifier (lazy TF):")
    clf = Classifier()
    check("model list", clf.available_models() == ["VGG16", "ResNet50", "EfficientNetB0"])
    try:
        clf.predict(Image.new("RGB", (224, 224)), "VGG16")
        print("  (TensorFlow is installed; prediction ran for real)")
    except TFNotAvailable:
        check("raises TFNotAvailable when TF missing", True)
    except Exception as e:
        check(f"TF present but predict errored: {e}", False)


def main():
    test_units()
    test_sobel()
    test_analyze_image()
    test_list_images()
    test_caption()
    test_classifier_lazy()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
