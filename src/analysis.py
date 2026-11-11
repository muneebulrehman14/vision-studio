"""Core image analysis. Deterministic and TensorFlow-free.

This module owns every pixel-level measurement (resolution, edges, quality,
color, histogram, EXIF) and is shared by both the web and desktop front ends.
TensorFlow-based classification lives in :mod:`classifier` and is imported
lazily so all of these features work even when TF is not installed.

Optional: install ``opencv-python`` to get true Canny edges. Without it the
edge-mask fallback is used and the label is renamed to "Edge mask".
"""

import math
import os
import time
from datetime import datetime

import numpy as np
from PIL import ExifTags, Image, ImageDraw, ImageFilter, ImageOps

# optional OpenCV for true Canny edges
try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

import glob  # noqa: E402

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff")


def list_images(folder):
    """Return a sorted, de-duplicated list of image paths inside ``folder``."""
    paths = []
    for ext in IMG_EXTS:
        paths += glob.glob(os.path.join(folder, "*" + ext))
        paths += glob.glob(os.path.join(folder, "*" + ext.upper()))
    return sorted(set(paths))



MODEL_NAMES = ["VGG16", "ResNet50", "EfficientNetB0"]

COLOR_NAMES = [
    ("Black", (15, 15, 15)), ("White", (245, 245, 245)), ("Gray", (128, 128, 128)),
    ("Light gray", (200, 200, 200)), ("Red", (210, 40, 40)), ("Orange", (240, 140, 30)),
    ("Yellow", (240, 220, 50)), ("Green", (50, 160, 70)), ("Dark green", (20, 90, 40)),
    ("Teal", (30, 160, 160)), ("Sky blue", (120, 190, 240)), ("Blue", (40, 80, 200)),
    ("Navy", (20, 30, 90)), ("Purple", (130, 60, 170)), ("Pink", (240, 140, 180)),
    ("Brown", (120, 75, 40)), ("Beige", (225, 205, 170)),
]


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def human_size(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def aspect_text(w, h):
    g = math.gcd(w, h)
    a, b = w // g, h // g
    if max(a, b) <= 30:
        return f"{a}:{b}"
    r = w / h
    for x, y in [(16, 9), (4, 3), (3, 2), (1, 1), (5, 4), (21, 9), (16, 10),
                 (9, 16), (3, 4), (2, 3)]:
        if abs(r - x / y) < 0.02:
            return f"about {x}:{y}"
    return f"{r:.2f}:1"


def resolution_class(w, h):
    longest = max(w, h)
    for limit, name in [(7680, "8K UHD"), (3840, "4K UHD"), (2560, "QHD (2K)"),
                        (1920, "Full HD"), (1280, "HD"), (640, "SD")]:
        if longest >= limit:
            return name
    return "Low (thumbnail size)"


def sobel(gray):
    p = np.pad(gray, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:])
    return gx, gy


def laplacian_variance(gray):
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def label_for(value, bands):
    """bands = [(upper_limit, text), ...] ascending; last text catches the rest."""
    for limit, text in bands[:-1]:
        if value < limit:
            return text
    return bands[-1][1]


def color_name(rgb):
    return min(COLOR_NAMES, key=lambda c: sum((a - b) ** 2 for a, b in zip(c[1], rgb)))[0]


def hex_of(rgb):
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in rgb)


def read_exif(raw):
    """Return (list of (label, value), has_gps)."""
    rows, gps = [], False
    try:
        ex = raw.getexif()
        if not ex:
            return rows, gps
        tags = {ExifTags.TAGS.get(k, k): v for k, v in ex.items()}
        try:
            tags.update({ExifTags.TAGS.get(k, k): v for k, v in ex.get_ifd(0x8769).items()})
        except Exception:
            pass
        gps = 0x8825 in ex
        wanted = [("Make", "Camera make"), ("Model", "Camera model"), ("LensModel", "Lens"),
                  ("DateTimeOriginal", "Taken"), ("DateTime", "Modified (EXIF)"),
                  ("ExposureTime", "Exposure"), ("FNumber", "Aperture"),
                  ("ISOSpeedRatings", "ISO"), ("FocalLength", "Focal length"),
                  ("Flash", "Flash"), ("Software", "Software")]
        for key, label in wanted:
            if key not in tags:
                continue
            v = tags[key]
            try:
                if key == "ExposureTime":
                    f = float(v)
                    v = f"1/{round(1 / f)} s" if 0 < f < 1 else f"{f:g} s"
                elif key == "FNumber":
                    v = f"f/{float(v):.1f}"
                elif key == "FocalLength":
                    v = f"{float(v):.0f} mm"
                else:
                    v = str(v).replace("\x00", "").strip()[:60]
            except Exception:
                v = str(v)[:60]
            if key == "DateTime" and "DateTimeOriginal" in tags:
                continue
            rows.append((label, v))
    except Exception:
        pass
    return rows, gps


def make_histogram(arr, size=(640, 360)):
    w, h = size
    left, right, top, bottom = 44, w - 18, 20, h - 38
    base = Image.new("RGBA", size, (255, 255, 255, 255))
    hists = [np.bincount(arr[..., i].ravel(), minlength=256).astype(float) for i in range(3)]
    cap = float(np.percentile(np.concatenate(hists), 99.5)) or 1.0
    for hist, col in zip(hists, [(228, 60, 60), (40, 165, 80), (60, 100, 240)]):
        hist = np.convolve(hist, np.ones(3) / 3, mode="same")
        v = np.clip(hist / cap, 0, 1)
        pts = [(left + (right - left) * i / 255, bottom - (bottom - top) * v[i]) for i in range(256)]
        layer = Image.new("RGBA", size, (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ld.polygon([(left, bottom)] + pts + [(right, bottom)], fill=col + (60,))
        ld.line(pts, fill=col + (255,), width=2)
        base = Image.alpha_composite(base, layer)
    d = ImageDraw.Draw(base)
    d.line([(left, bottom), (right, bottom)], fill=(150, 158, 190, 255), width=1)
    d.line([(left, top), (left, bottom)], fill=(150, 158, 190, 255), width=1)
    for val in (0, 64, 128, 192, 255):
        x = left + (right - left) * val / 255
        d.line([(x, bottom), (x, bottom + 5)], fill=(150, 158, 190, 255))
        d.text((x - 8, bottom + 9), str(val), fill=(107, 115, 153, 255))
    d.text((left + 4, 3), "Pixel count by brightness level (red / green / blue)",
           fill=(107, 115, 153, 255))
    return base.convert("RGB")


# ----------------------------------------------------------------------------
# the main analysis
# ----------------------------------------------------------------------------
def analyze_image(path, light=False):
    """Full analysis of one image file. ``light=True`` skips the preview images.

    Returns a dict with keys:
      path, rgb, sections, palette, short, flat, canny_title, seconds
      and (unless light) views: Original / Sobel / Canny(or Edge mask) / Grayscale / Histogram
    """
    t0 = time.time()
    raw = Image.open(path)
    fmt = raw.format or os.path.splitext(path)[1][1:].upper()
    mode = raw.mode
    dpi = raw.info.get("dpi")
    frames = getattr(raw, "n_frames", 1)
    has_alpha = mode in ("RGBA", "LA", "PA") or "transparency" in raw.info
    exif_rows, has_gps = read_exif(raw)
    img = ImageOps.exif_transpose(raw).convert("RGB")
    W, H = img.size

    work = img.copy()
    work.thumbnail((1024, 1024), Image.LANCZOS)
    rgb = np.asarray(work)
    gray = np.asarray(work.convert("L"), dtype=np.float32)

    # ---- edges -------------------------------------------------------------
    gx, gy = sobel(gray)
    mag = np.hypot(gx, gy)
    thr = max(60.0, float(mag.mean() + mag.std()))
    edge_mask = mag > thr
    edge_density = float(edge_mask.mean() * 100)
    sx, sy = float(np.abs(gx).sum()), float(np.abs(gy).sum())
    vert_share = sx / (sx + sy + 1e-9)
    direction = ("Mostly vertical edges" if vert_share > 0.58
                 else "Mostly horizontal edges" if vert_share < 0.42
                 else "Balanced directions")
    edge_level = label_for(edge_density,
                           [(3, "Smooth, little detail"), (8, "Moderate detail"),
                            (15, "Detailed"), (100, "Very busy / textured")])

    canny_pct = None
    if cv2 is not None:
        g8 = np.clip(gray, 0, 255).astype(np.uint8)
        g8 = cv2.GaussianBlur(g8, (5, 5), 0)
        med = float(np.median(g8))
        canny = cv2.Canny(g8, int(max(0, 0.66 * med)), int(min(255, 1.33 * med + 20)))
        canny_pct = float((canny > 0).mean() * 100)
        canny_img = Image.fromarray(255 - canny).convert("RGB")
        canny_title = "Canny edges"
    else:
        canny_img = Image.fromarray(np.where(edge_mask, 0, 255).astype(np.uint8)).convert("RGB")
        canny_title = "Edge mask"

    # ---- quality ----------------------------------------------------------
    sharp = laplacian_variance(gray)
    sharp_txt = label_for(sharp,
                          [(30, "Very soft / blurry"), (100, "Soft"),
                           (500, "Sharp"), (10 ** 9, "Very sharp")])
    brightness = float(gray.mean() / 255 * 100)
    bright_txt = label_for(brightness,
                           [(30, "Very dark"), (42, "Dark"), (62, "Well exposed"),
                            (75, "Bright"), (101, "Very bright")])
    contrast = float(gray.std())
    contrast_txt = label_for(contrast,
                             [(30, "Flat / low contrast"), (60, "Normal contrast"),
                              (999, "High contrast")])
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    p = hist[hist > 0] / hist.sum()
    entropy = float(-(p * np.log2(p)).sum())
    clip_dark = float((gray < 5).mean() * 100)
    clip_bright = float((gray > 250).mean() * 100)

    # ---- color -------------------------------------------------------------
    avg = rgb.reshape(-1, 3).mean(0)
    hsv = np.asarray(work.convert("HSV"))
    sat = float(hsv[..., 1].mean() / 255 * 100)
    warm = float(avg[0] - avg[2])
    temp = "Warm" if warm > 12 else "Cool" if warm < -12 else "Neutral"
    if sat < 6:
        temp = "Black & white / neutral"
    q = work.resize((120, 120)).quantize(colors=6)
    pal = q.getpalette()
    counts = sorted(q.getcolors(), reverse=True)
    total = float(sum(c for c, _ in counts))
    palette = [(c / total * 100, tuple(pal[i * 3:i * 3 + 3])) for c, i in counts]

    # ---- sections for the details panel ------------------------------------
    st = os.stat(path)
    mp = W * H / 1e6
    sections = [
        ("File", [
            ("Name", os.path.basename(path)),
            ("Folder", os.path.dirname(os.path.abspath(path))),
            ("Format", f"{fmt}  ({mode})"),
            ("File size", human_size(st.st_size)),
            ("Modified", datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")),
            ("Transparency", "Yes" if has_alpha else "No"),
            ("Frames", str(frames)),
        ]),
        ("Resolution", [
            ("Dimensions", f"{W} x {H} px"),
            ("Megapixels", f"{mp:.2f} MP"),
            ("Total pixels", f"{W * H:,}"),
            ("Aspect ratio", aspect_text(W, H)),
            ("Orientation", "Square" if W == H else "Landscape" if W > H else "Portrait"),
            ("Resolution class", resolution_class(W, H)),
            ("Stored DPI", f"{dpi[0]:.0f} x {dpi[1]:.0f}" if dpi else "Not stored"),
            ("Print size at 300 DPI",
             f"{W / 300:.1f} x {H / 300:.1f} in  ({W / 300 * 2.54:.1f} x {H / 300 * 2.54:.1f} cm)"),
            ("Bits per pixel", f"{len(raw.getbands()) * 8} ({'+'.join(raw.getbands())})"),
            ("Model input", "Resized to 224 x 224 for VGG16 / ResNet50 / EfficientNetB0"),
        ]),
        ("Edges", [
            ("Edge density (Sobel)", f"{edge_density:.1f}%  of pixels"),
            ("Detail level", edge_level),
            ("Dominant direction", direction),
            ("Vertical / horizontal edge share",
             f"{vert_share * 100:.0f}% / {(1 - vert_share) * 100:.0f}%"),
            ("Strongest gradient", f"{float(mag.max()):.0f}"),
            (canny_title,
             f"{canny_pct:.1f}%" if canny_pct is not None
             else "Install opencv-python to enable"),
        ]),
        ("Quality", [
            ("Sharpness", f"{sharp_txt}  (score {sharp:.0f})"),
            ("Brightness", f"{bright_txt}  ({brightness:.0f}%)"),
            ("Contrast", f"{contrast_txt}  (std {contrast:.0f})"),
            ("Detail richness (entropy)", f"{entropy:.2f} / 8 bits"),
            ("Pure black pixels", f"{clip_dark:.1f}%"),
            ("Pure white pixels", f"{clip_bright:.1f}%"),
        ]),
        ("Color", [
            ("Average color", f"{hex_of(avg)}  ({color_name(avg)})"),
            ("Dominant color",
             f"{hex_of(palette[0][1])}  ({color_name(palette[0][1])}, {palette[0][0]:.0f}%)"),
            ("Saturation", f"{sat:.0f}%"),
            ("Temperature", temp),
        ]),
        ("EXIF / metadata",
         (exif_rows + ([("GPS location", "Present in file")] if has_gps else []))
         or [("EXIF", "No EXIF metadata in this file")]),
    ]

    short = (f"{W} x {H}  |  {mp:.1f} MP  |  {resolution_class(W, H)}  |  {sharp_txt}  |  "
             f"edges {edge_density:.1f}%  |  {human_size(st.st_size)}")

    flat = {
        "file": os.path.basename(path), "format": fmt, "width": W, "height": H,
        "megapixels": round(mp, 2), "resolution_class": resolution_class(W, H),
        "aspect_ratio": aspect_text(W, H), "file_size_kb": round(st.st_size / 1024, 1),
        "sharpness_score": round(sharp, 1), "sharpness": sharp_txt,
        "brightness_pct": round(brightness, 1), "contrast_std": round(contrast, 1),
        "edge_density_pct": round(edge_density, 2), "edge_direction": direction,
        "avg_color": hex_of(avg), "dominant_color": color_name(palette[0][1]),
        "saturation_pct": round(sat, 1),
    }

    res = {
        "path": path, "rgb": img, "sections": sections, "palette": palette, "short": short,
        "flat": flat, "canny_title": canny_title, "seconds": time.time() - t0,
    }
    if not light:
        m = np.clip(mag / (np.percentile(mag, 99) + 1e-6), 0, 1)
        res["views"] = {
            "Original": img,
            "Sobel edges": Image.fromarray((255 * (1 - m)).astype(np.uint8)).convert("RGB"),
            canny_title: canny_img,
            "Grayscale": ImageOps.grayscale(img).convert("RGB"),
            "Histogram": make_histogram(rgb),
        }
    return res
