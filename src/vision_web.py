"""Vision Reader — web UI (Gradio).

A light, professional command-center with real 3D depth:
  * rotating 3D cube hero + extruded title
  * floating 3D result cards with per-model accent + glow bars
  * 3D push buttons
  * consensus banner (agree / disagree)
Tabs:  Single image · Compare 3 models · Batch / folder

Launch with:  python src/vision_web.py      (or via src/main.py --ui web)
"""

import base64
import html
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import list_images  # noqa: E402
from caption import make_caption  # noqa: E402
from classifier import Classifier, TFNotAvailable  # noqa: E402
from theme import HERO, MODEL_COLORS, WEB_CSS  # noqa: E402

CLF = Classifier()


# ===========================================================================
# image helpers
# ===========================================================================
def _open(image):
    img = Image.open(image) if isinstance(image, (str, os.PathLike)) else image
    return img.convert("RGB")


def predict(img, model_name, top=5):
    return CLF.predict(img, model_name, top=top)


# ===========================================================================
# card builders (3D look via the CSS in theme.WEB_CSS)
# ===========================================================================
def _thumb_b64(img, size=240):
    t = img.copy()
    t.thumbnail((size, size))
    buf = io.BytesIO()
    t.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _bars(preds, n=5):
    rows = ""
    for i, (lab, conf) in enumerate(preds[:n]):
        rows += (
            f'<div class="vr-row"><div class="vr-lab" title="{html.escape(lab)}">'
            f'{html.escape(lab)}</div>'
            f'<div class="vr-track"><div class="vr-bar" '
            f'style="width:{max(conf, 2):.1f}%;animation-delay:{i * .08:.2f}s"></div></div>'
            f'<div class="vr-pct">{conf:.1f}%</div></div>'
        )
    return rows


def _card(img, name, preds, secs=None, filename=None, mini=False):
    c, d = MODEL_COLORS[name]
    time_html = f'<span class="vr-time">{secs * 1000:.0f} ms</span>' if secs is not None else ""
    fname = f'<div class="vr-name">{html.escape(filename)}</div>' if filename else ""
    head = (f'<div class="vr-head"><span class="vr-badge">{name}</span>{time_html}</div>'
            if not mini else "")
    return (
        f'<div class="vr-card{" vr-mini" if mini else ""}" style="--c:{c};--d:{d}">'
        f'{head}{fname}'
        f'<div class="vr-body"><img class="vr-thumb" alt="input image" '
        f'src="data:image/jpeg;base64,{_thumb_b64(img)}">'
        f'<div class="vr-bars">{_bars(preds, 3 if mini else 5)}</div></div>'
        f'<div class="vr-cap">{html.escape(make_caption(preds))}</div></div>'
    )


def _msg(text):
    return f'<div class="vr-msg">{html.escape(text)}</div>'


def _err(text):
    return f'<div class="vr-err">{html.escape(text)}</div>'


# ===========================================================================
# tab callbacks
# ===========================================================================
def ui_single(image, model_name):
    if image is None:
        return _msg("Upload an image to begin.")
    img = _open(image)
    try:
        preds, secs = predict(img, model_name)
    except TFNotAvailable as e:
        return _err(str(e))
    except Exception as e:
        return _err(f"{model_name} failed: {e}")
    return f'<div class="vr-grid">{_card(img, model_name, preds, secs)}</div>'


def ui_compare(image):
    if image is None:
        return _msg("Upload an image to compare the three models.")
    img = _open(image)
    cards, tops = "", {}
    try:
        for name in CLF.available_models():
            preds, secs = predict(img, name)
            tops[name] = preds[0][0]
            cards += _card(img, name, preds, secs)
    except TFNotAvailable as e:
        return _err(str(e))
    if not tops:
        return _err("No models produced a result.")
    if len(tops) > 1 and len(set(tops.values())) == 1:
        banner = f'<div class="vr-agree">All three models agree: ' \
                 f'{html.escape(next(iter(tops.values())))}</div>'
    elif len(tops) > 1:
        parts = ", ".join(f"{k} says {v}" for k, v in tops.items())
        banner = f'<div class="vr-split">The models disagree. {html.escape(parts)}</div>'
    else:
        banner = ""
    return banner + f'<div class="vr-grid">{cards}</div>'


def ui_folder(files, folder, model_name):
    paths = []
    if folder and str(folder).strip():
        folder = str(folder).strip()
        if not os.path.isdir(folder):
            return _err(f"Folder not found: {folder}")
        paths = list_images(folder)
    elif files:
        paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]
    if not paths:
        return _msg("Upload several images, or type a folder path on this machine.")
    cards = ""
    try:
        _reg = CLF._reg()
        ctor, prep, decode = _reg[model_name]
        with CLF.lock:
            if model_name not in CLF.models:
                CLF.models[model_name] = ctor(weights="imagenet")
            for i in range(0, len(paths), 16):
                chunk = paths[i:i + 16]
                imgs = [_open(p) for p in chunk]
                x = np.concatenate([
                    prep(np.asarray(im.resize((224, 224)), dtype="float32")[None].copy())
                    for im in imgs
                ])
                decoded = decode(CLF.models[model_name].predict(x, verbose=0))
                for p, im, preds in zip(chunk, imgs, decoded):
                    cards += _card(im, model_name, preds,
                                   filename=os.path.basename(p), mini=True)
    except TFNotAvailable as e:
        return _err(str(e))
    return f'<div class="vr-grid">{cards}</div>'


# ===========================================================================
# layout
# ===========================================================================
def build_ui(share=False):
    import gradio as gr

    names = CLF.available_models()
    with gr.Blocks(title="Vision Reader") as demo:

        gr.HTML(HERO)

        with gr.Tabs():
            with gr.Tab("Single image"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img1 = gr.Image(type="pil", label="Image (jpg / png / webp)", height=320)
                        mdl = gr.Dropdown(names, value="VGG16", label="Model")
                        b1 = gr.Button("Read image", variant="primary", size="lg")
                    with gr.Column(scale=2):
                        out1 = gr.HTML(_msg("Upload an image to begin."))
                b1.click(ui_single, [img1, mdl], out1)

            with gr.Tab("Compare 3 models"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img2 = gr.Image(type="pil", label="Image (jpg / png / webp)", height=320)
                        b2 = gr.Button("Compare VGG16 / ResNet50 / EfficientNetB0",
                                       variant="primary", size="lg")
                    with gr.Column(scale=3):
                        out2 = gr.HTML(_msg("Upload an image to compare the three models."))
                b2.click(ui_compare, img2, out2)

            with gr.Tab("Batch / folder"):
                with gr.Row():
                    with gr.Column(scale=1):
                        files = gr.File(file_count="multiple", file_types=["image"],
                                        label="Upload several images")
                        folder = gr.Textbox(label="…or a folder path on this machine",
                                            placeholder="C:/photos  or  ./datasets/photos")
                        mdl3 = gr.Dropdown(names, value="EfficientNetB0", label="Model")
                        b3 = gr.Button("Read all", variant="primary", size="lg")
                    with gr.Column(scale=3):
                        out3 = gr.HTML(_msg("Upload several images, or type a folder path."))
                b3.click(ui_folder, [files, folder, mdl3], out3)

        gr.Markdown(
            '<p style="text-align:center;color:#6b7399;font-size:12px;margin:18px 0 8px">'
            'First use of each model downloads its ImageNet weights once '
            '(VGG16 ≈ 528 MB, ResNet50 ≈ 98 MB, EfficientNetB0 ≈ 29 MB), '
            'cached in <code>~/.keras</code>.</p>'
        )

    if share:
        demo.launch(share=True, inbrowser=True, css=WEB_CSS)
    else:
        demo.launch(inbrowser=True, css=WEB_CSS)
    return demo


def launch_ui(share=False):
    return build_ui(share=share)


# ===========================================================================
# CLI
# ===========================================================================
def _cli():
    import argparse
    ap = argparse.ArgumentParser(
        description="Read images with pretrained VGG16 / ResNet50 / EfficientNetB0")
    ap.add_argument("image", nargs="?", help="path to a jpg/png image")
    ap.add_argument("--folder", help="read every image in this folder")
    ap.add_argument("--model", default="VGG16", choices=CLF.available_models())
    ap.add_argument("--compare", action="store_true",
                    help="compare all three models on one image")
    ap.add_argument("--share", action="store_true",
                    help="publish a public Gradio link for the web UI")
    args = ap.parse_args()

    if args.folder:
        _run_folder_cli(args)
    elif args.image and args.compare:
        _run_compare_cli(args)
    elif args.image:
        _run_single_cli(args)
    else:
        build_ui(share=args.share)


def _run_single_cli(args):
    preds, secs = predict(args.image, args.model)
    print(f"\n[{args.model}] {os.path.basename(args.image)}  ({secs * 1000:.0f} ms)")
    for i, (label, conf) in enumerate(preds, 1):
        print(f"  {i}. {label:<28} {conf:6.2f}%")
    print("  ->", make_caption(preds))
    import matplotlib
    matplotlib.use("Agg" if os.environ.get("VR_HEADLESS") else "TkAgg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4),
                                gridspec_kw={"width_ratios": [1, 1.3]})
    a.imshow(_open(args.image))
    a.axis("off")
    a.set_title(make_caption(preds), fontsize=10)
    labels = [p[0] for p in preds][::-1]
    confs = [p[1] for p in preds][::-1]
    b.barh(labels, confs, color=MODEL_COLORS[args.model][0])
    b.set_xlim(0, 100)
    b.set_xlabel("Confidence (%)")
    b.set_title(f"{args.model} Top 5")
    plt.tight_layout()
    plt.show()


def _run_compare_cli(args):
    for name in CLF.available_models():
        preds, secs = predict(args.image, name)
        print(f"\n[{name}] {os.path.basename(args.image)}  ({secs * 1000:.0f} ms)")
        for i, (label, conf) in enumerate(preds, 1):
            print(f"  {i}. {label:<28} {conf:6.2f}%")
        print("  ->", make_caption(preds))


def _run_folder_cli(args):
    from analysis import list_images as _li
    paths = _li(args.folder)
    if not paths:
        print(f"No images found in {args.folder}")
        return
    for p in paths:
        preds, _ = predict(p, args.model)
        print(f"{os.path.basename(p):<30} {make_caption(preds)}")


if __name__ == "__main__":
    _cli()
