"""Vision Studio — desktop app (Tkinter).

A light, professional image-analysis studio with real 3D depth:
  * extruded 3D panels (colored bottom edge + soft drop shadow)
  * 3D push buttons that physically depress 4px on click
  * an image viewer that floats above the page with a blurred drop shadow
  * glowing per-model confidence bars
  * an isometric 3D cube logo in the header

Run with:  python src/vision_desktop.py      (or via src/main.py --ui desktop)
"""

import csv
import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageDraw, ImageFilter, ImageTk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis import IMG_EXTS, analyze_image, color_name, hex_of  # noqa: E402
from caption import make_caption  # noqa: E402
from classifier import Classifier, TFNotAvailable  # noqa: E402
from theme import (APP_BG, CARD, FONT, INK, LINE, MODEL_COLORS,  # noqa: E402
                   SHADOW, TRACK, MUTED, VIOLET)

MODEL_LIST = ["VGG16", "ResNet50", "EfficientNetB0"]


# ===========================================================================
# 3D drawing primitives
# ===========================================================================
def rrect(c, x0, y0, x1, y1, r, **kw):
    """A rounded-rect polygon (smooth)."""
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return c.create_polygon(pts, smooth=True, **kw)


def lighten(hexcol, amt=0.14):
    r, g, b = (int(hexcol[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(int(v + (255 - v) * amt) for v in (r, g, b))


def panel(parent, pad=8, accent=None):
    """A white 3D card: shadow layer + colored bottom edge + raised face.

    Returns (outer, inner). If ``accent`` is a (main, dark) color the card's
    bottom edge is tinted that color, otherwise a neutral gray.
    """
    outer = tk.Frame(parent, bg=parent.cget("bg"))
    shadow = tk.Frame(outer, bg=SHADOW)
    if accent:
        dark = accent[1]
        edge = tk.Frame(shadow, bg=dark, height=6)
        inner = tk.Frame(shadow, bg=CARD, highlightbackground=LINE,
                        highlightthickness=1)
        inner.pack(fill="both", expand=True, padx=(0, 3), pady=(0, 4))
        edge.pack(side="bottom", fill="x")
        shadow.pack(fill="both", expand=True)
        outer.configure(padx=pad, pady=pad)
        return outer, inner
    inner = tk.Frame(shadow, bg=CARD, highlightbackground=LINE, highlightthickness=1)
    inner.pack(fill="both", expand=True, padx=(0, 3), pady=(0, 4))
    shadow.pack(fill="both", expand=True)
    outer.configure(padx=pad, pady=pad)
    return outer, inner


class Btn3D(tk.Canvas):
    """A raised 3D button that physically depresses on click (4px travel)."""

    def __init__(self, master, text, command, colors, w=150, h=40, bg=APP_BG):
        super().__init__(master, width=w, height=h + 8, bg=bg,
                         highlightthickness=0, cursor="hand2")
        self.cmd, self.colors, self.w, self.h, self.text = command, colors, w, h, text
        self.down = self.hover = False
        self._draw()
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Enter>", lambda e: self._set(hover=True))
        self.bind("<Leave>", lambda e: self._set(hover=False, down=False))

    def _set(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        self._draw()

    def _draw(self):
        self.delete("all")
        off = 4 if self.down else 0
        main, dark = self.colors
        # 3D extrusion (dark layer) + top face
        rrect(self, 2, 6, self.w - 2, self.h + 7, 10, fill=dark, outline="")
        rrect(self, 2, 2 + off, self.w - 2, self.h + off, 10,
              fill=lighten(main) if self.hover else main, outline="")
        # top highlight for a glossier look
        rrect(self, 5, 5 + off, self.w - 5, 12 + off, 6,
              fill=lighten(main, 0.34), outline="")
        self.create_text(self.w / 2, (self.h + 6) / 2 + off, text=self.text,
                         fill="white", font=(FONT, 10, "bold"))

    def _press(self, _):
        self._set(down=True)

    def _release(self, e):
        was = self.down
        self._set(down=False)
        if was and 0 <= e.x <= self.w and 0 <= e.y <= self.h + 8:
            self.cmd()


class Viewer(tk.Canvas):
    """Image viewer that floats the picture above the page with a drop shadow."""

    def __init__(self, master):
        super().__init__(master, width=10, height=10, bg="white",
                          highlightthickness=0)
        self.src = None
        self.photo = None
        self._after = None
        self._text = ""
        self.bind("<Configure>", self._schedule)
        self._placeholder()

    def _placeholder(self, text=None):
        self.delete("all")
        w, h = max(self.winfo_width(), 200), max(self.winfo_height(), 200)
        self.create_text(w / 2, h / 2, text=text or "Open an image to begin",
                         fill=MUTED, font=(FONT, 13))

    def set(self, img, text=None):
        if img is not None:
            img = img.copy()
            img.thumbnail((1800, 1800))
        self.src = img
        self._text = text or ""
        self._render()

    def _schedule(self, _=None):
        if self._after:
            self.after_cancel(self._after)
        self._after = self.after(50, self._render)

    def _render(self):
        self._after = None
        if self.src is None:
            self._placeholder(self._text)
            return
        w, h = self.winfo_width(), self.winfo_height()
        if w < 60 or h < 60:
            return
        pad = 26
        im = self.src.copy()
        im.thumbnail((max(w - 2 * pad, 40), max(h - 2 * pad, 40)), Image.LANCZOS)
        canvas = Image.new("RGB", (im.width + 2 * pad, im.height + 2 * pad),
                           (255, 255, 255))
        sh = Image.new("L", canvas.size, 0)
        ImageDraw.Draw(sh).rectangle(
            [pad + 8, pad + 14, pad + im.width + 8, pad + im.height + 14], fill=140)
        sh = sh.filter(ImageFilter.GaussianBlur(13))
        canvas.paste(Image.new("RGB", canvas.size, (50, 66, 150)), (0, 0), sh)
        canvas.paste(im, (pad, pad))
        ImageDraw.Draw(canvas).rectangle(
            [pad - 1, pad - 1, pad + im.width, pad + im.height],
            outline=(255, 255, 255), width=2)
        self.photo = ImageTk.PhotoImage(canvas)
        self.delete("all")
        self.create_image(w / 2, h / 2, image=self.photo)


# ===========================================================================
# the app
# ===========================================================================
class App:
    def __init__(self, start_path=None):
        try:  # crisp text on Windows high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
        self.root = tk.Tk()
        self.root.title("Vision Studio")
        self.root.geometry("1380x820")
        self.root.minsize(1100, 680)
        self.root.configure(bg=APP_BG)

        self.clf = Classifier()
        self.q = queue.Queue()
        self.job = 0
        self.path = None
        self.folder_paths = []
        self.result = None
        self.preds = {}
        self.pmsg = ""

        self._style()
        self._header()
        self._toolbar()
        self._main()
        self._statusbar()

        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.after(80, self._poll)
        if start_path and os.path.isfile(start_path):
            self.root.after(200, lambda: self.load_path(start_path))

    # ---------------------------------------------------------------- styling
    def _style(self):
        s = ttk.Style()
        try:
            s.theme_use("clam")
        except Exception:
            pass
        s.configure("TNotebook", background=APP_BG, borderwidth=0)
        s.configure("TNotebook.Tab", padding=(16, 8), font=(FONT, 10, "bold"),
                    background="#e4e9fb", foreground=MUTED, borderwidth=0)
        s.map("TNotebook.Tab", background=[("selected", "white")],
              foreground=[("selected", MODEL_COLORS["VGG16"][1])])
        s.configure("Treeview", rowheight=28, font=(FONT, 10), background="white",
                    fieldbackground="white", foreground=INK, borderwidth=0)
        s.configure("TCombobox", padding=5, font=(FONT, 10))
        s.configure("Horizontal.TProgressbar", troughcolor="#e4e9fb",
                    background=MODEL_COLORS["VGG16"][0], borderwidth=0)

    # ----------------------------------------------------------------- header
    def _header(self):
        c = tk.Canvas(self.root, height=92, bg=APP_BG, highlightthickness=0)
        c.pack(fill="x")
        self.hdr = c
        c.bind("<Configure>", self._draw_header)

    def _draw_header(self, _=None):
        c = self.hdr
        c.delete("all")
        w = c.winfo_width()
        # soft vertical gradient background
        for i in range(92):
            t = i / 91
            r, g, b = (int(a + (z - a) * t)
                       for a, z in ((233, 250), (238, 245), (255, 255)))
            c.create_line(0, i, w, i, fill="#%02x%02x%02x" % (r, g, b))
        # isometric 3D cube logo (with a drop-shadow layer)
        cx, cy, s = 58, 46, 28
        k = 0.87 * s
        top = [cx, cy - s, cx + k, cy - s / 2, cx, cy, cx - k, cy - s / 2]
        right = [cx + k, cy - s / 2, cx + k, cy + s / 2, cx, cy + s, cx, cy]
        left = [cx - k, cy - s / 2, cx, cy, cx, cy + s, cx - k, cy + s / 2]
        # soft drop shadow of the cube
        c.create_polygon([p + 6 for p in left], fill="#c2ccee", outline="")
        c.create_polygon(left, fill="#4f6bed", outline="white")
        c.create_polygon(right, fill="#2f45b8", outline="white")
        c.create_polygon(top, fill="#a9b8ff", outline="white")
        # 3D extruded wordmark (layered shadows)
        for off, col in [(5, "#aebafc"), (4, "#9cabf9"), (3, "#8a9bf5"),
                         (2, "#788bf1"), (1, "#677be9")]:
            c.create_text(108 + off, 36 + off, text="Vision Studio",
                          anchor="w", fill=col, font=(FONT, 28, "bold"))
        c.create_text(108, 36, text="Vision Studio", anchor="w",
                      fill="#2f45b8", font=(FONT, 28, "bold"))
        c.create_text(110, 72, anchor="w", fill=MUTED, font=(FONT, 10),
                      text="Pretrained VGG16, ResNet50 and EfficientNetB0. "
                           "Reads objects, edges, resolution, quality and color.")

    # ---------------------------------------------------------------- toolbar
    def _toolbar(self):
        bar = tk.Frame(self.root, bg=APP_BG)
        bar.pack(fill="x", padx=16, pady=(6, 8))
        Btn3D(bar, "Open image", self.open_image, MODEL_COLORS["VGG16"]).pack(
            side="left", padx=(0, 12))
        Btn3D(bar, "Open folder", self.open_folder, MODEL_COLORS["ResNet50"]).pack(
            side="left", padx=(0, 20))
        tk.Label(bar, text="Model", bg=APP_BG, fg=MUTED, font=(FONT, 10)).pack(
            side="left", padx=(0, 6))
        self.model_var = tk.StringVar(value="All 3 models")
        cb = ttk.Combobox(bar, textvariable=self.model_var, state="readonly",
                          width=16, values=["All 3 models"] + MODEL_LIST)
        cb.pack(side="left", padx=(0, 14))
        cb.bind("<<ComboboxSelected>>", lambda e: self.analyze())
        Btn3D(bar, "Analyze", self.analyze, MODEL_COLORS["EfficientNetB0"],
               w=110).pack(side="left", padx=(0, 12))
        Btn3D(bar, "Export CSV", self.export_report, VIOLET, w=120).pack(side="left")

    # ------------------------------------------------------------------- main
    def _main(self):
        pw = tk.PanedWindow(self.root, orient="horizontal", bg=APP_BG,
                            sashwidth=8, bd=0, sashrelief="flat")
        pw.pack(fill="both", expand=True, padx=14, pady=(0, 6))

        # left: folder list
        lo, li = panel(pw, accent=MODEL_COLORS["VGG16"])
        tk.Label(li, text="Images", bg="white", fg=INK,
                 font=(FONT, 11, "bold")).pack(anchor="w", padx=12, pady=(10, 4))
        lf = tk.Frame(li, bg="white")
        lf.pack(fill="both", expand=True, padx=6, pady=(0, 8))
        sb = ttk.Scrollbar(lf, orient="vertical")
        self.listbox = tk.Listbox(
            lf, bd=0, highlightthickness=0, activestyle="none", font=(FONT, 10),
            selectbackground="#dfe6ff", selectforeground=INK, fg=INK,
            yscrollcommand=sb.set)
        sb.config(command=self.listbox.yview)
        sb.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_list)
        pw.add(lo, minsize=150, width=190)

        # middle: viewer
        mo, mi = panel(pw, accent=MODEL_COLORS["ResNet50"])
        seg = tk.Frame(mi, bg="white")
        seg.pack(fill="x", padx=10, pady=(10, 4))
        self.view_var = tk.StringVar(value="Original")
        self.seg_buttons = {}
        for name in ["Original", "Sobel edges", "Canny edges", "Grayscale",
                     "Histogram"]:
            rb = tk.Radiobutton(seg, text=name, value=name, variable=self.view_var,
                                indicatoron=0, command=self._show_view,
                                font=(FONT, 9, "bold"), bg="#f1f4ff", fg=INK,
                                selectcolor="#cfd9ff", activebackground="#e4eaff",
                                relief="raised", bd=2, padx=10, pady=4, cursor="hand2")
            rb.pack(side="left", padx=2)
            self.seg_buttons[name] = rb
        self.viewer = Viewer(mi)
        self.viewer.pack(fill="both", expand=True, padx=2, pady=2)
        self.summary = tk.Label(mi, text="", bg="white", fg=MUTED,
                                font=(FONT, 9), wraplength=620, justify="center")
        self.summary.pack(fill="x", padx=8, pady=(2, 8))
        pw.add(mo, minsize=380, width=650)

        # right: tabs
        ro = tk.Frame(pw, bg=APP_BG)
        nb = ttk.Notebook(ro)
        nb.pack(fill="both", expand=True)
        pw.add(ro, minsize=360, width=470)

        pf = tk.Frame(nb, bg="white")
        self.pcanvas = tk.Canvas(pf, bg="#f7f9ff", highlightthickness=0)
        psb = ttk.Scrollbar(pf, orient="vertical", command=self.pcanvas.yview)
        self.pcanvas.configure(yscrollcommand=psb.set)
        psb.pack(side="right", fill="y")
        self.pcanvas.pack(fill="both", expand=True)
        self.pcanvas.bind("<Configure>", lambda e: self._draw_preds())
        self.pcanvas.bind("<MouseWheel>",
                          lambda e: self.pcanvas.yview_scroll(int(-e.delta / 120), "units"))
        self.pcanvas.bind("<Button-4>", lambda e: self.pcanvas.yview_scroll(-1, "units"))
        self.pcanvas.bind("<Button-5>", lambda e: self.pcanvas.yview_scroll(1, "units"))
        nb.add(pf, text="What is it")

        df = tk.Frame(nb, bg="white")
        self.tree = ttk.Treeview(df, columns=("v",), show="tree", selectmode="browse")
        self.tree.column("#0", width=190, stretch=False)
        self.tree.column("v", width=300, stretch=True)
        self.tree.tag_configure("sec", background="#e8edff", font=(FONT, 10, "bold"),
                                foreground=MODEL_COLORS["VGG16"][1])
        tsb = ttk.Scrollbar(df, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tsb.set)
        tsb.pack(side="right", fill="y")
        self.tree.pack(fill="both", expand=True)
        nb.add(df, text="Details")

        cf = tk.Frame(nb, bg="white")
        self.ccanvas = tk.Canvas(cf, bg="white", highlightthickness=0)
        self.ccanvas.pack(fill="both", expand=True)
        self.ccanvas.bind("<Configure>", lambda e: self._draw_colors())
        nb.add(cf, text="Colors")
        self._draw_preds()

    def _statusbar(self):
        bar = tk.Frame(self.root, bg=APP_BG)
        bar.pack(fill="x", padx=16, pady=(0, 8))
        self.status = tk.Label(bar, text="Ready. Open an image to start.",
                               bg=APP_BG, fg=MUTED, font=(FONT, 9), anchor="w")
        self.status.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=160)
        self.progress.pack(side="right")

    # -------------------------------------------------------------- file flow
    def open_image(self):
        p = filedialog.askopenfilename(
            title="Choose an image",
            filetypes=[("Images", " ".join("*" + e for e in IMG_EXTS)),
                       ("All files", "*.*")])
        if p:
            self.load_path(p)

    def open_folder(self):
        d = filedialog.askdirectory(title="Choose a folder of images")
        if not d:
            return
        paths = sorted(os.path.join(d, f) for f in os.listdir(d)
                      if f.lower().endswith(IMG_EXTS))
        if not paths:
            messagebox.showinfo("Vision Studio", "No images found in that folder.")
            return
        self.folder_paths = paths
        self.listbox.delete(0, "end")
        for p in paths:
            self.listbox.insert("end", os.path.basename(p))
        self.listbox.selection_set(0)
        self.load_path(paths[0])

    def _on_list(self, _):
        sel = self.listbox.curselection()
        if sel and self.folder_paths:
            p = self.folder_paths[sel[0]]
            if p != self.path:
                self.load_path(p)

    def load_path(self, path):
        self.path = path
        self.analyze()

    def _selected_models(self):
        v = self.model_var.get()
        order = ["EfficientNetB0", "ResNet50", "VGG16"]  # fastest first
        return order if v.startswith("All") else [v]

    def analyze(self):
        if not self.path:
            return self.open_image()
        self.job += 1
        self.preds, self.pmsg = {}, ""
        self._draw_preds()
        self.progress.start(12)
        self.status.config(text=f"Analyzing {os.path.basename(self.path)} ...")
        threading.Thread(target=self._work,
                         args=(self.path, self.job, self._selected_models()),
                         daemon=True).start()

    # ---------------------------------------------------------------- workers
    def _work(self, path, job, models):
        try:
            res = analyze_image(path)
        except Exception as e:
            self.q.put(("error", job, f"Could not read this image: {e}"))
            self.q.put(("done", job, None))
            return
        self.q.put(("analysis", job, res))
        for m in models:
            if job != self.job:
                return
            self.q.put(("status", job, f"Running {m} (first run downloads weights)..."))
            try:
                preds, secs = self.clf.predict(res["rgb"], m)
                self.q.put(("pred", job, (m, preds, secs)))
            except TFNotAvailable as e:
                self.q.put(("pmsg", job, str(e)))
                break
            except Exception as e:
                self.q.put(("pmsg", job, f"{m} failed: {e}"))
                break
        self.q.put(("done", job, None))

    def _poll(self):
        try:
            while True:
                kind, job, data = self.q.get_nowait()
                if kind == "status":
                    self.status.config(text=data)
                    continue
                if kind == "export_done":
                    self.progress.stop()
                    self.status.config(text=data)
                    messagebox.showinfo("Vision Studio", data)
                    continue
                if job != self.job:
                    continue
                if kind == "analysis":
                    self._show_analysis(data)
                elif kind == "pred":
                    self.preds[data[0]] = (data[1], data[2])
                    self._draw_preds()
                elif kind in ("pmsg", "error"):
                    self.pmsg = data
                    self._draw_preds()
                    self.status.config(text=data.splitlines()[0])
                elif kind == "done":
                    self.progress.stop()
                    if self.result:
                        self.status.config(
                            text=f"Done. {os.path.basename(self.result['path'])}")
        except queue.Empty:
            pass
        self.root.after(80, self._poll)

    # ------------------------------------------------------------- rendering
    def _show_analysis(self, res):
        self.result = res
        self.summary.config(text=res["short"])
        self.seg_buttons["Canny edges"].config(text=res["canny_title"],
                                                value=res["canny_title"])
        if self.view_var.get() not in res["views"]:
            self.view_var.set("Original")
        self._show_view()
        self.tree.delete(*self.tree.get_children())
        for title, rows in res["sections"]:
            parent = self.tree.insert("", "end", text=title, open=True,
                                      tags=("sec",))
            for k, v in rows:
                self.tree.insert(parent, "end", text=k, values=(v,))
        self._draw_colors()

    def _show_view(self):
        if self.result:
            self.viewer.set(self.result["views"].get(self.view_var.get(),
                                                     self.result["views"]["Original"]))

    def _draw_preds(self):
        c = self.pcanvas
        c.delete("all")
        W = max(c.winfo_width(), 340)
        x0, x1 = 14, W - 20
        y = 14

        def line(text, size=10, bold=False, color=INK, x=x0, anchor="nw",
                 width=None):
            return c.create_text(x, y, text=text, anchor=anchor, fill=color,
                                 width=width,
                                 font=(FONT, size, "bold" if bold else "normal"))

        if not self.preds:
            msg = self.pmsg or ("Predictions appear here.\nThe first run of each model "
                                "downloads its ImageNet weights, so it can take a minute."
                                if self.path else
                                "Open an image to see what the models think it contains.")
            line(msg, 11, color=MUTED if not self.pmsg else "#b4361a",
                 width=x1 - x0)
            c.configure(scrollregion=(0, 0, W, 300))
            return

        tops = {m: p[0][0] for m, p in self.preds.items()}
        if len(tops) > 1:
            same = len(set(tops.values())) == 1
            txt = (f"All {len(tops)} models agree: {next(iter(tops.values()))}" if same
                   else "Models disagree: " + ", ".join(
                       f"{m} says {t}" for m, t in tops.items()))
            fill, edge, ink = ("#eafaf3", "#bfeed7", "#0e6b45") if same else \
                              ("#fff6e8", "#ffdca8", "#8a5200")
            tid = line(txt, 10, True, ink, x=x0 + 12, width=x1 - x0 - 24)
            y0, y1 = c.bbox(tid)[1] - 10, c.bbox(tid)[3] + 10
            rrect(c, x0, y0 + 4, x1, y1 + 4, 12, fill=edge, outline="")
            r = rrect(c, x0, y0, x1, y1, 12, fill=fill, outline="")
            c.tag_lower(r, tid)
            y = y1 + 22

        for name in MODEL_LIST:
            if name not in self.preds:
                continue
            preds, secs = self.preds[name]
            main, dark = MODEL_COLORS[name]
            top = y
            y += 12
            # 3D badge
            b = rrect(c, x0 + 12, y, x0 + 12 + 118, y + 26, 13, fill=main,
                      outline="")
            rrect(c, x0 + 12, y + 3, x0 + 12 + 118, y + 29, 13, fill=dark,
                  outline="")
            c.tag_raise(b)
            c.create_text(x0 + 12 + 59, y + 13, text=name, fill="white",
                          font=(FONT, 10, "bold"))
            c.create_text(x1 - 12, y + 13, text=f"{secs * 1000:.0f} ms",
                          anchor="e", fill=MUTED, font=(FONT, 9))
            y += 44
            lab_w = 128
            bx0, bx1 = x0 + 12 + lab_w, x1 - 62
            for lab, conf in preds:
                c.create_text(x0 + 12, y + 8, text=lab, anchor="w", fill=INK,
                              font=(FONT, 9))
                # glow track + 3D bar
                rrect(c, bx0, y + 1, bx1, y + 15, 7, fill=TRACK, outline="")
                bw = max((bx1 - bx0) * conf / 100, 10)
                rrect(c, bx0, y + 5, bx0 + bw, y + 19, 7, fill=dark, outline="")
                rrect(c, bx0, y + 1, bx0 + bw, y + 15, 7, fill=main, outline="")
                c.create_text(x1 - 12, y + 8, text=f"{conf:.1f}%", anchor="e",
                              fill=INK, font=(FONT, 9, "bold"))
                y += 28
            y += 4
            tid = c.create_text(x0 + 24, y + 10, text=make_caption(preds),
                                anchor="nw", fill=INK, width=x1 - x0 - 48,
                                font=(FONT, 10, "bold"))
            bb = c.bbox(tid)
            cap = rrect(c, x0 + 12, bb[1] - 8, x1 - 12, bb[3] + 8, 10,
                        fill="#eef2ff", outline="")
            c.tag_lower(cap, tid)
            y = bb[3] + 22
            # 3D card frame (shadow layer + face)
            sh = rrect(c, x0 + 3, top + 5, x1 + 3, y + 5, 16, fill=SHADOW,
                       outline="")
            face = rrect(c, x0, top, x1, y, 16, fill="white", outline=LINE)
            c.tag_lower(face)
            c.tag_lower(sh)
            y += 18
        c.configure(scrollregion=(0, 0, W, y))

    def _draw_colors(self):
        c = self.ccanvas
        c.delete("all")
        if not self.result:
            c.create_text(20, 20, anchor="nw", text="Open an image to see its palette.",
                          fill=MUTED, font=(FONT, 11))
            return
        W = max(c.winfo_width(), 320)
        c.create_text(18, 16, anchor="nw", text="Dominant colors", fill=INK,
                      font=(FONT, 12, "bold"))
        y = 52
        for pct, rgb in self.result["palette"]:
            col = hex_of(rgb)
            # 3D color block (shadow + face)
            rrect(c, 18, y + 4, 78, y + 54, 10, fill=SHADOW, outline="")
            rrect(c, 18, y, 78, y + 50, 10, fill=col, outline="white")
            c.create_text(92, y + 12, anchor="w",
                         text=f"{color_name(rgb)}   {col}", fill=INK,
                         font=(FONT, 10, "bold"))
            bw = (W - 110 - 60) * pct / 100
            rrect(c, 92, y + 28, 92 + (W - 170), y + 40, 6, fill=TRACK,
                  outline="")
            rrect(c, 92, y + 28, 92 + max(bw, 8), y + 40, 6, fill=col,
                  outline="")
            c.create_text(W - 16, y + 34, anchor="e", text=f"{pct:.0f}%",
                          fill=INK, font=(FONT, 10, "bold"))
            y += 68

    # ----------------------------------------------------------------- export
    def export_report(self):
        paths = self.folder_paths or ([self.path] if self.path else [])
        if not paths:
            return messagebox.showinfo("Vision Studio",
                                       "Open an image or folder first.")
        out = filedialog.asksaveasfilename(defaultextension=".csv",
                                           initialfile="vision_report.csv",
                                           filetypes=[("CSV", "*.csv")])
        if not out:
            return
        self.progress.start(12)
        threading.Thread(target=self._export_work,
                         args=(list(paths), out, self._selected_models()),
                         daemon=True).start()

    def _export_work(self, paths, out, models):
        rows = []
        try:
            for i, p in enumerate(paths, 1):
                self.q.put(("status", 0, f"Exporting {i}/{len(paths)}: "
                                          f"{os.path.basename(p)}"))
                try:
                    res = analyze_image(p, light=True)
                except Exception as e:
                    rows.append({"file": os.path.basename(p), "error": str(e)})
                    continue
                row = dict(res["flat"])
                for m in models:
                    try:
                        from PIL import Image as _I
                        img = _I.open(p).convert("RGB")
                        preds, _ = self.clf.predict(img, m)
                        row[f"{m}_top1"], row[f"{m}_conf_pct"] = \
                            preds[0][0], round(preds[0][1], 1)
                        row[f"{m}_caption"] = make_caption(preds)
                    except Exception as e:
                        row[f"{m}_top1"] = f"unavailable ({type(e).__name__})"
                        break
                rows.append(row)
            cols = []
            for r in rows:
                cols += [k for k in r if k not in cols]
            with open(out, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                w.writerows(rows)
            self.q.put(("export_done", 0, f"Saved {len(rows)} row(s) to {out}"))
        except Exception as e:
            self.q.put(("export_done", 0, f"Export failed: {e}"))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    App(sys.argv[1] if len(sys.argv) > 1 else None).run()
