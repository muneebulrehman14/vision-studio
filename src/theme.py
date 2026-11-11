"""Shared design system — the "3D depth" language used by BOTH front ends.

One light, professional theme with real depth:
  * extruded 3D panels (raised face + colored bottom edge + soft drop shadow)
  * a floating image viewer with a blurred drop shadow
  * subtle parallax / tilt on hover
  * soft color glow per model

The web app (Gradio) uses the CSS here; the desktop app (Tkinter) mirrors the
same palette and 3D primitives so the two feel like one product.
"""

# ---- per-model accents -----------------------------------------------------
INDIGO = ("#4f6bed", "#3549c0")     # primary / VGG16
TEAL = ("#0fa3a3", "#0a7272")       # ResNet50
ORANGE = ("#f2762e", "#c4560f")     # EfficientNetB0
VIOLET = ("#7a5af0", "#5a3dc4")     # actions / export

MODEL_COLORS = {
    "VGG16": INDIGO,
    "ResNet50": TEAL,
    "EfficientNetB0": ORANGE,
}

# semantic
OK = ("#1f9d63", "#0e6b45")
WARN = ("#f5a524", "#c07a0a")
ERROR = ("#e05252", "#a32e2e")

# ---- light palette (single source of truth) ---------------------------------
APP_BG = "#eef1fb"        # app canvas base (soft blue-lavender)
BG_DEEP = "#dfe4f5"       # page background gradient end
CARD = "#ffffff"
INK = "#1d2340"
MUTED = "#6b7399"
LINE = "#dbe2ff"
SHADOW = "#cdd6f6"        # 3D extrude / drop-shadow color
TRACK = "#e9edfd"
FONT = "Segoe UI"

# a flat dict mirror for convenience
LIGHT = {"bg": APP_BG, "deep": BG_DEEP, "card": CARD, "ink": INK, "muted": MUTED,
         "line": LINE, "shadow": SHADOW, "track": TRACK, "font": FONT}


# ===========================================================================
# WEB CSS — real 3D depth
# ===========================================================================
WEB_CSS = """
<style>
:root{
  --bg:#eef1fb; --bg-deep:#dfe4f5; --card:#fff;
  --ink:#1d2340; --muted:#6b7399; --line:#dbe2ff;
  --accent:#4f6bed; --accent-d:#3549c0;
  --teal:#0fa3a3; --orange:#f2762e; --violet:#7a5af0;
  --ok:#1f9d63; --warn:#f5a524;
}
.gradio-container{
  background:
    radial-gradient(1200px 500px at 20% -10%, #ffffff 0%, rgba(255,255,255,0) 60%),
    radial-gradient(1000px 500px at 100% 0%, #e6ecff 0%, rgba(230,236,255,0) 55%),
    linear-gradient(165deg, var(--bg) 0%, var(--bg-deep) 100%)!important;
  color:var(--ink);
  font-family:"Segoe UI",Inter,system-ui,sans-serif;
}

/* ---- 3D extruded panel: face + colored bottom edge + drop shadow -------- */
.p3d{--c:#4f6bed;--d:#3549c0;background:var(--card);border:1px solid var(--line);
  border-radius:20px;position:relative;
  box-shadow:0 1px 2px rgba(30,40,90,.05),0 16px 34px rgba(30,40,90,.14);
  transition:transform .35s cubic-bezier(.2,.8,.2,1),box-shadow .35s ease}
.p3d::before{content:"";position:absolute;left:6px;right:6px;bottom:-7px;height:14px;
  border-radius:0 0 18px 18px;background:var(--d);
  filter:blur(6px);opacity:.45;z-index:-1;transition:opacity .35s}
.p3d:hover{transform:translateY(-6px)}
.p3d:hover::before{opacity:.75}

/* card with a colored top accent strip */
.vr-card{--c:#4f6bed;--d:#3549c0;background:var(--card);border:1px solid var(--line);
  border-radius:18px;padding:18px;position:relative;overflow:hidden;
  box-shadow:0 1px 2px rgba(30,40,90,.05),0 14px 30px rgba(30,40,90,.12);
  transition:transform .3s cubic-bezier(.2,.8,.2,1),box-shadow .3s ease}
.vr-card::before{content:"";position:absolute;inset:0 0 auto 0;height:5px;
  background:linear-gradient(90deg,var(--c),var(--d))}
.vr-card:hover{transform:translateY(-4px) scale(1.01);
  box-shadow:0 2px 4px rgba(30,40,90,.08),0 24px 46px rgba(30,40,90,.18)}

/* ---- 3D buttons --------------------------------------------------------- */
.vr-btn{--c:#4f6bed;--d:#3549c0;position:relative;color:#fff;font-weight:700;
  font-size:15px;padding:14px 26px;border-radius:14px;border:none;cursor:pointer;
  background:linear-gradient(180deg,var(--c),var(--c));
  box-shadow:0 6px 0 var(--d),0 14px 22px rgba(30,40,90,.28);
  transition:transform .12s ease,box-shadow .12s ease}
.vr-btn:hover{transform:translateY(-1px);
  box-shadow:0 7px 0 var(--d),0 18px 30px rgba(30,40,90,.34)}
.vr-btn:active{transform:translateY(5px);
  box-shadow:0 1px 0 var(--d),0 6px 10px rgba(30,40,90,.25)}

/* progress bars with a 3D fill + glow */
.vr-track{height:12px;background:var(--line);border-radius:9px;overflow:hidden;
  box-shadow:inset 0 2px 4px rgba(30,40,90,.10)}
.vr-bar{height:12px;border-radius:9px;transform-origin:left;
  background:linear-gradient(180deg,color-mix(in srgb,var(--c) 78%,#fff),var(--c));
  box-shadow:0 3px 6px color-mix(in srgb,var(--d) 60%,transparent);
  animation:vrgrow .9s cubic-bezier(.2,.8,.2,1) both}
@keyframes vrgrow{from{transform:scaleX(0)}to{transform:scaleX(1)}}

/* ---- hero --------------------------------------------------------------- */
.vr-hero{display:flex;align-items:center;gap:32px;padding:34px 16px 18px;flex-wrap:wrap}
.vr-cubewrap{width:150px;height:150px;perspective:820px;flex:none}
.vr-cube{width:84px;height:84px;position:relative;transform-style:preserve-3d;
  margin:33px auto;animation:vrspin 16s linear infinite}
.vr-cube div{position:absolute;width:84px;height:84px;display:flex;align-items:center;
  justify-content:center;font-size:32px;border:2px solid rgba(255,255,255,.6);
  background:linear-gradient(135deg,rgba(79,107,237,.95),rgba(15,163,163,.88));
  box-shadow:inset 0 0 26px rgba(255,255,255,.45)}
.vr-cube .f1{transform:translateZ(42px)} .vr-cube .f2{transform:rotateY(180deg) translateZ(42px)}
.vr-cube .f3{transform:rotateY(90deg) translateZ(42px)} .vr-cube .f4{transform:rotateY(-90deg) translateZ(42px)}
.vr-cube .f5{transform:rotateX(90deg) translateZ(42px)} .vr-cube .f6{transform:rotateX(-90deg) translateZ(42px)}
@keyframes vrspin{from{transform:rotateX(-22deg) rotateY(0)}to{transform:rotateX(-22deg) rotateY(360deg)}}
.vr-title{font-size:52px;font-weight:800;letter-spacing:-1.6px;margin:0;color:#2d3f9e;
  text-shadow:2px 3px 0 #cdd6f6,4px 6px 0 #b9c4f2,8px 12px 24px rgba(45,63,158,.35)}
.vr-sub{margin:10px 0 0;color:var(--muted);font-size:16px;max-width:560px;line-height:1.55}
.vr-chips{display:flex;gap:10px;margin-top:16px;flex-wrap:wrap}
.vr-chip{font-size:13px;font-weight:600;padding:7px 15px;border-radius:999px;
  border:1px solid var(--line);background:#fff;color:var(--ink);
  box-shadow:0 3px 0 #d5dcf6,0 6px 12px rgba(30,40,90,.10)}
.vr-chip b{color:var(--accent-d)}

/* grid of floating 3D cards */
.vr-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:22px;padding:20px 4px}

/* Gradio controls: give them the same depth language */
.gradio-container button.primary{background:linear-gradient(180deg,#5c76f3,#4f6bed)!important;
  border:none!important;color:#fff!important;font-weight:800!important;letter-spacing:.2px;
  border-radius:14px!important;
  box-shadow:0 5px 0 #3549c0,0 12px 20px rgba(30,40,90,.26)!important;
  transition:transform .12s ease,box-shadow .12s ease!important}
.gradio-container button.primary:hover{transform:translateY(-1px);
  box-shadow:0 6px 0 #3549c0,0 16px 26px rgba(30,40,90,.30)!important}
.gradio-container button.primary:active{transform:translateY(4px);
  box-shadow:0 1px 0 #3549c0,0 5px 10px rgba(30,40,90,.22)!important}
.gradio-container .block{border-radius:16px}
.gradio-container .form,.gradio-container input[type=text],
.gradio-container .wrap-inner{border-radius:12px!important}
.gradio-container .upload-container,.gradio-container .image-container{
  border-radius:16px!important;box-shadow:inset 0 0 0 1px var(--line),0 10px 22px rgba(30,40,90,.08)!important}
.gradio-container .tab-nav button{font-weight:700!important;font-size:15px!important}

/* model badge + time */
.vr-head{display:flex;justify-content:space-between;align-items:center;margin:6px 0 12px}
.vr-badge{background:linear-gradient(180deg,var(--c),var(--d));color:#fff;font-weight:700;
  font-size:13px;padding:6px 15px;border-radius:999px;
  box-shadow:0 4px 0 var(--d),0 8px 14px color-mix(in srgb,var(--d) 50%,transparent)}
.vr-time{color:var(--muted);font-size:12px;font-weight:700}
.vr-body{display:flex;gap:16px;align-items:flex-start}
.vr-thumb{width:118px;height:118px;object-fit:cover;border-radius:14px;flex:none;
  border:3px solid #fff;
  box-shadow:0 10px 22px rgba(30,40,90,.22);transition:transform .3s}
.vr-card:hover .vr-thumb{transform:perspective(500px) rotateY(6deg) scale(1.05)}
.vr-bars{flex:1;min-width:0}
.vr-row{display:grid;grid-template-columns:minmax(92px,1.15fr) 1.7fr 50px;gap:10px;
  align-items:center;margin:0 0 11px;font-size:12.5px}
.vr-lab{white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:600;
  color:#3a4270}
/* compact (3-up compare) cards: stack label above a full-width bar */
.vr-compact .vr-row{grid-template-columns:1fr auto;
  grid-template-areas:"lab pct" "bar bar";gap:5px 8px;margin-bottom:13px}
.vr-compact .vr-lab{grid-area:lab;font-size:13px}
.vr-compact .vr-track{grid-area:bar}
.vr-compact .vr-pct{grid-area:pct;font-size:13px}
.vr-compact .vr-thumb{width:104px;height:104px}
.vr-pct{text-align:right;font-weight:800;color:var(--ink)}
.vr-cap{margin-top:12px;padding:13px 15px;border-radius:14px;font-weight:600;
  line-height:1.4;font-size:14px;
  background:linear-gradient(135deg,#eef2ff,#fbfcff);border:1px solid var(--line);
  box-shadow:inset 0 1px 0 #fff,0 2px 6px rgba(30,40,90,.05)}
.vr-agree{margin:14px 4px 4px;padding:15px 18px;border-radius:16px;font-weight:700;
  background:#eafaf3;border:1px solid #bfeed7;color:var(--ok);
  box-shadow:0 5px 0 #bfeed7,0 10px 18px rgba(14,107,69,.14)}
.vr-split{margin:14px 4px 4px;padding:15px 18px;border-radius:16px;font-weight:700;
  background:#fff6e8;border:1px solid #ffdca8;color:#8a5200;
  box-shadow:0 5px 0 #ffdca8,0 10px 18px rgba(138,82,0,.14)}
.vr-name{font-size:12px;color:var(--muted);margin-bottom:8px;
  overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-weight:600}
.vr-mini .vr-body{flex-direction:column}
.vr-mini .vr-thumb{width:100%;height:150px}
.vr-mini .vr-bars{width:100%;margin-top:8px}
.vr-msg{padding:28px;border-radius:18px;background:#fff;border:1px dashed #c4cdf5;
  color:var(--muted);text-align:center;font-size:15px;line-height:1.6;
  box-shadow:0 10px 24px rgba(30,40,90,.08)}
.vr-err{padding:22px;border-radius:18px;background:#fdecec;border:1px solid #f5c6c6;
  color:#8a1e1e;font-size:15px;line-height:1.6;
  box-shadow:0 8px 18px rgba(163,46,46,.14)}
@media (prefers-reduced-motion:reduce){.vr-cube,.vr-bar{animation:none}
  .vr-card,.vr-btn,.vr-thumb{transition:none}}
</style>
"""

# a small hero HTML block shared across the web app
HERO = """
<div class="vr-hero">
  <div class="vr-cubewrap"><div class="vr-cube">
    <div class="f1">🐱</div><div class="f2">🚗</div><div class="f3">🍕</div>
    <div class="f4">✈️</div><div class="f5">🌼</div><div class="f6">⚽</div>
  </div></div>
  <div>
    <h1 class="vr-title">Vision Reader</h1>
    <p class="vr-sub">Drop in any photo and three ImageNet-trained models — VGG16,
    ResNet50 and EfficientNetB0 — tell you what they see across 1000 classes.
    No training, no datasets — instant inference.</p>
    <div class="vr-chips">
      <span class="vr-chip"><b>3</b> pretrained models</span>
      <span class="vr-chip"><b>1000</b> ImageNet classes</span>
      <span class="vr-chip"><b>0</b> training required</span>
    </div>
  </div>
</div>
"""
