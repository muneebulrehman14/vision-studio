<div align="center">

# 🚀 Vision Studio

**Professional Image Analysis with Pretrained ImageNet Models** — VGG16, ResNet50 & EfficientNetB0

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-ff6f00.svg?logo=tensorflow)
![Gradio](https://img.shields.io/badge/Gradio-4.x-orange.svg?logo=gradio)
![Tkinter](https://img.shields.io/badge/GUI-Tkinter-lightgrey.svg?logo=python)
![Status](https://img.shields.io/badge/status-production-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

<br>

<img src="https://capsule-render.vercel.app/api?type=waving&color=4f6bed&height=120&section=header" width="100%"/>

</div>

---

<div align="center">

### ✨ One Image. Three Pretrained Models. Zero Training.

</div>

<table>
  <tr>
    <td><b>🖼️ What it sees</b></td>
    <td>Object recognition across 1000 ImageNet classes with top-5 predictions per model</td>
  </tr>
  <tr>
    <td><b>🤖 3 Models</b></td>
    <td>VGG16 • ResNet50 • EfficientNetB0 — side-by-side comparison & consensus banner</td>
  </tr>
  <tr>
    <td><b>📐 Resolution</b></td>
    <td>Pixels, megapixels, aspect ratio, HD/4K class, print size, DPI</td>
  </tr>
  <tr>
    <td><b>🔍 Edges</b></td>
    <td>Sobel, Canny (optional), edge density, dominant direction</td>
  </tr>
  <tr>
    <td><b>✨ Quality</b></td>
    <td>Sharpness, brightness, contrast, entropy, clipping</td>
  </tr>
  <tr>
    <td><b>🎨 Color</b></td>
    <td>Dominant palette, average color, saturation, temperature</td>
  </tr>
  <tr>
    <td><b>📋 EXIF</b></td>
    <td>Camera make/model, exposure, ISO, focal length, GPS presence</td>
  </tr>
  <tr>
    <td><b>📊 CSV Export</b></td>
    <td>Batch export analysis reports for folders of images</td>
  </tr>
</table>

---

## 🎨 Screenshots / Preview

<div align="center">

### Web UI — Gradio Command Center

<img src="https://capsule-render.vercel.app/api?type=sliced&color=4f6bed&height=100&section=footer" width="100%"/>

| Single Image | Compare 3 Models | Batch / Folder |
|:---:|:---:|:---:|
| Floating 3D result card with per-model accent | Consensus banner (agree/disagree) | Card grid for multiple images |
| Glow-filled animated confidence bars | Side-by-side VGG16 / ResNet50 / EfficientNetB0 | Folder path or multi-upload |

</div>

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/vision-studio.git
cd vision-studio
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run

**Web UI (recommended):**
```bash
python src/vision_web.py
```
Opens at `http://127.0.0.1:7860` automatically.

**Desktop Studio:**
```bash
python src/vision_desktop.py
```

**Headless / Batch:**
```bash
python src/main.py photo.jpg --csv report.csv
python src/batch.py photos/ --csv report.csv --recursive
```

---

## 🧩 Project Structure

```
vision-studio/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── assets/                 # Sample images & UI previews
├── docs/                   # Documentation
│   └── api.md
├── src/
│   ├── __init__.py
│   ├── main.py             # Unified CLI entry point
│   ├── batch.py            # Headless folder → CSV runner
│   ├── theme.py            # Shared light 3D design tokens
│   ├── analysis.py         # Pixel-level core (edges, quality, color, EXIF)
│   ├── caption.py          # Top-k predictions → English sentence
│   ├── classifier.py       # Lazy TF classifiers (VGG16 / ResNet50 / EfficientNetB0)
│   ├── vision_web.py       # Gradio command-center web UI
│   └── vision_desktop.py   # Tkinter desktop studio
└── tests/
    └── test_core.py        # 23 TF-free tests
```

---

## 🛠️ Features in Detail

### 🖼️ Image Classification
Three pretrained ImageNet models analyze any photo and return top-5 predictions with confidence percentages and a plain-English caption.

### 🤖 Model Comparison
Upload one image and see all three models side-by-side. A smart banner shows whether they **agree** or **disagree**.

### 📊 Batch Processing
Analyze entire folders of images and export everything to a CSV report — resolution, edges, quality, color, EXIF, and model predictions.

### 💻 Two Interfaces
- **Web UI** — Gradio-powered command center with 3D cards, glowing bars, and animated hero
- **Desktop Studio** — Tkinter app with 3D panels, pressable buttons, floating image viewer, and isometric logo

---

## 🧪 Run Tests

```bash
python tests/test_core.py
```

All 23 tests pass without TensorFlow installed (analysis core is TF-free).

---

## 📦 Requirements

```text
numpy>=1.24
Pillow>=10.0
gradio>=4.0
matplotlib>=3.8
tensorflow>=2.15   # optional — for object prediction
opencv-python>=4.8 # optional — for true Canny edges
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### Made with ❤️ and pretrained weights

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/vision-studio.svg?style=social)](https://github.com/YOUR_USERNAME/vision-studio/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/vision-studio.svg?style=social)](https://github.com/YOUR_USERNAME/vision-studio/network/members)

<img src="https://capsule-render.vercel.app/api?type=waving&color=4f6bed&height=100&section=footer" width="100%"/>

</div>
