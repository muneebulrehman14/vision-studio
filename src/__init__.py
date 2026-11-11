"""Vision Project — image analysis with pretrained models.

Modules:
    analysis        pixel-level core (TF-free)
    caption         top-k predictions -> English sentence
    classifier      lazy TF classifiers (VGG16 / ResNet50 / EfficientNetB0)
    theme           shared light design tokens
    vision_web      Gradio command-center web UI
    vision_desktop  Tkinter desktop studio
    batch           headless folder -> CSV runner
"""

__version__ = "2.0.0"
