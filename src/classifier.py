"""Pretrained ImageNet classifiers (inference only — no training, ever).

Models: VGG16, ResNet50, EfficientNetB0, weights='imagenet'.
TensorFlow is imported lazily so the rest of the app (edges, quality,
colors, resolution, EXIF) works even when TF is not installed.
"""

import threading
import time

import numpy as np


class TFNotAvailable(RuntimeError):
    """Raised when TensorFlow cannot be imported at all."""


class Classifier:
    """Loads pretrained models on demand and caches them. Thread-safe."""

    def __init__(self):
        self.models = {}
        self.registry = None
        self.lock = threading.Lock()

    # -- registry (lazy TF import) -------------------------------------------
    def _reg(self):
        if self.registry is None:
            try:
                from tensorflow.keras.applications import (
                    EfficientNetB0, ResNet50, VGG16,
                    efficientnet, resnet50, vgg16,
                )
            except ImportError as e:
                raise TFNotAvailable(
                    "TensorFlow is not installed. "
                    "Install it with:  pip install tensorflow\n"
                    "Edges, resolution, quality and color analysis still work "
                    "without it — only object prediction needs TF."
                ) from e
            self.registry = {
                "VGG16": (VGG16, vgg16.preprocess_input, vgg16.decode_predictions),
                "ResNet50": (ResNet50, resnet50.preprocess_input,
                             resnet50.decode_predictions),
                "EfficientNetB0": (EfficientNetB0, efficientnet.preprocess_input,
                                    efficientnet.decode_predictions),
            }
        return self.registry

    def available_models(self):
        return ["VGG16", "ResNet50", "EfficientNetB0"]

    # -- inference -------------------------------------------------------------
    def predict(self, img, name, top=5):
        """Predict on a PIL image. Returns ([(label, conf_pct), ...], seconds)."""
        with self.lock:
            ctor, prep, decode = self._reg()[name]
            if name not in self.models:
                self.models[name] = ctor(weights="imagenet")
            x = prep(np.asarray(
                img.resize((224, 224)), dtype="float32")[None].copy())
            t0 = time.time()
            raw = self.models[name].predict(x, verbose=0)
            secs = time.time() - t0
        preds = [(lab.replace("_", " ").title(), float(p) * 100)
                 for _, lab, p in decode(raw, top=top)[0]]
        return preds, secs

    def warm(self, names=None):
        """Force-load the models up front (optional, for batch jobs)."""
        for name in (names or self.available_models()):
            with self.lock:
                ctor = self._reg()[name][0]
                if name not in self.models:
                    self.models[name] = ctor(weights="imagenet")


# ---------------------------------------------------------------------------
# plain-function convenience API (used by CLI / web without an instance)
# ---------------------------------------------------------------------------
_default = Classifier()

def predict(image, model_name="VGG16", top=5):
    """Return (top-k list of (label, conf_pct), seconds)."""
    return _default.predict(image, model_name, top=top)

def get_default_classifier():
    return _default
