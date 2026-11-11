"""Build the Windows executable:  python build_exe.py

Produces dist/Vision Studio/Vision Studio.exe — a double-click launcher that
starts the local server and opens the browser. Uses a one-folder build because
TensorFlow is ~1.4 GB: startup stays fast (no unpacking) and the app can be
zipped and moved as a single folder.
"""

import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
NAME = "Vision Studio"
ICON = os.path.join(ROOT, "assets", "vision_studio.ico")

ARGS = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--onedir",
    "--console",
    "--name", NAME,
    "--paths", os.path.join(ROOT, "src"),
    # ship the source too: vision_web imports its siblings dynamically inside
    # the frozen bundle (sys.path.insert(dirname(__file__))).
    "--add-data", f"{os.path.join(ROOT, 'src')}{os.pathsep}src",
    # browser tab icon (used as the Gradio favicon at runtime)
    "--add-data", f"{os.path.join(ROOT, 'assets', 'vision_studio.ico')}{os.pathsep}assets",
    # TensorFlow and Keras load shared libraries and data files at runtime.
    "--collect-all", "tensorflow",
    "--collect-all", "keras",
    # Gradio serves its frontend assets from package data, and these wheels ship
    # version.txt / template files that the imports above will not pull in.
    "--collect-all", "gradio",
    "--collect-all", "gradio_client",
    "--collect-all", "safehttpx",
    "--collect-all", "groovy",
    "--specpath", os.path.join(ROOT, "build"),
    "--hidden-import", "uvicorn.logging",
    "--hidden-import", "uvicorn.loops.auto",
    "--hidden-import", "uvicorn.protocols.http.auto",
    "--hidden-import", "uvicorn.protocols.websockets.auto",
    "--hidden-import", "uvicorn.lifespan.on",
    os.path.join(ROOT, "launcher.py"),
]

if os.path.exists(ICON):
    ARGS[ARGS.index("--console") + 1:ARGS.index("--console") + 1] = ["--icon", ICON]

# Packages that call importlib.metadata at runtime need their metadata copied in.
# Distribution names differ from import names (tensorflow-cpu, gradio-client), so
# resolve each one and skip whatever is not installed.
METADATA = ["gradio", "gradio-client", "fastapi", "starlette", "uvicorn",
            "pydantic", "safehttpx", "keras", "tensorflow", "tensorflow-cpu"]


def _add_metadata_args():
    import importlib.metadata as md

    added = []
    for name in METADATA:
        try:
            md.distribution(name)
        except md.PackageNotFoundError:
            continue
        ARGS.extend(["--copy-metadata", name])
        added.append(name)
    return added


def main():
    dist = os.path.join(ROOT, "dist", NAME)
    if os.path.isdir(dist):
        shutil.rmtree(dist, ignore_errors=True)

    print("Building", NAME, "... this bundles TensorFlow, so expect several minutes.")
    print("  metadata:", ", ".join(_add_metadata_args()))
    result = subprocess.run(ARGS, cwd=ROOT)
    if result.returncode != 0:
        print("\nBuild failed. Re-run with --log-level DEBUG for details.")
        return result.returncode

    exe = os.path.join(dist, f"{NAME}.exe")
    size = 0
    for base, _dirs, files in os.walk(dist):
        size += sum(os.path.getsize(os.path.join(base, f)) for f in files)
    print(f"\nDone: {exe}")
    print(f"Folder size: {size / 1e9:.2f} GB")
    print("Double-click the exe: the server starts and the browser opens.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
