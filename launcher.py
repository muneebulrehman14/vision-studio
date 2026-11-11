"""Double-click entry point for the Windows build.

Starts the Vision Studio server bound to localhost, opens the default browser at
the right port, and keeps running until this window is closed.

PyInstaller freezes this file into dist/Vision Studio/Vision Studio.exe.
"""

import multiprocessing
import os
import socket
import sys
import tempfile
import threading
import webbrowser


def _base_dir():
    """Where the bundled files live (onefile: _MEIPASS, onedir: next to the exe)."""
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _free_port(preferred=7860, span=25):
    """First free port, so a second copy (or another app) never blocks startup."""
    for port in range(preferred, preferred + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return 0  # let Gradio choose


def _open_browser_later(url, delay=2.0):
    threading.Timer(delay, lambda: webbrowser.open(url)).start()


def main():
    multiprocessing.freeze_support()

    root = _base_dir()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    # Serverless-friendly paths also help here: keep temp artefacts out of the
    # app folder so the install stays read-only and clean.
    temp_root = os.path.join(tempfile.gettempdir(), "vision-studio")
    os.makedirs(temp_root, exist_ok=True)
    os.environ.setdefault("GRADIO_TEMP_DIR", os.path.join(temp_root, "gradio"))
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(temp_root, "matplotlib"))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    try:
        from src.theme import WEB_CSS
        from src.vision_web import create_app, TF_PRESENT
    except Exception as exc:  # pragma: no cover - startup failure dialogue
        print("\nVision Studio could not start.\n")
        print(f"  {type(exc).__name__}: {exc}\n")
        input("Press Enter to close this window...")
        return 1

    demo = create_app()
    port = _free_port()
    url = f"http://127.0.0.1:{port}"

    print("=" * 62)
    print("  Vision Studio")
    print("=" * 62)
    print(f"  Server : {url}")
    print(f"  Models : {'VGG16, ResNet50, EfficientNetB0' if TF_PRESENT else 'not available (analysis only)'}")
    print()
    print("  The browser opens automatically.")
    print("  Keep this window open — close it to stop the server.")
    print("=" * 62)

    _open_browser_later(url)

    try:
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            inbrowser=False,
            css=WEB_CSS,
            show_api=False,
            quiet=True,
        )
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # pragma: no cover
        print(f"\nServer stopped: {type(exc).__name__}: {exc}")
        input("Press Enter to close this window...")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
