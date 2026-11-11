"""Generate assets/vision_studio.ico — the app icon for the Windows build.

Rounded indigo square + isometric cube, drawn at 4x and downsampled so the
edges stay clean at 16 px in the taskbar.
"""

import os

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "assets", "vision_studio.ico")
S = 1024
M = 4  # supersample factor


def gradient(size, top=(92, 118, 243), bottom=(53, 73, 192)):
    img = Image.new("RGB", (1, size))
    px = img.load()
    for y in range(size):
        t = y / max(size - 1, 1)
        px[0, y] = tuple(round(top[i] + (bottom[i] - top[i]) * t) for i in range(3))
    return img.resize((size, size), Image.BILINEAR)


def main():
    big = gradient(S * M)

    mask = Image.new("L", (S * M, S * M), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, S * M - 1, S * M - 1), radius=int(0.22 * S * M), fill=255)

    icon = Image.new("RGBA", (S * M, S * M), (0, 0, 0, 0))
    icon.paste(big, (0, 0), mask)

    d = ImageDraw.Draw(icon, "RGBA")
    cx, half, top_y, mid_y, bot_y = 512 * M, 232 * M, 268 * M, 402 * M, 536 * M
    top = [(cx, top_y), (cx + half, mid_y), (cx, bot_y), (cx - half, mid_y)]
    left = [(cx - half, mid_y), (cx, bot_y), (cx, bot_y + 268 * M), (cx - half, mid_y + 268 * M)]
    right = [(cx, bot_y), (cx + half, mid_y), (cx + half, mid_y + 268 * M), (cx, bot_y + 268 * M)]

    d.polygon(top, fill=(255, 255, 255, 255))
    d.polygon(left, fill=(199, 210, 255, 255))
    d.polygon(right, fill=(150, 170, 250, 255))

    # separate the faces so the cube still reads at 16 px
    edge = (53, 73, 192, 255)
    w = int(9 * M)
    d.line([top[3], top[2], top[1]], fill=edge, width=w, joint="curve")
    d.line([left[0], left[1]], fill=edge, width=w, joint="curve")
    d.line([right[0], right[1]], fill=edge, width=w, joint="curve")
    d.line([(cx, bot_y), (cx, bot_y + 268 * M)], fill=edge, width=w)

    # small eye dot on the top face — suggests "vision"
    r = 34 * M
    ex, ey = cx, top_y + 118 * M
    d.ellipse((ex - r, ey - r, ex + r, ey + r), fill=(53, 73, 192, 235))

    icon = icon.resize((S, S), Image.LANCZOS)
    icon.save(OUT, sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    print("wrote", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
