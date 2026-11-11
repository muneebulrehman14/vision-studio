"""Text captioning: turn top-k predictions into a plain-English sentence.

Shared by both the web and desktop front ends.
"""


def article(word):
    return "an" if word[:1].lower() in "aeiou" else "a"


def make_caption(preds):
    """preds = [(label, confidence_pct), ...] ; returns one English sentence."""
    if not preds:
        return "No prediction available."
    (l1, c1), (l2, c2) = preds[0], preds[1]
    if c1 >= 50:
        return f"This image contains {article(l1)} {l1} ({c1:.0f}%)"
    if c1 >= 25:
        return (f"Most likely {article(l1)} {l1} ({c1:.0f}%), "
                f"possibly {article(l2)} {l2} ({c2:.0f}%)")
    return (f"Not sure: could be {article(l1)} {l1} ({c1:.0f}%) "
            f"or {article(l2)} {l2} ({c2:.0f}%)")
