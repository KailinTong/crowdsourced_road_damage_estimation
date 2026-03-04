from pathlib import Path
from PIL import Image


# =========================
# CONFIGURATION
# =========================
SCENARIO_NAME = "Graz_A2"#brussels_rural/Graz_A2
IMAGE_ROOT = "image"

TARGET_STEPS = [300, 5400, 9000]
TARGET_SCALE = 1.0
TARGET_DECAY = 0.01

OVERLAP_PIXELS = 800   # <-- increase for more overlap
OUTPUT_NAME = "tight_compare_overlap.png"


# =========================
# FILENAME FILTER
# =========================
import re

COMPARE_RE = re.compile(
    r"^compare_.*steps(?P<steps>\d+).*scale(?P<scale>[\d.]+).*decay(?P<decay>[\d.]+).*\.(png|jpg|jpeg)$",
    re.IGNORECASE,
)


def collect_images(folder: Path):
    images = {}

    for file in folder.iterdir():
        match = COMPARE_RE.match(file.name)
        if not match:
            continue

        step = int(match.group("steps"))
        scale = float(match.group("scale"))
        decay = float(match.group("decay"))

        if (
            step in TARGET_STEPS
            and abs(scale - TARGET_SCALE) < 1e-9
            and abs(decay - TARGET_DECAY) < 1e-9
        ):
            images[step] = Image.open(file).convert("RGB")

    return images


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    folder = Path(IMAGE_ROOT) / SCENARIO_NAME
    images = collect_images(folder)

    # Sort by step
    ordered = [images[s] for s in sorted(images.keys())]

    if len(ordered) != 3:
        raise ValueError("Did not find exactly 3 images.")

    w, h = ordered[0].size

    # Final canvas width with overlap
    total_width = w + (w - OVERLAP_PIXELS) * 2
    canvas = Image.new("RGB", (total_width, h), (255, 255, 255))

    # Place first image
    canvas.paste(ordered[0], (0, 0))

    # Place second image (overlapping first)
    canvas.paste(ordered[1], (w - OVERLAP_PIXELS, 0))

    # Place third image (overlapping second)
    canvas.paste(ordered[2], (2 * (w - OVERLAP_PIXELS), 0))

    out_path = folder / Path("data_visualisation")/ OUTPUT_NAME
    canvas.save(out_path, dpi=(300, 300))


    print(f"[OK] Saved to {out_path}")
