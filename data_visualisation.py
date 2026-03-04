# make_montages.py
# Create side-by-side montages from existing PNGs.
#
# Usage:
#   python make_montages.py --scenario brussels_rural
#
# Output:
#   image/<scenario>/montages/*.png

import os
import argparse
from PIL import Image, ImageOps, ImageDraw, ImageFont


def safe_open(path: str):
    if not os.path.isfile(path):
        return None
    img = Image.open(path).convert("RGB")
    return img


def add_title_bar(img: Image.Image, title: str, bar_h: int = 60):
    """Add a white title bar above the image."""
    w, h = img.size
    out = Image.new("RGB", (w, h + bar_h), (255, 255, 255))
    out.paste(img, (0, bar_h))

    draw = ImageDraw.Draw(out)
    # Use default font (portable). If you want bigger fonts, you can point to a .ttf.
    draw.text((10, 10), title, fill=(0, 0, 0))
    return out


def hstack(images, pad=20, bg=(255, 255, 255)):
    """Horizontally stack images (already same height)."""
    heights = [im.size[1] for im in images]
    if len(set(heights)) != 1:
        raise ValueError("Images must have same height for hstack. Use resize_to_same_height().")
    total_w = sum(im.size[0] for im in images) + pad * (len(images) - 1)
    h = heights[0]
    canvas = Image.new("RGB", (total_w, h), bg)
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.size[0] + pad
    return canvas


def resize_to_same_height(images, target_h=None):
    """Resize each image to same height (keeping aspect)."""
    if target_h is None:
        target_h = min(im.size[1] for im in images)  # conservative (avoid upscaling)
    out = []
    for im in images:
        w, h = im.size
        new_w = int(round(w * (target_h / h)))
        out.append(im.resize((new_w, target_h), Image.Resampling.LANCZOS))
    return out


def build_triptych(paths, titles, out_path, pad=30, title_bar=True):
    imgs = []
    used_titles = []
    for p, t in zip(paths, titles):
        im = safe_open(p)
        if im is None:
            print(f"[WARN] Missing: {p}")
            continue
        imgs.append(im)
        used_titles.append(t)

    if len(imgs) < 2:
        print(f"[WARN] Not enough images found to build montage: {out_path}")
        return

    imgs = resize_to_same_height(imgs)
    if title_bar:
        imgs = [add_title_bar(im, t) for im, t in zip(imgs, used_titles)]
        imgs = resize_to_same_height(imgs)  # re-normalize after adding bars

    montage = hstack(imgs, pad=pad)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    montage.save(out_path, quality=95)
    print(f"[OK] Wrote montage: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--image_root", default="image")
    ap.add_argument("--steps", default="300,5400,9000", help="Comma-separated steps for triptychs")
    args = ap.parse_args()

    scenario = args.scenario
    img_dir = os.path.join(args.image_root, scenario)
    conv_dir = os.path.join(img_dir, "convergence_summary")
    out_dir = os.path.join(img_dir, "data_visualisation")

    steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    if len(steps) != 3:
        raise ValueError("--steps must have exactly three values for side-by-side triptychs (e.g., 300,5400,9000)")

    s1, s2, s3 = steps

    # 1) clustered_probability_map triptych (scale=1.0 decay=0.01)
    build_triptych(
        paths=[
            os.path.join(img_dir, f"clustered_probability_map_steps{s1}_scale1.0_decay0.01.png"),
            os.path.join(img_dir, f"clustered_probability_map_steps{s2}_scale1.0_decay0.01.png"),
            os.path.join(img_dir, f"clustered_probability_map_steps{s3}_scale1.0_decay0.01.png"),
        ],
        titles=[f"steps={s1}", f"steps={s2}", f"steps={s3}"],
        out_path=os.path.join(out_dir, f"triptych_clustered_probability_map_scale1.0_decay0.01_{s1}_{s2}_{s3}.png"),
    )

    # 2) compare_steps triptych (scale=1.0 decay=0.01)
    build_triptych(
        paths=[
            os.path.join(img_dir, f"compare_steps{s1}_scale1.0_decay0.01.png"),
            os.path.join(img_dir, f"compare_steps{s2}_scale1.0_decay0.01.png"),
            os.path.join(img_dir, f"compare_steps{s3}_scale1.0_decay0.01.png"),
        ],
        titles=[f"steps={s1}", f"steps={s2}", f"steps={s3}"],
        out_path=os.path.join(out_dir, f"triptych_compare_scale1.0_decay0.01_{s1}_{s2}_{s3}.png"),
    )

    # 3) curve_coverage_iog80 triptych (different decay)
    build_triptych(
        paths=[
            os.path.join(conv_dir, "curve_coverage_iog80_decay0.1.png"),
            os.path.join(conv_dir, "curve_coverage_iog80_decay0.05.png"),
            os.path.join(conv_dir, "curve_coverage_iog80_decay0.01.png"),
        ],
        titles=["decay=0.1", "decay=0.05", "decay=0.01"],
        out_path=os.path.join(out_dir, "triptych_curve_coverage_iog80_by_decay.png"),
    )

    # 4) curve_mean_best_iog triptych (different decay)
    build_triptych(
        paths=[
            os.path.join(conv_dir, "curve_mean_best_iog_decay0.1.png"),
            os.path.join(conv_dir, "curve_mean_best_iog_decay0.05.png"),
            os.path.join(conv_dir, "curve_mean_best_iog_decay0.01.png"),
        ],
        titles=["decay=0.1", "decay=0.05", "decay=0.01"],
        out_path=os.path.join(out_dir, "triptych_curve_mean_best_iog_by_decay.png"),
    )

    # 5) heatmap_coverage_iog80 triptych (steps s1/s2/s3)
    build_triptych(
        paths=[
            os.path.join(conv_dir, f"heatmap_coverage_iog80_steps{s1}.png"),
            os.path.join(conv_dir, f"heatmap_coverage_iog80_steps{s2}.png"),
            os.path.join(conv_dir, f"heatmap_coverage_iog80_steps{s3}.png"),
        ],
        titles=[f"steps={s1}", f"steps={s2}", f"steps={s3}"],
        out_path=os.path.join(out_dir, f"triptych_heatmap_coverage_iog80_{s1}_{s2}_{s3}.png"),
    )


# if __name__ == "__main__":
#     main()
if __name__ == "__main__":

    # ========= USER SETTINGS =========
    SCENARIO = "brussels_rural"#brussels_rural/Graz_A2
    IMAGE_ROOT = "image"
    STEPS = [300, 5400, 9000]
    # ==================================

    scenario = SCENARIO
    img_dir = os.path.join(IMAGE_ROOT, scenario)
    conv_dir = os.path.join(img_dir, "convergence_summary")
    out_dir = os.path.join(img_dir, "data_visualisation")

    s1, s2, s3 = STEPS

    # 1) Clustered probability maps
    build_triptych(
        [
            os.path.join(img_dir, f"clustered_probability_map_steps{s1}_scale1.0_decay0.01.png"),
            os.path.join(img_dir, f"clustered_probability_map_steps{s2}_scale1.0_decay0.01.png"),
            os.path.join(img_dir, f"clustered_probability_map_steps{s3}_scale1.0_decay0.01.png"),
        ],
        [f"steps={s1}", f"steps={s2}", f"steps={s3}"],
        os.path.join(out_dir, "triptych_clustered_probability.png"),
    )

    # 2) Compare plots
    build_triptych(
        [
            os.path.join(img_dir, f"compare_steps{s1}_scale1.0_decay0.01.png"),
            os.path.join(img_dir, f"compare_steps{s2}_scale1.0_decay0.01.png"),
            os.path.join(img_dir, f"compare_steps{s3}_scale1.0_decay0.01.png"),
        ],
        [f"steps={s1}", f"steps={s2}", f"steps={s3}"],
        os.path.join(out_dir, "triptych_compare.png"),
    )

    # 3) Coverage curves
    build_triptych(
        [
            os.path.join(conv_dir, "curve_coverage_iog80_decay0.1.png"),
            os.path.join(conv_dir, "curve_coverage_iog80_decay0.05.png"),
            os.path.join(conv_dir, "curve_coverage_iog80_decay0.01.png"),
        ],
        ["decay=0.1", "decay=0.05", "decay=0.01"],
        os.path.join(out_dir, "triptych_curve_coverage.png"),
    )

    # 4) Mean IoG curves
    build_triptych(
        [
            os.path.join(conv_dir, "curve_mean_best_iog_decay0.1.png"),
            os.path.join(conv_dir, "curve_mean_best_iog_decay0.05.png"),
            os.path.join(conv_dir, "curve_mean_best_iog_decay0.01.png"),
        ],
        ["decay=0.1", "decay=0.05", "decay=0.01"],
        os.path.join(out_dir, "triptych_curve_mean_iog.png"),
    )

    # 5) Heatmaps
    build_triptych(
        [
            os.path.join(conv_dir, f"heatmap_coverage_iog80_steps{s1}.png"),
            os.path.join(conv_dir, f"heatmap_coverage_iog80_steps{s2}.png"),
            os.path.join(conv_dir, f"heatmap_coverage_iog80_steps{s3}.png"),
        ],
        [f"steps={s1}", f"steps={s2}", f"steps={s3}"],
        os.path.join(out_dir, "triptych_heatmap.png"),
    )

    print("Data_visualisation generation complete.")

