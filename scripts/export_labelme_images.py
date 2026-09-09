#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spatial_axis_gat.config import ensure_dir, load_config, resolve_path
from spatial_axis_gat.data import choose_batches, read_adata, scale_spatial_to_image


def draw_points(pixel_coords: np.ndarray, colors: np.ndarray, width: int, height: int, point_size: int) -> Image.Image:
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for (x, y), color in zip(pixel_coords, colors, strict=False):
        xy = (x - point_size, y - point_size, x + point_size, y + point_size)
        draw.ellipse(xy, fill=tuple(map(int, color[:3])))
    return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    adata = read_adata(cfg["data"]["h5ad_path"], backed="r")
    batch_key = cfg["data"]["batch_key"]
    color_key = cfg["labelme"]["color_by"]
    image_dir = ensure_dir(resolve_path(cfg, cfg["labelme"]["image_dir"]))
    ensure_dir(resolve_path(cfg, cfg["labelme"]["json_dir"]))

    batches = choose_batches(adata.obs, batch_key, cfg["figures"].get("batches"))
    cmap = mpl.colormaps[cfg["labelme"].get("cmap", "viridis")]
    rows = []

    for batch in batches:
        mask = np.asarray(adata.obs[batch_key].astype(str) == str(batch))
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        coords = np.asarray(adata.obsm[cfg["data"]["spatial_key"]][idx])
        values = np.asarray(adata.obs[color_key].iloc[idx], dtype=float)
        pixels, transform = scale_spatial_to_image(coords, int(cfg["labelme"]["default_image_width"]))
        norm = mpl.colors.Normalize(vmin=np.nanpercentile(values, 1), vmax=np.nanpercentile(values, 99))
        colors = (cmap(norm(values))[:, :3] * 255).astype(np.uint8)
        width = int(transform["image_width"])
        height = int(transform["image_height"])
        image = draw_points(pixels, colors, width, height, int(cfg["labelme"]["point_size"]))
        image_name = f"{batch}_label_img.png"
        image.save(image_dir / image_name)
        rows.append({"batch": batch, "image_name": image_name, **transform})
        print(f"Wrote {image_dir / image_name} with {len(idx)} cells")

    manifest = pd.DataFrame(rows)
    manifest_path = resolve_path(cfg, cfg["labelme"]["manifest_csv"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()

