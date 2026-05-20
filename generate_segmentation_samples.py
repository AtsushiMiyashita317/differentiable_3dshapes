from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import differentiable_3dshapes as r3d


SEGMENTATION_PALETTE = np.array(
    [
        [114, 168, 214],  # background
        [88, 138, 71],  # floor
        [204, 126, 64],  # wall_back
        [186, 74, 94],  # wall_front
        [118, 91, 177],  # wall_left
        [83, 159, 154],  # wall_right
        [238, 210, 82],  # object
    ],
    dtype=np.uint8,
)

FACTOR_NAMES = ["size", "orientation", "floor_hue", "wall_hue", "object_hue"]
BASE_FACTOR_VALUES = {
    "size": 1.0,
    "orientation": 0.16,
    "floor_hue": 0.05,
    "wall_hue": 0.36,
    "object_hue": 0.68,
}
FACTOR_SWEEP_RANGES = {
    "size": (0.7, 1.5),
    "orientation": (0.0, 1.0),
    "floor_hue": (0.0, 1.0),
    "wall_hue": (0.0, 1.0),
    "object_hue": (0.0, 1.0),
}


def _save_rgb(path: Path, image_hwc: np.ndarray) -> None:
    """Save a float RGB image in [0, 1]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(image_hwc, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def _chw_to_hwc(x: torch.Tensor) -> np.ndarray:
    """Convert CHW image tensor to HWC NumPy array."""
    return x.detach().cpu().permute(1, 2, 0).numpy()


def _diff_to_rgb(diff_chw: torch.Tensor) -> np.ndarray:
    """Visualize signed image derivatives as RGB around neutral gray."""
    diff = _chw_to_hwc(diff_chw)
    scale = float(np.percentile(np.abs(diff), 99.5))
    if scale < 1e-8:
        scale = 1.0
    return np.clip(0.5 + 0.5 * diff / scale, 0.0, 1.0)


def _labels_to_rgb(labels_hw: torch.Tensor) -> np.ndarray:
    """Colorize segmentation labels."""
    labels = labels_hw.detach().cpu().numpy().astype(np.int64)
    return SEGMENTATION_PALETTE[labels]


def _masks_to_rgb_grid(masks_chw: torch.Tensor) -> np.ndarray:
    """Lay out soft class masks as colored panels."""
    masks = masks_chw.detach().cpu().numpy()
    panels = []
    for class_id in range(masks.shape[0]):
        color = SEGMENTATION_PALETTE[class_id].astype(np.float32) / 255.0
        panels.append(masks[class_id, :, :, None] * color[None, None, :])
    return np.concatenate(panels, axis=1)


def _float_rgb_to_u8(image_hwc: np.ndarray) -> np.ndarray:
    """Convert float RGB in [0, 1] to uint8."""
    return (np.clip(image_hwc, 0.0, 1.0) * 255).astype(np.uint8)


def _gif_frame_from_factor_batch(
    image: torch.Tensor,
    jac: tuple[torch.Tensor, ...],
    labels: torch.Tensor,
) -> np.ndarray:
    """Build one GIF frame: rows vary target factors, columns show image/diffs/labels."""
    rows = []
    for row_idx in range(len(FACTOR_NAMES)):
        panels = [_chw_to_hwc(image[row_idx])]
        panels.extend(_diff_to_rgb(j[row_idx]) for j in jac)
        panels.append(_labels_to_rgb(labels[row_idx]).astype(np.float32) / 255.0)
        rows.append(np.concatenate(panels, axis=1))
    return _float_rgb_to_u8(np.concatenate(rows, axis=0))


def _factor_batch_for_frame(
    shape: int,
    device: torch.device,
    frame_idx: int,
    num_frames: int,
) -> dict[str, torch.Tensor]:
    """Create a batch where each row sweeps one continuous factor."""
    denom = max(num_frames - 1, 1)
    t = frame_idx / denom
    values_by_name: dict[str, list[float]] = {name: [] for name in FACTOR_NAMES}

    for sweep_name in FACTOR_NAMES:
        row_values = dict(BASE_FACTOR_VALUES)
        lo, hi = FACTOR_SWEEP_RANGES[sweep_name]
        value = lo + (hi - lo) * t
        if sweep_name != "size":
            value = value % 1.0
        row_values[sweep_name] = value
        for name in FACTOR_NAMES:
            values_by_name[name].append(row_values[name])

    out = {
        "shape": torch.full((len(FACTOR_NAMES),), int(shape), device=device, dtype=torch.int64),
    }
    for name in FACTOR_NAMES:
        out[name] = torch.tensor(values_by_name[name], device=device, dtype=torch.float32)
    return out


def generate_factor_sweep_gif(
    renderer: r3d.Differentiable3Dshapes,
    out_path: Path,
    device: torch.device,
    shape: int,
    num_frames: int,
    duration_ms: int,
) -> None:
    """Save a GIF sweeping each continuous factor in one row."""
    frames: list[Image.Image] = []
    for frame_idx in range(num_frames):
        factors = _factor_batch_for_frame(
            shape=shape,
            device=device,
            frame_idx=frame_idx,
            num_frames=num_frames,
        )
        jac, image, segmentation = renderer.forward(
            **factors,
            return_grad=True,
            return_segmentation=True,
        )
        frame = _gif_frame_from_factor_batch(image, jac, segmentation.labels)
        frames.append(Image.fromarray(frame))
        print(f"factor sweep frame {frame_idx + 1}/{num_frames} done")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def generate_samples(
    out_dir: Path,
    device: torch.device,
    image_size: int,
    ssaa_scale: int,
    shape: int,
    gif_frames: int,
    gif_duration_ms: int,
) -> None:
    """Render one image and save image, derivatives, and GT segmentation samples."""
    renderer = r3d.Differentiable3Dshapes(
        hue_v=0.9,
        shadow_strength=0.8,
        ssaa_scale=ssaa_scale,
        image_size=image_size,
        output_chw=True,
    ).to(device)

    factors = dict(
        shape=torch.tensor([shape], device=device),
        size=torch.tensor([1.0], device=device),
        orientation=torch.tensor([0.16], device=device),
        floor_hue=torch.tensor([0.05], device=device),
        wall_hue=torch.tensor([0.36], device=device),
        object_hue=torch.tensor([0.68], device=device),
    )

    jac, image, segmentation = renderer.forward(
        **factors,
        return_grad=True,
        return_segmentation=True,
    )

    image_0 = image[0]
    jac_0 = [j[0] for j in jac]
    labels_0 = segmentation.labels[0]
    masks_0 = segmentation.masks[0]

    _save_rgb(out_dir / "image.png", _chw_to_hwc(image_0))
    _save_rgb(out_dir / "segmentation_labels.png", _labels_to_rgb(labels_0).astype(np.float32) / 255.0)
    _save_rgb(out_dir / "segmentation_masks.png", _masks_to_rgb_grid(masks_0))

    factor_names = ["size", "orientation", "floor_hue", "wall_hue", "object_hue"]
    diff_panels = []
    for name, diff in zip(factor_names, jac_0):
        diff_rgb = _diff_to_rgb(diff)
        diff_panels.append(diff_rgb)
        _save_rgb(out_dir / f"differential_{name}.png", diff_rgb)
    _save_rgb(out_dir / "differentials.png", np.concatenate(diff_panels, axis=1))
    generate_factor_sweep_gif(
        renderer=renderer,
        out_path=out_dir / "factor_sweep.gif",
        device=device,
        shape=shape,
        num_frames=gif_frames,
        duration_ms=gif_duration_ms,
    )

    summary = (
        f"image_shape={tuple(image.shape)}\n"
        f"labels_shape={tuple(segmentation.labels.shape)}\n"
        f"masks_shape={tuple(segmentation.masks.shape)}\n"
        f"label_ids={sorted(int(x) for x in torch.unique(segmentation.labels).cpu())}\n"
        f"class_names={list(segmentation.class_names)}\n"
        f"factor_sweep_gif={out_dir / 'factor_sweep.gif'}\n"
        f"factor_sweep_layout=rows:{FACTOR_NAMES}; columns:image,{FACTOR_NAMES},segmentation_labels\n"
    )
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"saved samples: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate image, derivative, and segmentation samples.")
    parser.add_argument("--out-dir", type=Path, default=Path("samples/segmentation_smoke"))
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--ssaa-scale", type=int, default=1)
    parser.add_argument("--shape", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--gif-frames", type=int, default=24)
    parser.add_argument("--gif-duration-ms", type=int, default=120)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    generate_samples(
        out_dir=args.out_dir,
        device=device,
        image_size=args.image_size,
        ssaa_scale=args.ssaa_scale,
        shape=args.shape,
        gif_frames=args.gif_frames,
        gif_duration_ms=args.gif_duration_ms,
    )


if __name__ == "__main__":
    main()
