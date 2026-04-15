from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import torch
from line_profiler import LineProfiler

import differentiable_3dshapes as r3d


def _make_inputs(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    fixed_shape: int,
    mixed_shapes: bool,
) -> dict[str, torch.Tensor]:
    """Sample random latent factors used for profiling runs."""
    if mixed_shapes:
        shape = torch.randint(0, 4, (batch_size,), device=device, dtype=torch.int64)
    else:
        shape = torch.full((batch_size,), int(fixed_shape), device=device, dtype=torch.int64)

    return {
        "shape": shape,
        "size": torch.empty((batch_size,), device=device, dtype=dtype).uniform_(0.7, 1.5),
        "orientation": torch.empty((batch_size,), device=device, dtype=dtype).uniform_(0.0, 1.0),
        "floor_hue": torch.empty((batch_size,), device=device, dtype=dtype).uniform_(0.0, 1.0),
        "wall_hue": torch.empty((batch_size,), device=device, dtype=dtype).uniform_(0.0, 1.0),
        "object_hue": torch.empty((batch_size,), device=device, dtype=dtype).uniform_(0.0, 1.0),
    }


def _synchronize_if_needed(device: torch.device) -> None:
    """Synchronize CUDA work so timing includes actual kernel execution."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_render(
    batch_size: int,
    repeats: int,
    warmup: int,
    image_size: int,
    ssaa_scale: int,
    shadow_strength: float,
    fixed_shape: int,
    mixed_shapes: bool,
    with_grad: bool,
    device: torch.device,
    dtype: torch.dtype,
    out_path: Path | None,
) -> None:
    """Run line profiling for the current renderer and print/save timing breakdown."""
    inputs = _make_inputs(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        fixed_shape=fixed_shape,
        mixed_shapes=mixed_shapes,
    )

    renderer = r3d.Differentiable3Dshapes(
        hue_v=0.9,
        shadow_strength=shadow_strength,
        ssaa_scale=ssaa_scale,
        image_size=image_size,
        output_chw=True,
    ).to(device)

    def run_once() -> torch.Tensor | tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Single profiled forward pass."""
        return renderer.forward(
            shape=inputs["shape"],
            size=inputs["size"],
            orientation=inputs["orientation"],
            floor_hue=inputs["floor_hue"],
            wall_hue=inputs["wall_hue"],
            object_hue=inputs["object_hue"],
            return_grad=with_grad,
        )

    # Warm up kernels/allocators before collecting profile numbers.
    for _ in range(warmup):
        _ = run_once()
        _synchronize_if_needed(device)

    lp = LineProfiler(
        r3d.Differentiable3Dshapes.forward,
        r3d.render_3dshapes_image,
        r3d._prepare_render_context,
        r3d._build_camera_rays,
        r3d._build_scene_params,
        r3d._render_floor_layer,
        r3d._render_wall_layers,
        r3d._render_object_layer,
        r3d._composite_layers,
        r3d._finalize_render_output,
    )

    wrapped = lp(run_once)

    # Profile timing across repeated wrapped calls.
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = wrapped()
    _synchronize_if_needed(device)
    elapsed = time.perf_counter() - t0

    header = (
        f"device={device} dtype={dtype} batch={batch_size} repeats={repeats} "
        f"fixed_shape={fixed_shape} mixed_shapes={mixed_shapes} with_grad={with_grad}\n"
        f"image_size={image_size} ssaa_scale={ssaa_scale} shadow_strength={shadow_strength}\n"
        f"total_elapsed={elapsed:.4f}s avg_per_call={elapsed / repeats:.6f}s\n\n"
    )
    print(header, end="")

    stats_buf = io.StringIO()
    lp.print_stats(stream=stats_buf)
    stats_text = stats_buf.getvalue()
    print(stats_text, end="")

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(header + stats_text, encoding="utf-8")
        print(f"\nSaved profile: {out_path}")


def main() -> None:
    """Parse CLI options and execute the line profiler."""
    parser = argparse.ArgumentParser(description="Line profiler for differentiable_3dshapes.py")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--ssaa-scale", type=int, default=1)
    parser.add_argument("--shadow-strength", type=float, default=1.0)
    parser.add_argument("--fixed-shape", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--mixed-shapes", action="store_true", help="Profile mixed shape batches (0..3)")
    parser.add_argument("--with-grad", action="store_true", help="Include jacobian path (return_grad=True)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out", type=Path, default=None, help="Path to save profiler text output")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    dtype = torch.float32
    profile_render(
        batch_size=args.batch_size,
        repeats=args.repeats,
        warmup=args.warmup,
        image_size=args.image_size,
        ssaa_scale=args.ssaa_scale,
        shadow_strength=args.shadow_strength,
        fixed_shape=args.fixed_shape,
        mixed_shapes=args.mixed_shapes,
        with_grad=args.with_grad,
        device=device,
        dtype=dtype,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
