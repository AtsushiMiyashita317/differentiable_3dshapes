from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import torch
from line_profiler import LineProfiler

import source as r3d


def _make_inputs(batch_size: int, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "shape": torch.randint(0, 4, (batch_size,), device=device, dtype=torch.int64),
        "size": torch.empty((batch_size,), device=device, dtype=dtype).uniform_(0.7, 1.5),
        "orientation": torch.empty((batch_size,), device=device, dtype=dtype).uniform_(0.0, 1.0),
        "floor_hue": torch.empty((batch_size,), device=device, dtype=dtype).uniform_(0.0, 1.0),
        "wall_hue": torch.empty((batch_size,), device=device, dtype=dtype).uniform_(0.0, 1.0),
        "object_hue": torch.empty((batch_size,), device=device, dtype=dtype).uniform_(0.0, 1.0),
    }


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def profile_render(
    batch_size: int,
    repeats: int,
    warmup: int,
    image_size: int,
    ssaa_scale: int,
    device: torch.device,
    dtype: torch.dtype,
    out_path: Path | None,
) -> None:
    inputs = _make_inputs(batch_size=batch_size, device=device, dtype=dtype)

    kwargs = dict(
        hue_v=0.9,
        shadow_strength=0.8,
        ssaa_scale=ssaa_scale,
        image_size=image_size,
        output_chw=True,
    )

    renderer = r3d.Render3DShapesModule(**kwargs).to(device)

    def run_once() -> torch.Tensor:
        return renderer.forward(
            shape=inputs["shape"],
            size=inputs["size"],
            orientation=inputs["orientation"],
            floor_hue=inputs["floor_hue"],
            wall_hue=inputs["wall_hue"],
            object_hue=inputs["object_hue"],
            return_grad=False,
        )

    # Warmup
    for _ in range(warmup):
        _ = run_once()
        _synchronize_if_needed(device)

    lp = LineProfiler(
        r3d.Render3DShapesModule.forward,
        r3d.render_3dshapes_image,
        r3d._render_room,
        r3d._soft_shadow_floor,
        r3d._ray_march_object,
        r3d._normal_from_sdf,
        r3d._sdf_object_world,
    )

    wrapped = lp(run_once)

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = wrapped()
    _synchronize_if_needed(device)
    elapsed = time.perf_counter() - t0

    header = (
        f"device={device} dtype={dtype} batch={batch_size} repeats={repeats}\n"
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
    parser = argparse.ArgumentParser(description="Line profiler for renderer_3dshapes_sim.py")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--ssaa-scale", type=int, default=4)
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
        device=device,
        dtype=dtype,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
