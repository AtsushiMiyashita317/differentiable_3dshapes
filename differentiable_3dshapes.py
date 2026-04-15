from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Utility
# ------------------------------

def _parse_image_size(image_size: int | Tuple[int, int]) -> Tuple[int, int]:
    """Normalize `image_size` to `(height, width)` and validate positivity."""
    if isinstance(image_size, int):
        h = w = int(image_size)
    else:
        h, w = int(image_size[0]), int(image_size[1])
    if h <= 0 or w <= 0:
        raise ValueError("image_size must be positive")
    return h, w


@dataclass(frozen=True)
class CameraConfig:
    radius: float = 6.0
    height: float = 1.8
    target: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    yaw_offset_deg: float = 0.0


@dataclass(frozen=True)
class RoomConfig:
    floor_y: float = -1.2
    wall_top_y: float = 5.0
    half_width: float = 10.0
    half_depth: float = 10.0
    draw_back_wall: bool = True
    draw_left_wall: bool = True
    draw_right_wall: bool = True
    draw_front_wall: bool = True


@dataclass(frozen=True)
class ObjectConfig:
    base_scales: Tuple[Tuple[float, float, float], ...] = (
        (1.0, 1.0, 1.0),  # shape 0: cube
        (1.0, 1.0, 1.0),  # shape 1: cylinder
        (1.0, 1.0, 1.0),  # shape 2: sphere
        (1.0, 1.0, 1.0),  # shape 3: capsule
    )
    ground_clearance_by_shape: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    center_x: float = 0.0
    center_z: float = 0.0
    global_scale_multiplier: float = 1.0
    use_auto_grounding: bool = True


@dataclass(frozen=True)
class LightingConfig:
    light_position_world: Tuple[float, float, float] = (-7.0, 30.0, -2.0)
    ambient: float = 0.55
    diffuse: float = 0.55


@dataclass(frozen=True)
class MeshResolutionConfig:
    cylinder_radial_steps: int = 64
    sphere_lat_steps: int = 36
    sphere_lon_steps: int = 72
    capsule_radial_steps: int = 56
    capsule_hemi_steps: int = 18


# ------------------------------
# Primitive meshes
# ------------------------------


def _device_cache_key(device: torch.device) -> tuple[str, int]:
    """Convert a torch device into a hashable cache key."""
    return device.type, -1 if device.index is None else int(device.index)


def _device_from_cache_key(device_type: str, device_index: int) -> torch.device:
    """Reconstruct a torch device from cached `(type, index)` fields."""
    return torch.device(device_type) if device_index < 0 else torch.device(device_type, device_index)


@lru_cache(maxsize=128)
def _cached_scalar_tensor(device_type: str, device_index: int, dtype: torch.dtype, value: float) -> torch.Tensor:
    """Return a cached scalar tensor for repeated constant creation."""
    device = _device_from_cache_key(device_type, device_index)
    return torch.tensor(float(value), device=device, dtype=dtype)


@lru_cache(maxsize=128)
def _cached_vec3_tensor(
    device_type: str,
    device_index: int,
    dtype: torch.dtype,
    x: float,
    y: float,
    z: float,
) -> torch.Tensor:
    """Return a cached length-3 tensor on the requested device and dtype."""
    device = _device_from_cache_key(device_type, device_index)
    return torch.tensor([x, y, z], device=device, dtype=dtype)


@lru_cache(maxsize=32)
def _cached_image_plane(
    h: int,
    w: int,
    device_type: str,
    device_index: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build and cache supersampled image-plane coordinate grids."""
    device = _device_from_cache_key(device_type, device_index)
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5 - (w - 1) * 0.5) / (48.0 * (w / 64.0))
    ys = -((torch.arange(h, device=device, dtype=dtype) + 0.5 - (h - 1) * 0.5) / (48.0 * (w / 64.0)))
    return torch.meshgrid(xs, ys, indexing="xy")


def _downsample_mean(
    image_hwc: np.ndarray | torch.Tensor,
    factor: int,
) -> np.ndarray | torch.Tensor:
    """Downsample an image by block-average with NumPy or torch backend."""
    if factor <= 1:
        return image_hwc
    if torch.is_tensor(image_hwc):
        if image_hwc.ndim == 3:
            h, w, _ = image_hwc.shape
            x = image_hwc.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            squeeze_batch = True
        elif image_hwc.ndim == 4:
            _, h, w, _ = image_hwc.shape
            x = image_hwc.permute(0, 3, 1, 2)  # (B, C, H, W)
            squeeze_batch = False
        else:
            raise ValueError("torch image tensor must be HWC or BHWC")

        # Area interpolation keeps energy stable after supersampling.
        x = F.interpolate(x, size=(h // factor, w // factor), mode="area")
        if squeeze_batch:
            return x.squeeze(0).permute(1, 2, 0)
        return x.permute(0, 2, 3, 1)
    h, w, c = image_hwc.shape
    image_hwc = image_hwc.reshape(h // factor, factor, w // factor, factor, c)
    return image_hwc.mean(axis=(1, 3))


def _as_torch_scalar(x: float | torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert a scalar-like input into a tensor on the target device/dtype."""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(float(x), device=device, dtype=dtype)


def _as_torch_vec3(
    x: Sequence[float] | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert input into a length-3 tensor on the target device/dtype."""
    if torch.is_tensor(x):
        out = x.to(device=device, dtype=dtype).reshape(-1)
        if out.numel() != 3:
            raise ValueError("Expected a length-3 vector tensor")
        return out
    out = torch.tensor(x, device=device, dtype=dtype).reshape(-1)
    if out.numel() != 3:
        raise ValueError("Expected a length-3 vector")
    return out


def _as_torch_shape_scalar(
    x: Sequence[float] | torch.Tensor,
    shape_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pick one shape-dependent scalar and return it as a tensor."""
    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype)
        if t.ndim == 0:
            return t
        if t.shape[0] <= shape_id:
            raise ValueError("Tensor shape axis is too short for selected shape_id")
        return t[shape_id]
    return torch.tensor(float(x[shape_id]), device=device, dtype=dtype)


def _as_torch_shape_vec3(
    x: Sequence[Sequence[float]] | torch.Tensor,
    shape_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pick one shape-dependent 3D vector and return it as a tensor."""
    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype)
        if t.ndim == 1:
            if t.numel() != 3:
                raise ValueError("Expected a length-3 vector tensor")
            return t
        if t.ndim >= 2:
            if t.shape[0] <= shape_id:
                raise ValueError("Tensor shape axis is too short for selected shape_id")
            out = t[shape_id].reshape(-1)
            if out.numel() != 3:
                raise ValueError("Expected selected shape vector to have length 3")
            return out
        raise ValueError("Unsupported tensor rank for shape vector")
    out = torch.tensor(x[shape_id], device=device, dtype=dtype).reshape(-1)
    if out.numel() != 3:
        raise ValueError("Expected selected shape vector to have length 3")
    return out


def _hue_to_rgb_constrained(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Map hue/value factors to a constrained RGB gamut used by 3dshapes."""
    h = h * (2.0 * math.pi)  # hue in [0,1) -> angle in [0,2pi)
    v = torch.clamp(v, 0.0, 1.0)
    c0 = torch.stack([v, (1.5 - v) * 0.5, (1.5 - v) * 0.5], dim=-1)

    dkey = _device_cache_key(h.device)
    m = _cached_vec3_tensor(dkey[0], dkey[1], h.dtype, 0.5, 0.5, 0.5)
    p0 = c0 - m
    rho2 = torch.sum(p0 * p0, dim=-1, keepdim=True)
    eps = _cached_scalar_tensor(dkey[0], dkey[1], h.dtype, 1e-12)
    rho = torch.sqrt(torch.clamp(rho2, min=eps))
    u = p0 / rho

    n = _cached_vec3_tensor(dkey[0], dkey[1], h.dtype, 1.0, 1.0, 1.0) / math.sqrt(3.0)
    w = torch.cross(n, u, dim=-1)
    w_norm = torch.sqrt(torch.clamp(torch.sum(w * w, dim=-1, keepdim=True), min=eps))
    w = w / w_norm

    p = rho * (torch.cos(h)[..., None] * u + torch.sin(h)[..., None] * w)
    c = m + p
    return torch.where(rho2 < eps, m, c)


def _normal_object_analytic(
    p_world: torch.Tensor,
    shape_id: int,
    center_world: torch.Tensor,
    scale_xyz: torch.Tensor,
) -> torch.Tensor:
    """Analytic surface normal in world space for supported primitives."""
    scale_safe = torch.clamp(scale_xyz, min=1e-8)
    p_local = (p_world - center_world) / scale_safe
    px, py, pz = p_local[..., 0], p_local[..., 1], p_local[..., 2]
    eps = torch.as_tensor(1e-8, device=p_world.device, dtype=p_world.dtype)

    if shape_id == 2:
        n_local = p_local / torch.clamp(torch.linalg.norm(p_local, dim=-1, keepdim=True), min=eps)
    elif shape_id == 1:
        r = torch.sqrt(torch.clamp(px * px + pz * pz, min=eps))
        side_n = torch.stack([px / r, torch.zeros_like(py), pz / r], dim=-1)
        cap_n = torch.stack([torch.zeros_like(px), torch.sign(py), torch.zeros_like(pz)], dim=-1)
        side_res = torch.abs(r - 1.0)
        cap_res = torch.abs(torch.abs(py) - 1.0)
        use_side = side_res <= cap_res
        n_local = torch.where(use_side[..., None], side_n, cap_n)
    elif shape_id == 3:
        cy = torch.clamp(py, -1.0, 1.0)
        v = torch.stack([px, py - cy, pz], dim=-1)
        n_local = v / torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)
    else:
        ax = torch.abs(px)
        ay = torch.abs(py)
        az = torch.abs(pz)
        mx = (ax >= ay) & (ax >= az)
        my = (~mx) & (ay >= az)
        mz = (~mx) & (~my)
        n_local = torch.stack(
            [
                torch.where(mx, torch.sign(px), torch.zeros_like(px)),
                torch.where(my, torch.sign(py), torch.zeros_like(py)),
                torch.where(mz, torch.sign(pz), torch.zeros_like(pz)),
            ],
            dim=-1,
        )

    # Transform local normal by inverse-transpose of diagonal scale matrix.
    n_world = n_local / scale_safe
    return n_world / torch.clamp(torch.linalg.norm(n_world, dim=-1, keepdim=True), min=eps)


def _smooth_positive(x: torch.Tensor, delta: float = 1e-3) -> torch.Tensor:
    """C1-smooth approximation of relu(x)."""
    d = torch.as_tensor(delta, device=x.device, dtype=x.dtype)
    lim = torch.as_tensor(1e4, device=x.device, dtype=x.dtype)
    x_safe = torch.where(
        torch.isfinite(x),
        x,
        torch.where(x >= 0.0, lim, -lim),
    )
    return 0.5 * (x_safe + torch.sqrt(x_safe * x_safe + d * d))


def _occlusion_from_segment_length(seg_len: torch.Tensor, sharpness: float) -> torch.Tensor:
    """Saturating occlusion with d/dL=0 at L=0."""
    k = torch.as_tensor(float(sharpness), device=seg_len.device, dtype=seg_len.dtype)
    l = torch.clamp(seg_len, min=0.0)
    return 1.0 - torch.exp(-k * l * l)


def _ray_object_hit_interval_t(
    ray_origin_world: torch.Tensor,
    ray_dir_world: torch.Tensor,
    shape_id: int,
    center_world: torch.Tensor,
    scale_xyz: torch.Tensor,
    t_far: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Analytic entry/exit interval along ray parameter t (world-distance parameter)."""
    # Shape notation:
    #   P = ray grid dims (e.g. B,H,W) or any broadcastable leading dims.
    #   Vector tensors have shape (P, 3), scalar fields have shape (P,).
    dtype = ray_dir_world.dtype
    device = ray_dir_world.device
    inf = torch.full_like(ray_dir_world[..., 0], float("inf"))  # (P,)
    ninf = torch.full_like(ray_dir_world[..., 0], -float("inf"))  # (P,)
    eps = torch.as_tensor(1e-10, device=device, dtype=dtype)  # ()
    sqrt_eps = torch.as_tensor(1e-12, device=device, dtype=dtype)  # ()

    scale_xyz_safe = torch.clamp(scale_xyz, min=1e-8)  # (P, 3)
    ro = (ray_origin_world - center_world) / scale_xyz_safe  # (P, 3)
    rd = ray_dir_world / scale_xyz_safe  # (P, 3)

    ox, oy, oz = ro[..., 0], ro[..., 1], ro[..., 2]  # each (P,)
    dx, dy, dz = rd[..., 0], rd[..., 1], rd[..., 2]  # each (P,)

    if shape_id == 0:
        parallel = torch.abs(rd) < eps  # (P, 3)
        outside_parallel = parallel & ((ro < -1.0) | (ro > 1.0))  # (P, 3)
        no_hit_parallel = outside_parallel.any(dim=-1)  # (P,)
        rd_safe = torch.where(torch.abs(rd) > eps, rd, torch.where(rd >= 0.0, eps, -eps))  # (P, 3)
        inv_rd = 1.0 / rd_safe  # (P, 3)
        t0 = (-1.0 - ro) * inv_rd  # (P, 3)
        t1 = (1.0 - ro) * inv_rd  # (P, 3)
        t_near_axis = torch.minimum(t0, t1)  # (P, 3)
        t_far_axis = torch.maximum(t0, t1)  # (P, 3)
        t_near_axis = torch.where(parallel, ninf[..., None], t_near_axis)  # (P, 3)
        t_far_axis = torch.where(parallel, inf[..., None], t_far_axis)  # (P, 3)
        t_enter = t_near_axis.max(dim=-1).values  # (P,)
        t_exit = t_far_axis.min(dim=-1).values  # (P,)
        valid = (~no_hit_parallel) & (t_exit >= t_enter) & (t_exit > 0.0)  # (P,)
        return torch.where(valid, t_enter, inf), torch.where(valid, t_exit, ninf)

    def _push_root(
        roots: list[torch.Tensor],
        valids: list[torch.Tensor],
        t: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Append one candidate hit root with finite fallback values."""
        roots.append(torch.nan_to_num(t, nan=float(t_far), posinf=float(t_far), neginf=-float(t_far)))
        valids.append(valid)

    roots: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []

    def _quadratic(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Solve quadratic roots robustly and return `(t0, t1, valid_mask)`."""
        disc = b * b - 4.0 * a * c  # (P,)
        den = 2.0 * a  # (P,)
        den_ok = torch.abs(den) > eps  # (P,)
        root_ok = (disc >= 0.0) & den_ok  # (P,)
        den_safe = torch.where(den_ok, den, torch.ones_like(den))  # (P,)
        # Avoid infinite/unstable gradients around tangential hits (disc ~= 0).
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=sqrt_eps))  # (P,)
        t0 = (-b - sqrt_disc) / den_safe  # (P,)
        t1 = (-b + sqrt_disc) / den_safe  # (P,)
        return t0, t1, root_ok

    if shape_id == 2:
        t0, t1, ok = _quadratic(
            dx * dx + dy * dy + dz * dz,
            2.0 * (ox * dx + oy * dy + oz * dz),
            ox * ox + oy * oy + oz * oz - 1.0,
        )
        _push_root(roots, valids, t0, ok)
        _push_root(roots, valids, t1, ok)
    elif shape_id == 1:
        t0, t1, ok_side = _quadratic(
            dx * dx + dz * dz,
            2.0 * (ox * dx + oz * dz),
            ox * ox + oz * oz - 1.0,
        )
        y0 = oy + t0 * dy
        y1 = oy + t1 * dy
        _push_root(roots, valids, t0, ok_side & (y0 >= -1.0) & (y0 <= 1.0))
        _push_root(roots, valids, t1, ok_side & (y1 >= -1.0) & (y1 <= 1.0))

        dy_ok = torch.abs(dy) > eps
        dy_safe = torch.where(dy_ok, dy, torch.ones_like(dy))
        t_top = (1.0 - oy) / dy_safe
        x_top = ox + t_top * dx
        z_top = oz + t_top * dz
        _push_root(roots, valids, t_top, dy_ok & (x_top * x_top + z_top * z_top <= 1.0))

        t_bot = (-1.0 - oy) / dy_safe
        x_bot = ox + t_bot * dx
        z_bot = oz + t_bot * dz
        _push_root(roots, valids, t_bot, dy_ok & (x_bot * x_bot + z_bot * z_bot <= 1.0))
    else:
        t0, t1, ok_side = _quadratic(
            dx * dx + dz * dz,
            2.0 * (ox * dx + oz * dz),
            ox * ox + oz * oz - 1.0,
        )
        y0 = oy + t0 * dy
        y1 = oy + t1 * dy
        _push_root(roots, valids, t0, ok_side & (y0 >= -1.0) & (y0 <= 1.0))
        _push_root(roots, valids, t1, ok_side & (y1 >= -1.0) & (y1 <= 1.0))

        a = dx * dx + dy * dy + dz * dz
        for cy in (-1.0, 1.0):
            by = oy - cy
            ts0, ts1, ok_s = _quadratic(
                a,
                2.0 * (ox * dx + by * dy + oz * dz),
                ox * ox + by * by + oz * oz - 1.0,
            )
            _push_root(roots, valids, ts0, ok_s)
            _push_root(roots, valids, ts1, ok_s)

    if len(roots) == 0:
        return inf, ninf
    t_stack = torch.stack(roots, dim=0)  # (K, P)
    valid_stack = torch.stack(valids, dim=0)  # (K, P)
    t_enter = torch.where(valid_stack, t_stack, inf).min(dim=0).values  # (P,)
    t_exit = torch.where(valid_stack, t_stack, ninf).max(dim=0).values  # (P,)
    valid = torch.any(valid_stack, dim=0) & (t_exit >= t_enter) & (t_exit > 0.0)  # (P,)
    return torch.where(valid, t_enter, inf), torch.where(valid, t_exit, ninf)


def _factor_to_1d(
    x: int | float | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    """Convert an input factor to a 1D tensor for batch broadcasting."""
    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype)
        if t.ndim == 0:
            return t.reshape(1)
        if t.ndim == 1:
            return t
        raise ValueError(f"{name} must be scalar or 1D tensor")
    return torch.tensor([float(x)], device=device, dtype=dtype)


@dataclass(frozen=True)
class _RenderContext:
    # Notation: B=batch_size, H=render_h, W=render_w, S=#surfaces.
    batch_size: int
    shape_id: int
    ssaa_scale: int
    base_h: int
    base_w: int
    render_h: int
    render_w: int
    device: torch.device
    dtype: torch.dtype
    size_batch: torch.Tensor  # (B,)
    orientation_batch: torch.Tensor  # (B,)
    floor_hue_batch: torch.Tensor  # (B,)
    wall_hue_batch: torch.Tensor  # (B,)
    object_hue_batch: torch.Tensor  # (B,)
    hue_v: float | torch.Tensor  # scalar tensor () or python float
    shadow_strength: float | torch.Tensor  # scalar tensor () or python float
    lighting_config: LightingConfig
    mesh_resolution_config: MeshResolutionConfig
    camera_config: CameraConfig
    room_config: RoomConfig
    object_config: ObjectConfig
    output_chw: bool


@dataclass(frozen=True)
class _CameraRays:
    camera_position: torch.Tensor  # (B, 3)
    ray_origin_world: torch.Tensor  # (B, 1, 1, 3)
    ray_direction: torch.Tensor  # (B, H, W, 3)


@dataclass(frozen=True)
class _SceneParams:
    floor_rgb: torch.Tensor  # (B, 3)
    wall_rgb: torch.Tensor  # (B, 3)
    object_rgb: torch.Tensor  # (B, 3)
    light_position_world: torch.Tensor  # (B, 3)
    ambient: torch.Tensor  # ()
    diffuse: torch.Tensor  # ()
    shadow_strength_t: torch.Tensor  # ()
    room_floor_y: torch.Tensor  # ()
    room_top_y: torch.Tensor  # ()
    room_min_x: torch.Tensor  # ()
    room_max_x: torch.Tensor  # ()
    room_min_z: torch.Tensor  # ()
    room_max_z: torch.Tensor  # ()
    denominator_eps: torch.Tensor  # ()
    room_surface_normals: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    object_center_world_batch: torch.Tensor  # (B, 3)
    object_scale_batch: torch.Tensor  # (B, 3)
    infinite_t: torch.Tensor  # (B, H, W)
    zero_alpha: torch.Tensor  # (B, H, W)
    background_rgb: torch.Tensor  # (B, H, W, 3)
    room_floor_y_batch: torch.Tensor  # (B,)
    room_top_y_batch: torch.Tensor  # (B,)


@dataclass(frozen=True)
class _Layer:
    t: torch.Tensor  # (B, H, W)
    color: torch.Tensor  # (B, H, W, 3)
    alpha: torch.Tensor  # (B, H, W)


def _prepare_render_context(
    shape: int,
    size: float | torch.Tensor,
    orientation: float | torch.Tensor,
    floor_hue: float | torch.Tensor,
    wall_hue: float | torch.Tensor,
    object_hue: float | torch.Tensor,
    *,
    hue_v: float | torch.Tensor,
    shadow_strength: float | torch.Tensor,
    ssaa_scale: int,
    image_size: int | Tuple[int, int],
    lighting_config: LightingConfig | None,
    mesh_resolution_config: MeshResolutionConfig | None,
    camera_config: CameraConfig | None,
    room_config: RoomConfig | None,
    object_config: ObjectConfig | None,
    output_chw: bool,
) -> _RenderContext:
    """Normalize renderer inputs and static configs into a single context object."""
    input_refs = [size, orientation, floor_hue, wall_hue, object_hue, hue_v]
    floating_tensor_ref = next((x for x in input_refs if torch.is_tensor(x) and torch.is_floating_point(x)), None)
    tensor_ref = next((x for x in input_refs if torch.is_tensor(x)), None)
    reference_tensor = floating_tensor_ref if floating_tensor_ref is not None else tensor_ref
    device = reference_tensor.device if reference_tensor is not None else torch.device("cpu")
    dtype = (
        reference_tensor.dtype
        if reference_tensor is not None and torch.is_floating_point(reference_tensor)
        else torch.float32
    )

    size_batch = _factor_to_1d(size, device=device, dtype=dtype, name="size")
    orientation_batch = _factor_to_1d(orientation, device=device, dtype=dtype, name="orientation")
    floor_hue_batch = _factor_to_1d(floor_hue, device=device, dtype=dtype, name="floor_hue")
    wall_hue_batch = _factor_to_1d(wall_hue, device=device, dtype=dtype, name="wall_hue")
    object_hue_batch = _factor_to_1d(object_hue, device=device, dtype=dtype, name="object_hue")

    batch_lengths = [
        size_batch.shape[0],
        orientation_batch.shape[0],
        floor_hue_batch.shape[0],
        wall_hue_batch.shape[0],
        object_hue_batch.shape[0],
    ]
    batch_size = max(batch_lengths)
    names = ["size", "orientation", "floor_hue", "wall_hue", "object_hue"]
    for name, length in zip(names, batch_lengths):
        if length not in (1, batch_size):
            raise ValueError(f"{name} batch size must be 1 or {batch_size}, got {length}")

    def _expand_to_batch(x: torch.Tensor) -> torch.Tensor:
        return x if x.shape[0] == batch_size else x.expand(batch_size)

    size_batch = _expand_to_batch(size_batch).clamp(0.4, 1.8)
    orientation_batch = _expand_to_batch(orientation_batch)
    floor_hue_batch = _expand_to_batch(floor_hue_batch)
    wall_hue_batch = _expand_to_batch(wall_hue_batch)
    object_hue_batch = _expand_to_batch(object_hue_batch)

    ssaa_scale = int(max(1, ssaa_scale))
    base_h, base_w = _parse_image_size(image_size)
    render_h = base_h * ssaa_scale
    render_w = base_w * ssaa_scale

    return _RenderContext(
        batch_size=batch_size,
        shape_id=max(0, min(3, int(shape))),
        ssaa_scale=ssaa_scale,
        base_h=base_h,
        base_w=base_w,
        render_h=render_h,
        render_w=render_w,
        device=device,
        dtype=dtype,
        size_batch=size_batch,
        orientation_batch=orientation_batch,
        floor_hue_batch=floor_hue_batch,
        wall_hue_batch=wall_hue_batch,
        object_hue_batch=object_hue_batch,
        hue_v=hue_v,
        shadow_strength=shadow_strength,
        lighting_config=lighting_config or LightingConfig(),
        mesh_resolution_config=mesh_resolution_config or MeshResolutionConfig(),
        camera_config=camera_config or CameraConfig(),
        room_config=room_config or RoomConfig(),
        object_config=object_config or ObjectConfig(),
        output_chw=output_chw,
    )


def _build_camera_rays(ctx: _RenderContext) -> _CameraRays:
    """Construct camera position and per-pixel world-space rays."""
    device_key = _device_cache_key(ctx.device)
    yaw_offset_deg_t = _as_torch_scalar(ctx.camera_config.yaw_offset_deg, ctx.device, ctx.dtype)
    theta = 2.0 * torch.pi * torch.remainder(ctx.orientation_batch, 1.0) + (
        yaw_offset_deg_t * (torch.pi / 180.0)
    )

    image_plane_x, image_plane_y = _cached_image_plane(
        ctx.render_h,
        ctx.render_w,
        device_key[0],
        device_key[1],
        ctx.dtype,
    )

    cam_radius = _as_torch_scalar(ctx.camera_config.radius, ctx.device, ctx.dtype)
    cam_height = _as_torch_scalar(ctx.camera_config.height, ctx.device, ctx.dtype)
    target = _as_torch_vec3(ctx.camera_config.target, ctx.device, ctx.dtype)
    camera_position = torch.stack(
        [
            cam_radius * torch.sin(theta),
            torch.full_like(theta, cam_height),
            cam_radius * torch.cos(theta),
        ],
        dim=-1,
    )

    camera_forward = target[None, :] - camera_position
    camera_forward = camera_forward / torch.clamp(torch.linalg.norm(camera_forward, dim=-1, keepdim=True), min=1e-8)
    world_up_hint = _cached_vec3_tensor(device_key[0], device_key[1], ctx.dtype, 0.0, 1.0, 0.0).expand(
        ctx.batch_size, 3
    )
    camera_right = torch.cross(camera_forward, world_up_hint, dim=-1)
    camera_right = camera_right / torch.clamp(torch.linalg.norm(camera_right, dim=-1, keepdim=True), min=1e-8)
    camera_up = torch.cross(camera_right, camera_forward, dim=-1)
    camera_up = camera_up / torch.clamp(torch.linalg.norm(camera_up, dim=-1, keepdim=True), min=1e-8)

    ray_direction = (
        camera_right[:, None, None, :] * image_plane_x[None, :, :, None]
        + camera_up[:, None, None, :] * image_plane_y[None, :, :, None]
        + camera_forward[:, None, None, :]
    )
    ray_direction = ray_direction / torch.clamp(torch.linalg.norm(ray_direction, dim=-1, keepdim=True), min=1e-8)

    return _CameraRays(
        camera_position=camera_position,
        ray_origin_world=camera_position[:, None, None, :],
        ray_direction=ray_direction,
    )


def _build_scene_params(ctx: _RenderContext) -> _SceneParams:
    """Build scene/material/object tensors shared by all render layers."""
    device_key = _device_cache_key(ctx.device)
    hue_value = _as_torch_scalar(ctx.hue_v, ctx.device, ctx.dtype).clamp(0.0, 1.0)
    floor_rgb = _hue_to_rgb_constrained(ctx.floor_hue_batch, hue_value)
    wall_rgb = _hue_to_rgb_constrained(ctx.wall_hue_batch, hue_value)
    object_rgb = _hue_to_rgb_constrained(ctx.object_hue_batch, hue_value)

    yaw_offset_deg_t = _as_torch_scalar(ctx.camera_config.yaw_offset_deg, ctx.device, ctx.dtype)
    theta = 2.0 * torch.pi * torch.remainder(ctx.orientation_batch, 1.0) + (
        yaw_offset_deg_t * (torch.pi / 180.0)
    )
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    zeros = torch.zeros_like(cos_theta)
    ones = torch.ones_like(cos_theta)
    yaw_rotation = torch.stack(
        [
            torch.stack([cos_theta, zeros, sin_theta], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sin_theta, zeros, cos_theta], dim=-1),
        ],
        dim=1,
    )

    light_position_base = _as_torch_vec3(ctx.lighting_config.light_position_world, ctx.device, ctx.dtype)
    light_position_world = yaw_rotation @ light_position_base
    ambient = _as_torch_scalar(ctx.lighting_config.ambient, ctx.device, ctx.dtype).clamp(0.0, 1.0)
    diffuse = _as_torch_scalar(ctx.lighting_config.diffuse, ctx.device, ctx.dtype).clamp(0.0, 2.0)
    shadow_strength_t = _as_torch_scalar(ctx.shadow_strength, ctx.device, ctx.dtype).clamp(0.0, 1.0)

    room_floor_y = _as_torch_scalar(ctx.room_config.floor_y, ctx.device, ctx.dtype)
    room_top_y = _as_torch_scalar(ctx.room_config.wall_top_y, ctx.device, ctx.dtype)
    room_half_width = _as_torch_scalar(ctx.room_config.half_width, ctx.device, ctx.dtype)
    room_half_depth = _as_torch_scalar(ctx.room_config.half_depth, ctx.device, ctx.dtype)
    room_min_x, room_max_x = -room_half_width, room_half_width
    room_min_z, room_max_z = -room_half_depth, room_half_depth

    denominator_eps = _cached_scalar_tensor(device_key[0], device_key[1], ctx.dtype, 1e-8)
    sky_color = _cached_vec3_tensor(device_key[0], device_key[1], ctx.dtype, 0.72, 0.88, 1.0)
    room_surface_normals = (
        _cached_vec3_tensor(device_key[0], device_key[1], ctx.dtype, 0.0, 1.0, 0.0),
        _cached_vec3_tensor(device_key[0], device_key[1], ctx.dtype, 0.0, 0.0, 1.0),
        _cached_vec3_tensor(device_key[0], device_key[1], ctx.dtype, 0.0, 0.0, -1.0),
        _cached_vec3_tensor(device_key[0], device_key[1], ctx.dtype, 1.0, 0.0, 0.0),
        _cached_vec3_tensor(device_key[0], device_key[1], ctx.dtype, -1.0, 0.0, 0.0),
    )

    object_base_scale = _as_torch_shape_vec3(ctx.object_config.base_scales, ctx.shape_id, ctx.device, ctx.dtype)
    object_base_scale = object_base_scale * _as_torch_scalar(
        ctx.object_config.global_scale_multiplier, ctx.device, ctx.dtype
    )
    object_scale_batch = object_base_scale[None, :] * ctx.size_batch[:, None]

    object_height_extent = _cached_scalar_tensor(
        device_key[0], device_key[1], ctx.dtype, 2.0 if ctx.shape_id == 3 else 1.0
    )
    ground_clearance = _as_torch_shape_scalar(
        ctx.object_config.ground_clearance_by_shape, ctx.shape_id, ctx.device, ctx.dtype
    )
    object_center_y = room_floor_y + ground_clearance + object_height_extent * object_scale_batch[:, 1]
    object_center_world_batch = torch.stack(
        [
            _as_torch_scalar(ctx.object_config.center_x, ctx.device, ctx.dtype).expand(ctx.batch_size),
            object_center_y,
            _as_torch_scalar(ctx.object_config.center_z, ctx.device, ctx.dtype).expand(ctx.batch_size),
        ],
        dim=-1,
    )

    infinite_t = torch.full((ctx.batch_size, ctx.render_h, ctx.render_w), float("inf"), device=ctx.device, dtype=ctx.dtype)
    zero_alpha = torch.zeros((ctx.batch_size, ctx.render_h, ctx.render_w), device=ctx.device, dtype=ctx.dtype)
    background_rgb = sky_color[None, None, None, :].expand(ctx.batch_size, ctx.render_h, ctx.render_w, 3).clone()
    room_floor_y_batch = room_floor_y.expand(ctx.batch_size)
    room_top_y_batch = room_top_y.expand(ctx.batch_size)

    return _SceneParams(
        floor_rgb=floor_rgb,
        wall_rgb=wall_rgb,
        object_rgb=object_rgb,
        light_position_world=light_position_world,
        ambient=ambient,
        diffuse=diffuse,
        shadow_strength_t=shadow_strength_t,
        room_floor_y=room_floor_y,
        room_top_y=room_top_y,
        room_min_x=room_min_x,
        room_max_x=room_max_x,
        room_min_z=room_min_z,
        room_max_z=room_max_z,
        denominator_eps=denominator_eps,
        room_surface_normals=room_surface_normals,
        object_center_world_batch=object_center_world_batch,
        object_scale_batch=object_scale_batch,
        infinite_t=infinite_t,
        zero_alpha=zero_alpha,
        background_rgb=background_rgb,
        room_floor_y_batch=room_floor_y_batch,
        room_top_y_batch=room_top_y_batch,
    )


def _plane_hit(
    axis: int,
    coord: torch.Tensor,
    *,
    ctx: _RenderContext,
    rays: _CameraRays,
    scene: _SceneParams,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect world-space rays with one axis-aligned plane."""
    plane_coord_batch = coord if coord.ndim > 0 else coord.expand(ctx.batch_size)
    ray_denominator_eps = 1e-8
    ray_axis_denominator = rays.ray_direction[..., axis]
    ray_axis_denominator_safe = torch.where(
        torch.abs(ray_axis_denominator) > ray_denominator_eps,
        ray_axis_denominator,
        torch.where(ray_axis_denominator >= 0.0, scene.denominator_eps, -scene.denominator_eps),
    )
    t_hit = (plane_coord_batch[:, None, None] - rays.ray_origin_world[..., axis]) / ray_axis_denominator_safe
    hit_point_world = rays.ray_origin_world + rays.ray_direction * t_hit[..., None]
    is_valid_hit = (torch.abs(ray_axis_denominator) > ray_denominator_eps) & (t_hit > 1e-6)
    return t_hit, hit_point_world, is_valid_hit


def _shade(
    base_rgb: torch.Tensor,
    normal: torch.Tensor,
    points_world: torch.Tensor,
    *,
    scene: _SceneParams,
) -> torch.Tensor:
    """Apply Lambertian shading for points on a surface."""
    light_dir = scene.light_position_world[:, None, None, :] - points_world
    light_dir = light_dir / torch.clamp(torch.linalg.norm(light_dir, dim=-1, keepdim=True), min=1e-8)
    surface_normal = normal if normal.ndim == 4 else normal[None, None, None, :]
    lambert_term = torch.clamp(torch.sum(light_dir * surface_normal, dim=-1), min=0.0)
    return torch.clamp(base_rgb[:, None, None, :] * (scene.ambient + scene.diffuse * lambert_term)[..., None], 0.0, 1.0)


def _render_floor_layer(
    *,
    ctx: _RenderContext,
    rays: _CameraRays,
    scene: _SceneParams,
) -> _Layer:
    """Render floor color and alpha including soft object shadow."""
    t_floor, floor_hit_world, is_valid_floor = _plane_hit(axis=1, coord=scene.room_floor_y, ctx=ctx, rays=rays, scene=scene)
    floor_hit_world_safe = torch.where(is_valid_floor[..., None], floor_hit_world, rays.ray_origin_world)
    floor_normal = scene.room_surface_normals[0]

    light_vector_floor = scene.light_position_world[:, None, None, :] - floor_hit_world_safe
    light_distance_floor = torch.linalg.norm(light_vector_floor, dim=-1)
    light_direction_floor = light_vector_floor / torch.clamp(light_distance_floor[..., None], min=1e-8)

    t_light_enter, t_light_exit = _ray_object_hit_interval_t(
        ray_origin_world=floor_hit_world_safe,
        ray_dir_world=light_direction_floor,
        shape_id=ctx.shape_id,
        center_world=scene.object_center_world_batch[:, None, None, :],
        scale_xyz=scene.object_scale_batch[:, None, None, :],
        t_far=100.0,
    )
    shadow_segment_raw = torch.minimum(t_light_exit, light_distance_floor) - _smooth_positive(t_light_enter, delta=1e-3)
    shadow_segment = _smooth_positive(shadow_segment_raw, delta=1e-3)
    shadow_occlusion = _occlusion_from_segment_length(shadow_segment, sharpness=4.0)
    shadow_occlusion = torch.where(is_valid_floor, shadow_occlusion, scene.zero_alpha)

    ndotl_floor = torch.clamp(
        torch.sum(light_direction_floor * floor_normal[None, None, None, :], dim=-1),
        min=0.0,
    )
    floor_lit = scene.ambient + scene.diffuse * ndotl_floor * (1.0 - scene.shadow_strength_t * shadow_occlusion)
    floor_color = torch.clamp(scene.floor_rgb[:, None, None, :] * floor_lit[..., None], 0.0, 1.0)

    floor_edge_softness = 0.08
    alpha_floor = torch.ones_like(t_floor)
    if ctx.room_config.draw_left_wall:
        alpha_floor = alpha_floor * torch.sigmoid((floor_hit_world_safe[..., 0] - scene.room_min_x) / floor_edge_softness)
    if ctx.room_config.draw_right_wall:
        alpha_floor = alpha_floor * torch.sigmoid((scene.room_max_x - floor_hit_world_safe[..., 0]) / floor_edge_softness)
    if ctx.room_config.draw_back_wall:
        alpha_floor = alpha_floor * torch.sigmoid((floor_hit_world_safe[..., 2] - scene.room_min_z) / floor_edge_softness)
    if ctx.room_config.draw_front_wall:
        alpha_floor = alpha_floor * torch.sigmoid((scene.room_max_z - floor_hit_world_safe[..., 2]) / floor_edge_softness)
    alpha_floor = torch.clamp(alpha_floor, 0.0, 1.0)
    alpha_floor = torch.where(is_valid_floor, alpha_floor, scene.zero_alpha)

    return _Layer(
        t=torch.where(is_valid_floor, t_floor, scene.infinite_t),
        color=torch.where(is_valid_floor[..., None], floor_color, torch.zeros_like(floor_color)),
        alpha=alpha_floor,
    )


def _render_wall_layers(
    *,
    ctx: _RenderContext,
    rays: _CameraRays,
    scene: _SceneParams,
) -> list[_Layer]:
    """Render enabled room walls as independent compositing layers."""
    layers: list[_Layer] = []

    def _make_wall_layer(enabled: bool, axis: int, coord: torch.Tensor, normal: torch.Tensor) -> None:
        if not enabled:
            return
        t_wall, wall_hit_world, is_valid_wall = _plane_hit(axis=axis, coord=coord, ctx=ctx, rays=rays, scene=scene)
        wall_hit_world_safe = torch.where(is_valid_wall[..., None], wall_hit_world, rays.ray_origin_world)
        wall_color = _shade(scene.wall_rgb, normal, wall_hit_world_safe, scene=scene)

        wall_soft_y = 0.08
        wall_soft_side = 0.08
        alpha_floor_edge = torch.sigmoid(
            (wall_hit_world_safe[..., 1] - scene.room_floor_y_batch[:, None, None]) / wall_soft_y
        )
        if axis == 2:
            alpha_wall_span = (
                torch.sigmoid((wall_hit_world_safe[..., 0] - scene.room_min_x) / wall_soft_side)
                * torch.sigmoid((scene.room_max_x - wall_hit_world_safe[..., 0]) / wall_soft_side)
            )
        else:
            alpha_wall_span = (
                torch.sigmoid((wall_hit_world_safe[..., 2] - scene.room_min_z) / wall_soft_side)
                * torch.sigmoid((scene.room_max_z - wall_hit_world_safe[..., 2]) / wall_soft_side)
            )
        alpha_top_edge = torch.sigmoid((scene.room_top_y_batch[:, None, None] - wall_hit_world_safe[..., 1]) / wall_soft_y)
        alpha_wall = torch.clamp(alpha_floor_edge * alpha_wall_span * alpha_top_edge, 0.0, 1.0)
        alpha_wall = torch.where(is_valid_wall, alpha_wall, scene.zero_alpha)

        layers.append(
            _Layer(
                t=torch.where(is_valid_wall, t_wall, scene.infinite_t),
                color=torch.where(is_valid_wall[..., None], wall_color, torch.zeros_like(wall_color)),
                alpha=alpha_wall,
            )
        )

    _make_wall_layer(ctx.room_config.draw_back_wall, axis=2, coord=scene.room_min_z, normal=scene.room_surface_normals[1])
    _make_wall_layer(ctx.room_config.draw_front_wall, axis=2, coord=scene.room_max_z, normal=scene.room_surface_normals[2])
    _make_wall_layer(ctx.room_config.draw_left_wall, axis=0, coord=scene.room_min_x, normal=scene.room_surface_normals[3])
    _make_wall_layer(ctx.room_config.draw_right_wall, axis=0, coord=scene.room_max_x, normal=scene.room_surface_normals[4])
    return layers


def _render_object_layer(
    *,
    ctx: _RenderContext,
    rays: _CameraRays,
    scene: _SceneParams,
) -> _Layer:
    """Render the object layer using analytic entry/exit and normal formulas."""
    t_obj_enter, t_obj_exit = _ray_object_hit_interval_t(
        ray_origin_world=rays.ray_origin_world,
        ray_dir_world=rays.ray_direction,
        shape_id=ctx.shape_id,
        center_world=scene.object_center_world_batch[:, None, None, :],
        scale_xyz=scene.object_scale_batch[:, None, None, :],
        t_far=100.0,
    )
    is_valid_object = torch.isfinite(t_obj_enter) & (t_obj_exit > 0.0)
    t_obj_hit = torch.where(is_valid_object, torch.clamp(t_obj_enter, min=1e-5), scene.infinite_t)
    t_obj_safe = torch.where(is_valid_object, torch.clamp(t_obj_enter, min=1e-5), torch.zeros_like(t_obj_enter))
    p_obj = rays.ray_origin_world + rays.ray_direction * t_obj_safe[..., None]

    n_obj = _normal_object_analytic(
        p_obj,
        ctx.shape_id,
        scene.object_center_world_batch[:, None, None, :],
        scene.object_scale_batch[:, None, None, :],
    )
    obj_color = _shade(scene.object_rgb, n_obj, p_obj, scene=scene)
    obj_color = torch.where(is_valid_object[..., None], obj_color, torch.zeros_like(obj_color))

    view_segment_raw = t_obj_exit - _smooth_positive(t_obj_enter, delta=1e-3)
    view_segment = _smooth_positive(view_segment_raw, delta=1e-3)
    alpha_obj = _occlusion_from_segment_length(view_segment, sharpness=5.0)
    alpha_obj = torch.where(is_valid_object, torch.clamp(alpha_obj, 0.0, 1.0), scene.zero_alpha)

    return _Layer(t=t_obj_hit, color=obj_color, alpha=alpha_obj)


def _composite_layers(
    layers: Sequence[_Layer],
    *,
    ctx: _RenderContext,
    scene: _SceneParams,
) -> torch.Tensor:
    """Front-to-back alpha compositing after per-pixel depth sorting."""
    layer_t_stack = torch.stack([layer.t for layer in layers], dim=0)
    layer_color_stack = torch.stack([layer.color for layer in layers], dim=0)
    layer_alpha_stack = torch.stack([layer.alpha for layer in layers], dim=0)

    # Sort layers by depth at each pixel once, then composite in sorted order.
    sorted_idx = torch.argsort(layer_t_stack, dim=0)  # (S, B, H, W)
    sorted_t = torch.gather(layer_t_stack, 0, sorted_idx)
    sorted_alpha = torch.gather(layer_alpha_stack, 0, sorted_idx)
    sorted_color = torch.gather(
        layer_color_stack,
        0,
        sorted_idx[..., None].expand(-1, -1, -1, -1, 3),
    )
    sorted_alpha = torch.where(torch.isfinite(sorted_t), sorted_alpha, torch.zeros_like(sorted_alpha))

    remaining_transmittance = torch.ones((ctx.batch_size, ctx.render_h, ctx.render_w, 1), device=ctx.device, dtype=ctx.dtype)
    accumulated_rgb = torch.zeros_like(scene.background_rgb)
    for depth_order in range(sorted_t.shape[0]):
        alpha_i = sorted_alpha[depth_order]  # (B, H, W)
        color_i = sorted_color[depth_order]  # (B, H, W, 3)
        accumulated_rgb = accumulated_rgb + remaining_transmittance * alpha_i[..., None] * color_i
        remaining_transmittance = remaining_transmittance * (1.0 - alpha_i[..., None])

    return accumulated_rgb + remaining_transmittance * scene.background_rgb


def _finalize_render_output(out: torch.Tensor, *, ctx: _RenderContext) -> torch.Tensor:
    """Apply SSAA downsampling and requested channel/batch layout."""
    if ctx.ssaa_scale > 1:
        out = _downsample_mean(out, ctx.ssaa_scale)
    if ctx.output_chw:
        out = out.permute(0, 3, 1, 2)
    if ctx.batch_size == 1:
        return out[0]
    return out


def render_3dshapes_image(
    shape: int,
    size: float | torch.Tensor,
    orientation: float | torch.Tensor,
    floor_hue: float | torch.Tensor,
    wall_hue: float | torch.Tensor,
    object_hue: float | torch.Tensor,
    *,
    hue_v: float | torch.Tensor = 0.9,
    shadow_strength: float | torch.Tensor = 10.0,
    ssaa_scale: int = 4,
    image_size: int | Tuple[int, int] = 64,
    lighting_config: LightingConfig | None = None,
    mesh_resolution_config: MeshResolutionConfig | None = None,
    camera_config: CameraConfig | None = None,
    room_config: RoomConfig | None = None,
    object_config: ObjectConfig | None = None,
    output_chw: bool = True,
) -> torch.Tensor:
    """Differentiable renderer for a fixed shape id.

    `shape` is treated as an integer (0..3). For mixed-shape batches, use
    `render_3dshapes_image_grouped`.
    """
    ctx = _prepare_render_context(
        shape=shape,
        size=size,
        orientation=orientation,
        floor_hue=floor_hue,
        wall_hue=wall_hue,
        object_hue=object_hue,
        hue_v=hue_v,
        shadow_strength=shadow_strength,
        ssaa_scale=ssaa_scale,
        image_size=image_size,
        lighting_config=lighting_config,
        mesh_resolution_config=mesh_resolution_config,
        camera_config=camera_config,
        room_config=room_config,
        object_config=object_config,
        output_chw=output_chw,
    )
    rays = _build_camera_rays(ctx)
    scene = _build_scene_params(ctx)

    layers: list[_Layer] = []
    layers.append(_render_floor_layer(ctx=ctx, rays=rays, scene=scene))
    layers.extend(_render_wall_layers(ctx=ctx, rays=rays, scene=scene))
    layers.append(_render_object_layer(ctx=ctx, rays=rays, scene=scene))

    out = _composite_layers(layers, ctx=ctx, scene=scene)
    return _finalize_render_output(out, ctx=ctx)

class Differentiable3Dshapes(nn.Module):
    """nn.Module wrapper around `render_3dshapes_image`.

    Main factors are provided to `forward`; all other renderer settings are fixed at init.
    """

    def __init__(
        self,
        *,
        hue_v: float | torch.Tensor = 0.9,
        shadow_strength: float | torch.Tensor = 1.0,
        ssaa_scale: int = 1,
        image_size: int | Tuple[int, int] = 64,
        lighting_config: LightingConfig | None = None,
        mesh_resolution_config: MeshResolutionConfig | None = None,
        camera_config: CameraConfig | None = None,
        room_config: RoomConfig | None = None,
        object_config: ObjectConfig | None = None,
        output_chw: bool = True,
    ) -> None:
        """Store fixed renderer settings used by `forward`."""
        super().__init__()
        self.hue_v = hue_v
        self.shadow_strength = shadow_strength
        self.ssaa_scale = ssaa_scale
        self.image_size = image_size
        self.lighting_config = lighting_config
        self.mesh_resolution_config = mesh_resolution_config
        self.camera_config = camera_config
        self.room_config = room_config
        self.object_config = object_config
        self.output_chw = output_chw

    def forward(
        self,
        shape: int | torch.Tensor,
        size: float | torch.Tensor,
        orientation: float | torch.Tensor,
        floor_hue: float | torch.Tensor,
        wall_hue: float | torch.Tensor,
        object_hue: float | torch.Tensor,
        return_grad: bool = False,
    ) -> torch.Tensor | tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Render images (and optionally Jacobians) for batched latent factors."""
        kwargs = dict(
            hue_v=self.hue_v,
            shadow_strength=self.shadow_strength,
            ssaa_scale=self.ssaa_scale,
            image_size=self.image_size,
            lighting_config=self.lighting_config,
            mesh_resolution_config=self.mesh_resolution_config,
            camera_config=self.camera_config,
            room_config=self.room_config,
            object_config=self.object_config,
            output_chw=self.output_chw,
        )

        refs = [shape, size, orientation, floor_hue, wall_hue, object_hue, self.hue_v]
        float_ref = next((x for x in refs if torch.is_tensor(x) and torch.is_floating_point(x)), None)
        any_ref = next((x for x in refs if torch.is_tensor(x)), None)
        t_ref = float_ref if float_ref is not None else any_ref
        device = t_ref.device if t_ref is not None else torch.device("cpu")
        dtype = t_ref.dtype if t_ref is not None and torch.is_floating_point(t_ref) else torch.float32

        shape_b = _factor_to_1d(shape, device=device, dtype=dtype, name="shape")
        size_b = _factor_to_1d(size, device=device, dtype=dtype, name="size")
        ori_b = _factor_to_1d(orientation, device=device, dtype=dtype, name="orientation")
        floor_b = _factor_to_1d(floor_hue, device=device, dtype=dtype, name="floor_hue")
        wall_b = _factor_to_1d(wall_hue, device=device, dtype=dtype, name="wall_hue")
        obj_b = _factor_to_1d(object_hue, device=device, dtype=dtype, name="object_hue")

        lengths = [shape_b.shape[0], size_b.shape[0], ori_b.shape[0], floor_b.shape[0], wall_b.shape[0], obj_b.shape[0]]
        bsz = max(lengths)
        names = ["shape", "size", "orientation", "floor_hue", "wall_hue", "object_hue"]
        for n, ln in zip(names, lengths):
            if ln not in (1, bsz):
                raise ValueError(f"{n} batch size must be 1 or {bsz}, got {ln}")

        def expand1(x: torch.Tensor) -> torch.Tensor:
            """Broadcast a length-1 factor tensor to the current batch size."""
            return x if x.shape[0] == bsz else x.expand(bsz)

        shape_ids = torch.clamp(expand1(shape_b).to(torch.int64), 0, 3)
        size_b = expand1(size_b)
        ori_b = expand1(ori_b)
        floor_b = expand1(floor_b)
        wall_b = expand1(wall_b)
        obj_b = expand1(obj_b)

        if return_grad:
            # Group by discrete shape id so each group can use fixed-shape jacfwd.
            jac_out: list[torch.Tensor] | None = None
            img_out: torch.Tensor | None = None
            for sid in range(4):
                idx = torch.nonzero(shape_ids == sid, as_tuple=False).reshape(-1)
                if idx.numel() == 0:
                    continue

                def render_single_fixed_shape(
                    size_i: torch.Tensor,
                    orientation_i: torch.Tensor,
                    floor_hue_i: torch.Tensor,
                    wall_hue_i: torch.Tensor,
                    object_hue_i: torch.Tensor,
                ):
                    """Helper for jacfwd: returns `(image, aux_image)` for has_aux."""
                    image = render_3dshapes_image(
                        shape=sid,
                        size=size_i.unsqueeze(0),
                        orientation=orientation_i.unsqueeze(0),
                        floor_hue=floor_hue_i.unsqueeze(0),
                        wall_hue=wall_hue_i.unsqueeze(0),
                        object_hue=object_hue_i.unsqueeze(0),
                        **kwargs,
                    )
                    return image.squeeze(0), image.squeeze(0)

                renderer_with_grad = torch.func.vmap(
                    torch.func.jacfwd(
                        render_single_fixed_shape,
                        argnums=(0, 1, 2, 3, 4),
                        has_aux=True,
                    )
                )
                jac_g, img_g = renderer_with_grad(size_b[idx], ori_b[idx], floor_b[idx], wall_b[idx], obj_b[idx])

                if jac_out is None or img_out is None:
                    img_out = torch.empty((bsz,) + tuple(img_g.shape[1:]), device=img_g.device, dtype=img_g.dtype)
                    jac_out = [
                        torch.empty((bsz,) + tuple(j.shape[1:]), device=j.device, dtype=j.dtype)
                        for j in jac_g
                    ]
                img_out[idx] = img_g
                for k in range(5):
                    jac_out[k][idx] = jac_g[k]

            if jac_out is None or img_out is None:
                raise RuntimeError("No valid shape ids in input.")
            return tuple(jac_out), img_out

        if bsz == 1:
            sid = int(shape_ids[0].item())
            return render_3dshapes_image(
                shape=sid,
                size=size_b,
                orientation=ori_b,
                floor_hue=floor_b,
                wall_hue=wall_b,
                object_hue=obj_b,
                **kwargs,
            )

        out: torch.Tensor | None = None
        for sid in range(4):
            # Render each shape group once, then scatter back into original batch order.
            idx = torch.nonzero(shape_ids == sid, as_tuple=False).reshape(-1)
            if idx.numel() == 0:
                continue
            out_g = render_3dshapes_image(
                shape=sid,
                size=size_b[idx],
                orientation=ori_b[idx],
                floor_hue=floor_b[idx],
                wall_hue=wall_b[idx],
                object_hue=obj_b[idx],
                **kwargs,
            )
            if out_g.ndim == 3:
                out_g = out_g.unsqueeze(0)
            if out is None:
                out = torch.empty((bsz,) + tuple(out_g.shape[1:]), device=out_g.device, dtype=out_g.dtype)
            out[idx] = out_g

        if out is None:
            raise RuntimeError("No valid shape ids in input.")
        return out


__all__ = [
    "CameraConfig",
    "RoomConfig",
    "ObjectConfig",
    "LightingConfig",
    "MeshResolutionConfig",
    "render_3dshapes_image",
    "Differentiable3Dshapes",
]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Main settings + visualization GIF (image | differential image).
    light_cfg = LightingConfig(
        light_position_world=(-7, 30, -2),  # 光源位置 (world座標)
        ambient=0.55,
        diffuse=0.55,
    )
    mesh_res_cfg = MeshResolutionConfig(
        cylinder_radial_steps=64,
        sphere_lat_steps=36,
        sphere_lon_steps=72,
        capsule_radial_steps=56,
        capsule_hemi_steps=18,
    )
    cam_cfg = CameraConfig(
        radius=6.0,           # カメラと原点の距離
        height=1.8,           # カメラ高さ
        target=(0.0, 1.0, 0.0),  # 注視点
        yaw_offset_deg=0.0,   # 全体の向きオフセット
    )
    room_cfg = RoomConfig(
        floor_y=-1.2,
        wall_top_y=5.0,       # 壁高さ
        half_width=10.0,       # 左右の広さ
        half_depth=10.0,       # 前後の広さ
        draw_back_wall=True,
        draw_left_wall=True,
        draw_right_wall=True,
        draw_front_wall=True,  # Trueにすると手前壁も描画
    )
    obj_cfg = ObjectConfig(
        # shape 0..3 のベースサイズ
        base_scales=((1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0), (1.0, 1.0, 1.0)),
        ground_clearance_by_shape=(0.0, 0.0, 0.0, 0.0),  # auto_grounding時の床からの余白
        center_x=0.0,
        center_z=0.0,
        global_scale_multiplier=1.0,
        use_auto_grounding=True,
    )

    renderer = Differentiable3Dshapes(
        hue_v=0.9,
        shadow_strength=0.8,
        ssaa_scale=1,
        image_size=64,
        lighting_config=light_cfg,
        mesh_resolution_config=mesh_res_cfg,
        camera_config=cam_cfg,
        room_config=room_cfg,
        object_config=obj_cfg,
        output_chw=True,
    )
    renderer = renderer.to(device)

    out_dir = Path("samples")
    out_dir.mkdir(exist_ok=True)
    main_params = dict(
        shape=2,
        size=1.0,
        orientation=0.0,
        floor_hue=0.0,
        wall_hue=0.33,
        object_hue=0.66,
    )
    fixed_shape_id = int(main_params["shape"])

    # Build GIF frames:
    # rows = orientation / size / floor_hue / wall_hue / object_hue, cols = image | differential.
    n_frames = 128
    orientation_vals = np.linspace(0.0, 1.0, n_frames, endpoint=False, dtype=np.float32)
    size_vals = np.linspace(0.7, 1.5, n_frames, endpoint=True, dtype=np.float32)
    color_vals = np.linspace(0.0, 1.0, n_frames, endpoint=False, dtype=np.float32)

    frame_images: list[Image.Image] = []
    factor_specs = [
        ("orientation", torch.from_numpy(orientation_vals).to(device=device), 1),
        ("size", torch.from_numpy(size_vals).to(device=device), 0),
        ("floor_hue", torch.from_numpy(color_vals).to(device=device), 2),
        ("wall_hue", torch.from_numpy(color_vals).to(device=device), 3),
        ("object_hue", torch.from_numpy(color_vals).to(device=device), 4),
    ]

    base_shape = torch.full((n_frames,), float(main_params["shape"]), dtype=torch.float32, device=device)
    base_size = torch.full((n_frames,), float(main_params["size"]), dtype=torch.float32, device=device)
    base_ori = torch.full((n_frames,), float(main_params["orientation"]), dtype=torch.float32, device=device)
    base_floor = torch.full((n_frames,), float(main_params["floor_hue"]), dtype=torch.float32, device=device)
    base_wall = torch.full((n_frames,), float(main_params["wall_hue"]), dtype=torch.float32, device=device)
    base_obj = torch.full((n_frames,), float(main_params["object_hue"]), dtype=torch.float32, device=device)

    chunk_size = 128

    # row_frames_by_factor[k][t] is the k-th factor row at time t.
    row_frames_by_factor: list[list[np.ndarray]] = []
    for key, vals_t, jac_idx in factor_specs:
        shape_t = base_shape.clone()
        size_t = base_size.clone()
        ori_t = base_ori.clone()
        floor_t = base_floor.clone()
        wall_t = base_wall.clone()
        obj_t = base_obj.clone()
        if key == "orientation":
            ori_t = vals_t
        elif key == "size":
            size_t = vals_t
        elif key == "floor_hue":
            floor_t = vals_t
        elif key == "wall_hue":
            wall_t = vals_t
        elif key == "object_hue":
            obj_t = vals_t

        factor_rows: list[np.ndarray] = []
        for st in range(0, n_frames, chunk_size):
            ed = min(st + chunk_size, n_frames)
            jac, img = renderer.forward(
                fixed_shape_id, 
                size_t[st:ed], 
                ori_t[st:ed], 
                floor_t[st:ed], 
                wall_t[st:ed], 
                obj_t[st:ed], 
                return_grad=True
            )
            img_np = img.detach().cpu().numpy()  # (Tc,3,H,W)
            diff_np = jac[jac_idx].detach().cpu().numpy()  # (Tc,3,H,W)

            for t in range(ed - st):
                img_hwc = np.transpose(img_np[t], (1, 2, 0))
                diff_hwc = np.transpose(diff_np[t], (1, 2, 0))
                scale = float(np.percentile(np.abs(diff_hwc), 99.5))
                if scale < 1e-8:
                    scale = 1.0
                diff_vis = np.clip(0.5 + 0.5 * (diff_hwc / scale), 0.0, 1.0).astype(np.float32)
                factor_rows.append(np.concatenate([img_hwc, diff_vis], axis=1))
        row_frames_by_factor.append(factor_rows)
        print(f"{key}: batched {n_frames} frames done (chunk={chunk_size})")

    for t in range(n_frames):
        frame = np.concatenate([row_frames_by_factor[k][t] for k in range(len(row_frames_by_factor))], axis=0)
        frame_u8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        frame_images.append(Image.fromarray(frame_u8))
        print(f"frame {t + 1}/{n_frames} assembled")

    gif_path = out_dir / "main_factors_and_differentials.gif"
    frame_images[0].save(
        gif_path,
        save_all=True,
        append_images=frame_images[1:],
        duration=140,
        loop=0,
        optimize=False,
    )
    print("saved:", gif_path)
