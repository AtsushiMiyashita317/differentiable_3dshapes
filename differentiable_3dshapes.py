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
    return device.type, -1 if device.index is None else int(device.index)


def _device_from_cache_key(device_type: str, device_index: int) -> torch.device:
    return torch.device(device_type) if device_index < 0 else torch.device(device_type, device_index)


@lru_cache(maxsize=128)
def _cached_scalar_tensor(device_type: str, device_index: int, dtype: torch.dtype, value: float) -> torch.Tensor:
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
    device = _device_from_cache_key(device_type, device_index)
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5 - (w - 1) * 0.5) / (48.0 * (w / 64.0))
    ys = -((torch.arange(h, device=device, dtype=dtype) + 0.5 - (h - 1) * 0.5) / (48.0 * (w / 64.0)))
    return torch.meshgrid(xs, ys, indexing="xy")

def _downsample_mean(
    image_hwc: np.ndarray | torch.Tensor,
    factor: int,
) -> np.ndarray | torch.Tensor:
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
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(float(x), device=device, dtype=dtype)


def _as_torch_vec3(
    x: Sequence[float] | torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
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
    dtype = ray_dir_world.dtype
    device = ray_dir_world.device
    inf = torch.full_like(ray_dir_world[..., 0], float("inf"))
    ninf = torch.full_like(ray_dir_world[..., 0], -float("inf"))
    eps = torch.as_tensor(1e-10, device=device, dtype=dtype)
    sqrt_eps = torch.as_tensor(1e-12, device=device, dtype=dtype)

    scale_xyz_safe = torch.clamp(scale_xyz, min=1e-8)
    ro = (ray_origin_world - center_world) / scale_xyz_safe
    rd = ray_dir_world / scale_xyz_safe

    ox, oy, oz = ro[..., 0], ro[..., 1], ro[..., 2]
    dx, dy, dz = rd[..., 0], rd[..., 1], rd[..., 2]

    if shape_id == 0:
        parallel = torch.abs(rd) < eps
        outside_parallel = parallel & ((ro < -1.0) | (ro > 1.0))
        no_hit_parallel = outside_parallel.any(dim=-1)
        rd_safe = torch.where(torch.abs(rd) > eps, rd, torch.where(rd >= 0.0, eps, -eps))
        inv_rd = 1.0 / rd_safe
        t0 = (-1.0 - ro) * inv_rd
        t1 = (1.0 - ro) * inv_rd
        t_near_axis = torch.minimum(t0, t1)
        t_far_axis = torch.maximum(t0, t1)
        t_near_axis = torch.where(parallel, ninf[..., None], t_near_axis)
        t_far_axis = torch.where(parallel, inf[..., None], t_far_axis)
        t_enter = t_near_axis.max(dim=-1).values
        t_exit = t_far_axis.min(dim=-1).values
        valid = (~no_hit_parallel) & (t_exit >= t_enter) & (t_exit > 0.0)
        return torch.where(valid, t_enter, inf), torch.where(valid, t_exit, ninf)

    def _push_root(
        roots: list[torch.Tensor],
        valids: list[torch.Tensor],
        t: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        roots.append(torch.nan_to_num(t, nan=float(t_far), posinf=float(t_far), neginf=-float(t_far)))
        valids.append(valid)

    roots: list[torch.Tensor] = []
    valids: list[torch.Tensor] = []

    def _quadratic(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        disc = b * b - 4.0 * a * c
        den = 2.0 * a
        den_ok = torch.abs(den) > eps
        root_ok = (disc >= 0.0) & den_ok
        den_safe = torch.where(den_ok, den, torch.ones_like(den))
        # Avoid infinite/unstable gradients around tangential hits (disc ~= 0).
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=sqrt_eps))
        t0 = (-b - sqrt_disc) / den_safe
        t1 = (-b + sqrt_disc) / den_safe
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
    t_stack = torch.stack(roots, dim=0)
    valid_stack = torch.stack(valids, dim=0)
    t_enter = torch.where(valid_stack, t_stack, inf).min(dim=0).values
    t_exit = torch.where(valid_stack, t_stack, ninf).max(dim=0).values
    valid = torch.any(valid_stack, dim=0) & (t_exit >= t_enter) & (t_exit > 0.0)
    return torch.where(valid, t_enter, inf), torch.where(valid, t_exit, ninf)


def _factor_to_1d(
    x: int | float | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype)
        if t.ndim == 0:
            return t.reshape(1)
        if t.ndim == 1:
            return t
        raise ValueError(f"{name} must be scalar or 1D tensor")
    return torch.tensor([float(x)], device=device, dtype=dtype)


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
    refs = [size, orientation, floor_hue, wall_hue, object_hue, hue_v]
    float_ref = next((x for x in refs if torch.is_tensor(x) and torch.is_floating_point(x)), None)
    any_ref = next((x for x in refs if torch.is_tensor(x)), None)
    t_ref = float_ref if float_ref is not None else any_ref
    device = t_ref.device if t_ref is not None else torch.device("cpu")
    dtype = t_ref.dtype if t_ref is not None and torch.is_floating_point(t_ref) else torch.float32
    dkey = _device_cache_key(device)

    size_b = _factor_to_1d(size, device=device, dtype=dtype, name="size")
    ori_b = _factor_to_1d(orientation, device=device, dtype=dtype, name="orientation")
    floor_b = _factor_to_1d(floor_hue, device=device, dtype=dtype, name="floor_hue")
    wall_b = _factor_to_1d(wall_hue, device=device, dtype=dtype, name="wall_hue")
    obj_b = _factor_to_1d(object_hue, device=device, dtype=dtype, name="object_hue")

    lengths = [size_b.shape[0], ori_b.shape[0], floor_b.shape[0], wall_b.shape[0], obj_b.shape[0]]
    bsz = max(lengths)
    names = ["size", "orientation", "floor_hue", "wall_hue", "object_hue"]
    for n, ln in zip(names, lengths):
        if ln not in (1, bsz):
            raise ValueError(f"{n} batch size must be 1 or {bsz}, got {ln}")

    def expand1(x: torch.Tensor) -> torch.Tensor:
        return x if x.shape[0] == bsz else x.expand(bsz)

    size_b = expand1(size_b).clamp(0.4, 1.8)
    ori_b = expand1(ori_b)
    floor_b = expand1(floor_b)
    wall_b = expand1(wall_b)
    obj_b = expand1(obj_b)
    sid = max(0, min(3, int(shape)))

    lighting_config = lighting_config or LightingConfig()
    mesh_resolution_config = mesh_resolution_config or MeshResolutionConfig()
    camera_config = camera_config or CameraConfig()
    room_config = room_config or RoomConfig()
    object_config = object_config or ObjectConfig()

    yaw_offset_deg_t = _as_torch_scalar(camera_config.yaw_offset_deg, device, dtype)
    theta = 2.0 * torch.pi * torch.remainder(ori_b, 1.0) + (
        yaw_offset_deg_t * (torch.pi / 180.0)
    )

    ssaa_scale = int(max(1, ssaa_scale))
    h0, w0 = _parse_image_size(image_size)
    h = h0 * ssaa_scale
    w = w0 * ssaa_scale
    dx, dy = _cached_image_plane(h, w, dkey[0], dkey[1], dtype)

    cam_radius = _as_torch_scalar(camera_config.radius, device, dtype)
    cam_height = _as_torch_scalar(camera_config.height, device, dtype)
    target = _as_torch_vec3(camera_config.target, device, dtype)
    cam_pos = torch.stack(
        [
            cam_radius * torch.sin(theta),
            torch.full_like(theta, cam_height),
            cam_radius * torch.cos(theta),
        ],
        dim=-1,
    )

    forward = target[None, :] - cam_pos
    forward = forward / torch.clamp(torch.linalg.norm(forward, dim=-1, keepdim=True), min=1e-8)
    up_hint = _cached_vec3_tensor(dkey[0], dkey[1], dtype, 0.0, 1.0, 0.0).expand(bsz, 3)
    right = torch.cross(forward, up_hint, dim=-1)
    right = right / torch.clamp(torch.linalg.norm(right, dim=-1, keepdim=True), min=1e-8)
    up = torch.cross(right, forward, dim=-1)
    up = up / torch.clamp(torch.linalg.norm(up, dim=-1, keepdim=True), min=1e-8)

    ray_dir = (
        right[:, None, None, :] * dx[None, :, :, None]
        + up[:, None, None, :] * dy[None, :, :, None]
        + forward[:, None, None, :]
    )
    ray_dir = ray_dir / torch.clamp(torch.linalg.norm(ray_dir, dim=-1, keepdim=True), min=1e-8)

    hue_v_t = _as_torch_scalar(hue_v, device, dtype).clamp(0.0, 1.0)
    floor_rgb = _hue_to_rgb_constrained(floor_b, hue_v_t)
    wall_rgb = _hue_to_rgb_constrained(wall_b, hue_v_t)
    obj_rgb = _hue_to_rgb_constrained(obj_b, hue_v_t)

    light_base = _as_torch_vec3(lighting_config.light_position_world, device, dtype)
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.zeros_like(c)
    o = torch.ones_like(c)
    rot = torch.stack(
        [
            torch.stack([c, z, s], dim=-1),
            torch.stack([z, o, z], dim=-1),
            torch.stack([-s, z, c], dim=-1),
        ],
        dim=1,
    )
    light_pos_world = rot @ light_base
    ambient = _as_torch_scalar(lighting_config.ambient, device, dtype).clamp(0.0, 1.0)
    diffuse = _as_torch_scalar(lighting_config.diffuse, device, dtype).clamp(0.0, 2.0)
    shadow_s = _as_torch_scalar(shadow_strength, device, dtype).clamp(0.0, 1.0)
    room_y0 = _as_torch_scalar(room_config.floor_y, device, dtype)
    room_y1 = _as_torch_scalar(room_config.wall_top_y, device, dtype)
    half_w = _as_torch_scalar(room_config.half_width, device, dtype)
    half_d = _as_torch_scalar(room_config.half_depth, device, dtype)
    room_x0, room_x1 = -half_w, half_w
    room_z0, room_z1 = -half_d, half_d
    room_bounds = (room_y0, room_y1, room_x0, room_x1, room_z0, room_z1)
    room_eps = _cached_scalar_tensor(dkey[0], dkey[1], dtype, 1e-8)
    room_sky = _cached_vec3_tensor(dkey[0], dkey[1], dtype, 0.72, 0.88, 1.0)
    room_normals = (
        _cached_vec3_tensor(dkey[0], dkey[1], dtype, 0.0, 1.0, 0.0),
        _cached_vec3_tensor(dkey[0], dkey[1], dtype, 0.0, 0.0, 1.0),
        _cached_vec3_tensor(dkey[0], dkey[1], dtype, 0.0, 0.0, -1.0),
        _cached_vec3_tensor(dkey[0], dkey[1], dtype, 1.0, 0.0, 0.0),
        _cached_vec3_tensor(dkey[0], dkey[1], dtype, -1.0, 0.0, 0.0),
    )

    base_scale = _as_torch_shape_vec3(object_config.base_scales, sid, device, dtype)
    base_scale = base_scale * _as_torch_scalar(object_config.global_scale_multiplier, device, dtype)
    scale_g = base_scale[None, :] * size_b[:, None]

    extent_y = _cached_scalar_tensor(dkey[0], dkey[1], dtype, 2.0 if sid == 3 else 1.0)
    clearance = _as_torch_shape_scalar(object_config.ground_clearance_by_shape, sid, device, dtype)
    center_y = room_y0 + clearance + extent_y * scale_g[:, 1]
    center_world_g = torch.stack(
        [
            _as_torch_scalar(object_config.center_x, device, dtype).expand(bsz),
            center_y,
            _as_torch_scalar(object_config.center_z, device, dtype).expand(bsz),
        ],
        dim=-1,
    )
    ray_origin = cam_pos[:, None, None, :]
    den_eps = 1e-8
    inf_t = torch.full((bsz, h, w), float("inf"), device=device, dtype=dtype)
    zero_a = torch.zeros((bsz, h, w), device=device, dtype=dtype)
    bg = room_sky[None, None, None, :].expand(bsz, h, w, 3).clone()
    room_y0_b = room_y0.expand(bsz)
    room_y1_b = room_y1.expand(bsz)

    def _plane_hit(axis: int, coord: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coord_b = coord if coord.ndim > 0 else coord.expand(bsz)
        den = ray_dir[..., axis]
        den_safe = torch.where(torch.abs(den) > den_eps, den, torch.where(den >= 0.0, room_eps, -room_eps))
        t = (coord_b[:, None, None] - ray_origin[..., axis]) / den_safe
        p = ray_origin + ray_dir * t[..., None]
        valid = (torch.abs(den) > den_eps) & (t > 1e-6)
        return t, p, valid

    def _shade(base_rgb: torch.Tensor, normal: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        l = light_pos_world[:, None, None, :] - p
        l = l / torch.clamp(torch.linalg.norm(l, dim=-1, keepdim=True), min=1e-8)
        n = normal if normal.ndim == 4 else normal[None, None, None, :]
        ndotl = torch.clamp(torch.sum(l * n, dim=-1), min=0.0)
        return torch.clamp(base_rgb[:, None, None, :] * (ambient + diffuse * ndotl)[..., None], 0.0, 1.0)

    surf_t: list[torch.Tensor] = []
    surf_c: list[torch.Tensor] = []
    surf_a: list[torch.Tensor] = []

    # Floor (infinite plane), with position-dependent transmissivity and soft shadow by object segment length.
    t_floor, p_floor, valid_floor = _plane_hit(axis=1, coord=room_y0)
    p_floor_safe = torch.where(valid_floor[..., None], p_floor, ray_origin)
    n_floor = room_normals[0]
    l_floor = light_pos_world[:, None, None, :] - p_floor_safe
    light_dist = torch.linalg.norm(l_floor, dim=-1)
    l_floor_dir = l_floor / torch.clamp(light_dist[..., None], min=1e-8)
    t_l0, t_l1 = _ray_object_hit_interval_t(
        ray_origin_world=p_floor_safe,
        ray_dir_world=l_floor_dir,
        shape_id=sid,
        center_world=center_world_g[:, None, None, :],
        scale_xyz=scale_g[:, None, None, :],
        t_far=100.0,
    )
    seg_shadow_raw = torch.minimum(t_l1, light_dist) - _smooth_positive(t_l0, delta=1e-3)
    seg_shadow = _smooth_positive(seg_shadow_raw, delta=1e-3)
    occ_shadow = _occlusion_from_segment_length(seg_shadow, sharpness=4.0)
    occ_shadow = torch.where(valid_floor, occ_shadow, zero_a)
    ndotl_floor = torch.clamp(torch.sum(l_floor_dir * n_floor[None, None, None, :], dim=-1), min=0.0)
    floor_lit = ambient + diffuse * ndotl_floor * (1.0 - shadow_s * occ_shadow)
    floor_color = torch.clamp(floor_rgb[:, None, None, :] * floor_lit[..., None], 0.0, 1.0)

    floor_soft = 0.08
    alpha_floor = torch.ones_like(t_floor)
    if room_config.draw_left_wall:
        alpha_floor = alpha_floor * torch.sigmoid((p_floor_safe[..., 0] - room_x0) / floor_soft)
    if room_config.draw_right_wall:
        alpha_floor = alpha_floor * torch.sigmoid((room_x1 - p_floor_safe[..., 0]) / floor_soft)
    if room_config.draw_back_wall:
        alpha_floor = alpha_floor * torch.sigmoid((p_floor_safe[..., 2] - room_z0) / floor_soft)
    if room_config.draw_front_wall:
        alpha_floor = alpha_floor * torch.sigmoid((room_z1 - p_floor_safe[..., 2]) / floor_soft)
    alpha_floor = torch.clamp(alpha_floor, 0.0, 1.0)
    alpha_floor = torch.where(valid_floor, alpha_floor, zero_a)
    surf_t.append(torch.where(valid_floor, t_floor, inf_t))
    surf_c.append(torch.where(valid_floor[..., None], floor_color, torch.zeros_like(floor_color)))
    surf_a.append(alpha_floor)

    def _append_wall(enabled: bool, axis: int, coord: torch.Tensor, normal: torch.Tensor) -> None:
        if not enabled:
            return
        t_wall, p_wall, valid_wall = _plane_hit(axis=axis, coord=coord)
        p_wall_safe = torch.where(valid_wall[..., None], p_wall, ray_origin)
        wall_color = _shade(wall_rgb, normal, p_wall_safe)
        wall_soft_y = 0.08
        wall_soft_side = 0.08
        alpha_floor_edge = torch.sigmoid((p_wall_safe[..., 1] - room_y0_b[:, None, None]) / wall_soft_y)
        if axis == 2:
            alpha_wall_wall = (
                torch.sigmoid((p_wall_safe[..., 0] - room_x0) / wall_soft_side)
                * torch.sigmoid((room_x1 - p_wall_safe[..., 0]) / wall_soft_side)
            )
        else:
            alpha_wall_wall = (
                torch.sigmoid((p_wall_safe[..., 2] - room_z0) / wall_soft_side)
                * torch.sigmoid((room_z1 - p_wall_safe[..., 2]) / wall_soft_side)
            )
        alpha_top_edge = torch.sigmoid((room_y1_b[:, None, None] - p_wall_safe[..., 1]) / wall_soft_y)
        alpha_wall = torch.clamp(alpha_floor_edge * alpha_wall_wall * alpha_top_edge, 0.0, 1.0)
        alpha_wall = torch.where(valid_wall, alpha_wall, zero_a)
        surf_t.append(torch.where(valid_wall, t_wall, inf_t))
        surf_c.append(torch.where(valid_wall[..., None], wall_color, torch.zeros_like(wall_color)))
        surf_a.append(alpha_wall)

    _append_wall(room_config.draw_back_wall, axis=2, coord=room_z0, normal=room_normals[1])
    _append_wall(room_config.draw_front_wall, axis=2, coord=room_z1, normal=room_normals[2])
    _append_wall(room_config.draw_left_wall, axis=0, coord=room_x0, normal=room_normals[3])
    _append_wall(room_config.draw_right_wall, axis=0, coord=room_x1, normal=room_normals[4])

    # Object (analytic entry/exit), opacity from segment length with linear-near-zero and saturation behavior.
    t_obj0, t_obj1 = _ray_object_hit_interval_t(
        ray_origin_world=ray_origin,
        ray_dir_world=ray_dir,
        shape_id=sid,
        center_world=center_world_g[:, None, None, :],
        scale_xyz=scale_g[:, None, None, :],
        t_far=100.0,
    )
    valid_obj = torch.isfinite(t_obj0) & (t_obj1 > 0.0)
    t_obj_hit = torch.where(valid_obj, torch.clamp(t_obj0, min=1e-5), inf_t)
    t_obj_safe = torch.where(valid_obj, torch.clamp(t_obj0, min=1e-5), torch.zeros_like(t_obj0))
    p_obj = ray_origin + ray_dir * t_obj_safe[..., None]
    n_obj = _normal_object_analytic(
        p_obj,
        sid,
        center_world_g[:, None, None, :],
        scale_g[:, None, None, :],
    )
    obj_color = _shade(obj_rgb, n_obj, p_obj)
    obj_color = torch.where(valid_obj[..., None], obj_color, torch.zeros_like(obj_color))
    seg_view_raw = t_obj1 - _smooth_positive(t_obj0, delta=1e-3)
    seg_view = _smooth_positive(seg_view_raw, delta=1e-3)
    alpha_obj = _occlusion_from_segment_length(seg_view, sharpness=5.0)
    alpha_obj = torch.where(valid_obj, torch.clamp(alpha_obj, 0.0, 1.0), zero_a)
    surf_t.append(t_obj_hit)
    surf_c.append(obj_color)
    surf_a.append(alpha_obj)

    stack_t = torch.stack(surf_t, dim=0)
    stack_c = torch.stack(surf_c, dim=0)
    stack_a = torch.stack(surf_a, dim=0)
    remaining_t = stack_t.clone()
    trans = torch.ones((bsz, h, w, 1), device=device, dtype=dtype)
    out_acc = torch.zeros_like(bg)
    n_surfaces = stack_t.shape[0]
    inf_fill = torch.full((1, bsz, h, w), float("inf"), device=device, dtype=dtype)

    # Front-to-back alpha compositing by repeatedly selecting nearest remaining surface.
    for _ in range(n_surfaces):
        idx = torch.argmin(remaining_t, dim=0, keepdim=True)  # (1,B,H,W)
        t_sel = torch.gather(remaining_t, 0, idx).squeeze(0)
        a_sel = torch.gather(stack_a, 0, idx).squeeze(0)
        c_sel = torch.gather(stack_c, 0, idx[..., None].expand(-1, -1, -1, -1, 3)).squeeze(0)
        a_sel = torch.where(torch.isfinite(t_sel), a_sel, torch.zeros_like(a_sel))
        out_acc = out_acc + trans * a_sel[..., None] * c_sel
        trans = trans * (1.0 - a_sel[..., None])
        remaining_t = remaining_t.scatter(0, idx, inf_fill)

    out = out_acc + trans * bg

    if ssaa_scale > 1:
        out = _downsample_mean(out, ssaa_scale)
    if output_chw:
        out = out.permute(0, 3, 1, 2)
    if bsz == 1:
        return out[0]
    return out

class Differentiable3Dshapes(nn.Module):
    """nn.Module wrapper around `render_3dshapes_image`.

    Main factors are provided to `forward`; all other renderer settings are fixed at init.
    """

    def __init__(
        self,
        *,
        hue_v: float | torch.Tensor = 0.9,
        shadow_strength: float | torch.Tensor = 1.0,
        ssaa_scale: int = 4,
        image_size: int | Tuple[int, int] = 64,
        lighting_config: LightingConfig | None = None,
        mesh_resolution_config: MeshResolutionConfig | None = None,
        camera_config: CameraConfig | None = None,
        room_config: RoomConfig | None = None,
        object_config: ObjectConfig | None = None,
        output_chw: bool = True,
    ) -> None:
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
            return x if x.shape[0] == bsz else x.expand(bsz)

        shape_ids = torch.clamp(expand1(shape_b).to(torch.int64), 0, 3)
        size_b = expand1(size_b)
        ori_b = expand1(ori_b)
        floor_b = expand1(floor_b)
        wall_b = expand1(wall_b)
        obj_b = expand1(obj_b)

        if return_grad:
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
        ssaa_scale=4,
        image_size=64,
        lighting_config=light_cfg,
        mesh_resolution_config=mesh_res_cfg,
        camera_config=cam_cfg,
        room_config=room_cfg,
        object_config=obj_cfg,
        output_chw=True,
    )
    renderer = Differentiable3Dshapes()
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
    n_frames = 64
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

    chunk_size = 16

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
