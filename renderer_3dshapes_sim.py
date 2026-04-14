from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Utility
# ------------------------------

def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _parse_image_size(image_size: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(image_size, int):
        h = w = int(image_size)
    else:
        h, w = int(image_size[0]), int(image_size[1])
    if h <= 0 or w <= 0:
        raise ValueError("image_size must be positive")
    return h, w


@dataclass
class Mesh:
    verts: np.ndarray      # (N, 3)
    normals: np.ndarray    # (N, 3)
    faces: np.ndarray      # (M, 3) int


@dataclass(frozen=True)
class CameraConfig:
    radius: float = 6.0
    height: float = 1.0
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
    center_y_by_shape: Tuple[float, ...] = (-0.56, -0.26, -0.40, -0.22)
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


DEFAULT_CAMERA_CONFIG = CameraConfig()
DEFAULT_ROOM_CONFIG = RoomConfig()
DEFAULT_OBJECT_CONFIG = ObjectConfig()
DEFAULT_LIGHTING_CONFIG = LightingConfig()
DEFAULT_MESH_RESOLUTION_CONFIG = MeshResolutionConfig()


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

def _make_uv_sphere(lat_steps: int = 18, lon_steps: int = 36) -> Mesh:
    verts = []
    normals = []

    for i in range(lat_steps + 1):
        phi = math.pi * i / lat_steps
        y = math.cos(phi)
        r = math.sin(phi)
        for j in range(lon_steps):
            th = 2.0 * math.pi * j / lon_steps
            x = r * math.cos(th)
            z = r * math.sin(th)
            p = np.array([x, y, z], dtype=np.float32)
            verts.append(p)
            normals.append(p)

    verts = np.array(verts, dtype=np.float32)
    normals = _normalize(np.array(normals, dtype=np.float32))

    faces = []
    def idx(i: int, j: int) -> int:
        return i * lon_steps + (j % lon_steps)

    for i in range(lat_steps):
        for j in range(lon_steps):
            a = idx(i, j)
            b = idx(i + 1, j)
            c = idx(i + 1, j + 1)
            d = idx(i, j + 1)
            if i > 0:
                faces.append((a, b, d))
            if i < lat_steps - 1:
                faces.append((b, c, d))

    return Mesh(verts=verts, normals=normals, faces=np.array(faces, dtype=np.int32))


def _make_cylinder(radial_steps: int = 36, height_steps: int = 1) -> Mesh:
    verts = []
    normals = []
    faces = []

    # side
    for yid in range(height_steps + 1):
        y = -1.0 + 2.0 * yid / height_steps
        for j in range(radial_steps):
            th = 2.0 * math.pi * j / radial_steps
            x = math.cos(th)
            z = math.sin(th)
            verts.append((x, y, z))
            normals.append((x, 0.0, z))

    def side_idx(yid: int, j: int) -> int:
        return yid * radial_steps + (j % radial_steps)

    for yid in range(height_steps):
        for j in range(radial_steps):
            a = side_idx(yid, j)
            b = side_idx(yid + 1, j)
            c = side_idx(yid + 1, j + 1)
            d = side_idx(yid, j + 1)
            faces.append((a, b, d))
            faces.append((b, c, d))

    # caps (with duplicated vertices for sharp normal)
    base = len(verts)
    verts.append((0.0, 1.0, 0.0))
    normals.append((0.0, 1.0, 0.0))
    top_center = base
    for j in range(radial_steps):
        th = 2.0 * math.pi * j / radial_steps
        x = math.cos(th)
        z = math.sin(th)
        verts.append((x, 1.0, z))
        normals.append((0.0, 1.0, 0.0))
    for j in range(radial_steps):
        a = top_center
        b = top_center + 1 + j
        c = top_center + 1 + ((j + 1) % radial_steps)
        faces.append((a, b, c))

    base = len(verts)
    verts.append((0.0, -1.0, 0.0))
    normals.append((0.0, -1.0, 0.0))
    bot_center = base
    for j in range(radial_steps):
        th = 2.0 * math.pi * j / radial_steps
        x = math.cos(th)
        z = math.sin(th)
        verts.append((x, -1.0, z))
        normals.append((0.0, -1.0, 0.0))
    for j in range(radial_steps):
        a = bot_center
        b = bot_center + 1 + ((j + 1) % radial_steps)
        c = bot_center + 1 + j
        faces.append((a, b, c))

    return Mesh(
        verts=np.array(verts, dtype=np.float32),
        normals=_normalize(np.array(normals, dtype=np.float32)),
        faces=np.array(faces, dtype=np.int32),
    )


def _make_cube() -> Mesh:
    # 24 vertices (4 per face) so normals stay sharp.
    verts = []
    normals = []
    faces = []

    face_defs = [
        # normal, 4 corners (ccw)
        ((1, 0, 0), [(1, -1, -1), (1, -1, 1), (1, 1, 1), (1, 1, -1)]),
        ((-1, 0, 0), [(-1, -1, 1), (-1, -1, -1), (-1, 1, -1), (-1, 1, 1)]),
        ((0, 1, 0), [(-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)]),
        ((0, -1, 0), [(-1, -1, 1), (1, -1, 1), (1, -1, -1), (-1, -1, -1)]),
        ((0, 0, 1), [(-1, -1, 1), (-1, 1, 1), (1, 1, 1), (1, -1, 1)]),
        ((0, 0, -1), [(1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1)]),
    ]

    for nrm, corners in face_defs:
        base = len(verts)
        verts.extend(corners)
        normals.extend([nrm] * 4)
        faces.append((base + 0, base + 1, base + 2))
        faces.append((base + 0, base + 2, base + 3))

    return Mesh(
        verts=np.array(verts, dtype=np.float32),
        normals=_normalize(np.array(normals, dtype=np.float32)),
        faces=np.array(faces, dtype=np.int32),
    )


def _make_capsule(radial_steps: int = 28, hemi_steps: int = 8) -> Mesh:
    # Capsule aligned with y axis, total height ~4 before scaling.
    verts = []
    normals = []

    # Top hemisphere (center y=+1)
    for i in range(hemi_steps + 1):
        phi = (math.pi / 2.0) * i / hemi_steps
        y_local = math.cos(phi)
        r = math.sin(phi)
        y = 1.0 + y_local
        for j in range(radial_steps):
            th = 2.0 * math.pi * j / radial_steps
            x = r * math.cos(th)
            z = r * math.sin(th)
            verts.append((x, y, z))
            normals.append((x, y_local, z))

    # Bottom hemisphere (center y=-1)
    offset = len(verts)
    for i in range(hemi_steps + 1):
        phi = (math.pi / 2.0) * i / hemi_steps
        y_local = -math.cos(phi)
        r = math.sin(phi)
        y = -1.0 + y_local
        for j in range(radial_steps):
            th = 2.0 * math.pi * j / radial_steps
            x = r * math.cos(th)
            z = r * math.sin(th)
            verts.append((x, y, z))
            normals.append((x, y_local, z))

    faces = []

    def idx_top(i: int, j: int) -> int:
        return i * radial_steps + (j % radial_steps)

    def idx_bot(i: int, j: int) -> int:
        return offset + i * radial_steps + (j % radial_steps)

    # connect top hemisphere rows
    for i in range(hemi_steps):
        for j in range(radial_steps):
            a = idx_top(i, j)
            b = idx_top(i + 1, j)
            c = idx_top(i + 1, j + 1)
            d = idx_top(i, j + 1)
            faces.append((a, b, d))
            faces.append((b, c, d))

    # connect bottom hemisphere rows
    for i in range(hemi_steps):
        for j in range(radial_steps):
            a = idx_bot(i, j)
            b = idx_bot(i + 1, j)
            c = idx_bot(i + 1, j + 1)
            d = idx_bot(i, j + 1)
            faces.append((a, d, b))
            faces.append((b, d, c))

    # connect equators to create middle cylinder band
    i_top_eq = hemi_steps
    i_bot_eq = hemi_steps
    for j in range(radial_steps):
        a = idx_top(i_top_eq, j)
        b = idx_bot(i_bot_eq, j)
        c = idx_bot(i_bot_eq, j + 1)
        d = idx_top(i_top_eq, j + 1)
        faces.append((a, b, d))
        faces.append((b, c, d))

    return Mesh(
        verts=np.array(verts, dtype=np.float32),
        normals=_normalize(np.array(normals, dtype=np.float32)),
        faces=np.array(faces, dtype=np.int32),
    )


# ------------------------------
# Public API
# ------------------------------

@lru_cache(maxsize=64)
def _get_meshes(
    cylinder_radial_steps: int,
    sphere_lat_steps: int,
    sphere_lon_steps: int,
    capsule_radial_steps: int,
    capsule_hemi_steps: int,
) -> dict[int, Mesh]:
    return {
        0: _make_cube(),
        1: _make_cylinder(radial_steps=max(8, int(cylinder_radial_steps)), height_steps=1),
        2: _make_uv_sphere(
            lat_steps=max(4, int(sphere_lat_steps)),
            lon_steps=max(8, int(sphere_lon_steps)),
        ),
        3: _make_capsule(
            radial_steps=max(8, int(capsule_radial_steps)),
            hemi_steps=max(2, int(capsule_hemi_steps)),
        ),
    }


def _downsample_mean(
    image_hwc: np.ndarray | torch.Tensor,
    factor: int,
    prefilter_sigma: float = 0.0,
    filter_order: Literal["pre", "post"] = "pre",
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

        sigma = float(prefilter_sigma)

        def _gaussian_blur_nchw(inp: torch.Tensor, sig: float) -> torch.Tensor:
            radius = max(1, int(math.ceil(2.0 * sig)))
            ksize = radius * 2 + 1
            coords = torch.arange(-radius, radius + 1, device=inp.device, dtype=inp.dtype)
            kernel_1d = torch.exp(-(coords * coords) / (2.0 * sig * sig))
            kernel_1d = kernel_1d / torch.clamp(kernel_1d.sum(), min=1e-12)
            channels = inp.shape[1]
            kernel_x = kernel_1d.view(1, 1, 1, ksize).expand(channels, 1, 1, ksize)
            kernel_y = kernel_1d.view(1, 1, ksize, 1).expand(channels, 1, ksize, 1)
            out = F.conv2d(inp, kernel_x, padding=(0, radius), groups=channels)
            out = F.conv2d(out, kernel_y, padding=(radius, 0), groups=channels)
            return out

        if filter_order not in ("pre", "post"):
            raise ValueError(f"filter_order must be 'pre' or 'post', got {filter_order!r}")

        if sigma > 0.0 and filter_order == "pre":
            # Smooth high-res image before area downsampling.
            x = _gaussian_blur_nchw(x, sigma)

        # Area interpolation keeps energy stable after supersampling.
        x = F.interpolate(x, size=(h // factor, w // factor), mode="area")
        if sigma > 0.0 and filter_order == "post":
            # Smooth only after resize (requested variant).
            x = _gaussian_blur_nchw(x, sigma)
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


def _sdf_object_world(
    p_world: torch.Tensor,
    shape_id: int,
    center_world: torch.Tensor,
    scale_xyz: torch.Tensor,
    scale_min: torch.Tensor | None = None,
) -> torch.Tensor:
    p_local = (p_world - center_world) / scale_xyz
    px, py, pz = p_local[..., 0], p_local[..., 1], p_local[..., 2]

    if shape_id == 0:
        q = torch.abs(p_local) - 1.0
        outside = torch.linalg.norm(torch.clamp(q, min=0.0), dim=-1)
        inside = torch.clamp(torch.max(q, dim=-1).values, max=0.0)
        d_local = outside + inside
    elif shape_id == 1:
        d = torch.stack([torch.sqrt(px * px + pz * pz) - 1.0, torch.abs(py) - 1.0], dim=-1)
        outside = torch.linalg.norm(torch.clamp(d, min=0.0), dim=-1)
        inside = torch.clamp(torch.max(d, dim=-1).values, max=0.0)
        d_local = outside + inside
    elif shape_id == 2:
        d_local = torch.linalg.norm(p_local, dim=-1) - 1.0
    else:
        # Capsule: segment from y=-1 to y=+1 with radius 1.
        cy = torch.clamp(py, -1.0, 1.0)
        # Equivalent to norm(p_local - [0, cy, 0]) but avoids allocating a stacked tensor.
        dy = py - cy
        d_local = torch.sqrt(px * px + dy * dy + pz * pz) - 1.0

    if scale_min is None:
        scale_min = torch.min(scale_xyz)
    return d_local * scale_min


def _ray_march_object(
    cam_pos: torch.Tensor,
    ray_dir: torch.Tensor,
    shape_id: int,
    center_world: torch.Tensor,
    scale_xyz: torch.Tensor,
    max_steps: int = 80,
    eps: float = 1e-3,
    t_far: float = 100.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Kept for backward-compatible signature; intersection is now analytic.
    _ = max_steps, eps
    ray_origin = cam_pos[:, None, None, :]
    t_hit = _ray_object_first_hit_t(
        ray_origin_world=ray_origin,
        ray_dir_world=ray_dir,
        shape_id=shape_id,
        center_world=center_world[:, None, None, :],
        scale_xyz=scale_xyz[:, None, None, :],
        t_far=t_far,
    )
    hit = torch.isfinite(t_hit)
    t_out = torch.where(hit, t_hit, torch.full_like(t_hit, float(t_far)))
    return hit, t_out


def _ray_object_first_hit_t(
    ray_origin_world: torch.Tensor,
    ray_dir_world: torch.Tensor,
    shape_id: int,
    center_world: torch.Tensor,
    scale_xyz: torch.Tensor,
    t_far: float = 100.0,
    t_min: float = 1e-6,
    softmin_tau: float | None = None,
) -> torch.Tensor:
    dtype = ray_dir_world.dtype
    device = ray_dir_world.device
    dkey = _device_cache_key(device)
    t_eps = _cached_scalar_tensor(dkey[0], dkey[1], dtype, float(t_min))
    den_eps = _cached_scalar_tensor(dkey[0], dkey[1], dtype, 1e-12)
    sqrt_eps = _cached_scalar_tensor(dkey[0], dkey[1], dtype, 1e-12)
    one = _cached_scalar_tensor(dkey[0], dkey[1], dtype, 1.0)
    neg_one = _cached_scalar_tensor(dkey[0], dkey[1], dtype, -1.0)
    inf = _cached_scalar_tensor(dkey[0], dkey[1], dtype, float("inf"))
    t_far_t = _cached_scalar_tensor(dkey[0], dkey[1], dtype, float(t_far))
    finite_cap = _cached_scalar_tensor(dkey[0], dkey[1], dtype, float(t_far) + 4.0)

    scale_xyz_safe = torch.clamp(scale_xyz, min=1e-8)
    ro = (ray_origin_world - center_world) / scale_xyz_safe
    rd = ray_dir_world / scale_xyz_safe

    t_hit = torch.full_like(ray_dir_world[..., 0], float("inf"))
    t_candidates: list[torch.Tensor] = []
    valid_candidates: list[torch.Tensor] = []

    def _add_candidate(t_candidate: torch.Tensor, valid: torch.Tensor) -> None:
        nonlocal t_hit
        t_candidate_clean = torch.nan_to_num(
            t_candidate,
            nan=float(t_far) + 4.0,
            posinf=float(t_far) + 4.0,
            neginf=-float(t_far) - 4.0,
        )
        t_candidates.append(t_candidate_clean)
        valid_candidates.append(valid.to(dtype))
        t_valid = torch.where(valid & (t_candidate_clean > t_eps), t_candidate_clean, inf)
        t_hit = torch.minimum(t_hit, t_valid)

    ox, oy, oz = ro[..., 0], ro[..., 1], ro[..., 2]
    dx, dy, dz = rd[..., 0], rd[..., 1], rd[..., 2]

    if shape_id == 0:
        parallel = torch.abs(rd) < den_eps
        outside_parallel = parallel & ((ro < neg_one) | (ro > one))
        no_hit_parallel = outside_parallel.any(dim=-1)

        rd_safe = torch.where(torch.abs(rd) > den_eps, rd, torch.where(rd >= 0.0, den_eps, -den_eps))
        inv_rd = one / rd_safe
        t0 = (neg_one - ro) * inv_rd
        t1 = (one - ro) * inv_rd
        t_near_axis = torch.minimum(t0, t1)
        t_far_axis = torch.maximum(t0, t1)
        t_near_axis = torch.where(parallel, -inf, t_near_axis)
        t_far_axis = torch.where(parallel, inf, t_far_axis)

        t_near = t_near_axis.max(dim=-1).values
        t_far_box = t_far_axis.min(dim=-1).values
        valid = (~no_hit_parallel) & (t_far_box >= t_near) & (t_far_box > t_eps)
        t_box = torch.where(t_near > t_eps, t_near, t_far_box)
        _add_candidate(t_box, valid)
    elif shape_id == 1:
        a = dx * dx + dz * dz
        bq = 2.0 * (ox * dx + oz * dz)
        c = ox * ox + oz * oz - 1.0
        disc = bq * bq - 4.0 * a * c
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=sqrt_eps))
        den = 2.0 * a
        den_ok = torch.abs(den) > den_eps
        root_ok = (disc >= 0.0) & den_ok
        den_safe = torch.where(den_ok, den, one)

        t0 = (-bq - sqrt_disc) / den_safe
        t1 = (-bq + sqrt_disc) / den_safe
        y0 = oy + t0 * dy
        y1 = oy + t1 * dy
        _add_candidate(t0, root_ok & (y0 >= -1.0) & (y0 <= 1.0))
        _add_candidate(t1, root_ok & (y1 >= -1.0) & (y1 <= 1.0))

        dy_ok = torch.abs(dy) > den_eps
        dy_safe = torch.where(dy_ok, dy, one)
        t_cap_top = (one - oy) / dy_safe
        x_top = ox + t_cap_top * dx
        z_top = oz + t_cap_top * dz
        _add_candidate(t_cap_top, dy_ok & (x_top * x_top + z_top * z_top <= 1.0))

        t_cap_bot = (neg_one - oy) / dy_safe
        x_bot = ox + t_cap_bot * dx
        z_bot = oz + t_cap_bot * dz
        _add_candidate(t_cap_bot, dy_ok & (x_bot * x_bot + z_bot * z_bot <= 1.0))
    elif shape_id == 2:
        a = dx * dx + dy * dy + dz * dz
        bq = 2.0 * (ox * dx + oy * dy + oz * dz)
        c = ox * ox + oy * oy + oz * oz - 1.0
        disc = bq * bq - 4.0 * a * c
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=sqrt_eps))
        den = 2.0 * a
        den_ok = torch.abs(den) > den_eps
        root_ok = (disc >= 0.0) & den_ok
        den_safe = torch.where(den_ok, den, one)
        t0 = (-bq - sqrt_disc) / den_safe
        t1 = (-bq + sqrt_disc) / den_safe
        _add_candidate(t0, root_ok)
        _add_candidate(t1, root_ok)
    else:
        a = dx * dx + dz * dz
        bq = 2.0 * (ox * dx + oz * dz)
        c = ox * ox + oz * oz - 1.0
        disc = bq * bq - 4.0 * a * c
        sqrt_disc = torch.sqrt(torch.clamp(disc, min=sqrt_eps))
        den = 2.0 * a
        den_ok = torch.abs(den) > den_eps
        root_ok = (disc >= 0.0) & den_ok
        den_safe = torch.where(den_ok, den, one)
        t0 = (-bq - sqrt_disc) / den_safe
        t1 = (-bq + sqrt_disc) / den_safe
        y0 = oy + t0 * dy
        y1 = oy + t1 * dy
        _add_candidate(t0, root_ok & (y0 >= -1.0) & (y0 <= 1.0))
        _add_candidate(t1, root_ok & (y1 >= -1.0) & (y1 <= 1.0))

        a_s = dx * dx + dy * dy + dz * dz
        den_s = 2.0 * a_s
        den_s_ok = torch.abs(den_s) > den_eps
        den_s_safe = torch.where(den_s_ok, den_s, one)
        for cy in (-1.0, 1.0):
            ocy = oy - cy
            b_s = 2.0 * (ox * dx + ocy * dy + oz * dz)
            c_s = ox * ox + ocy * ocy + oz * oz - 1.0
            disc_s = b_s * b_s - 4.0 * a_s * c_s
            sqrt_disc_s = torch.sqrt(torch.clamp(disc_s, min=sqrt_eps))
            root_s_ok = (disc_s >= 0.0) & den_s_ok
            ts0 = (-b_s - sqrt_disc_s) / den_s_safe
            ts1 = (-b_s + sqrt_disc_s) / den_s_safe
            _add_candidate(ts0, root_s_ok)
            _add_candidate(ts1, root_s_ok)

    if softmin_tau is not None and softmin_tau > 0.0 and len(t_candidates) > 0:
        tau_t = torch.clamp(
            torch.as_tensor(softmin_tau, device=device, dtype=dtype),
            min=1e-6,
        )
        tc = torch.stack(t_candidates, dim=0)
        tc = torch.clamp(tc, min=-finite_cap, max=finite_cap)
        vc = torch.stack(valid_candidates, dim=0)
        t_gate = torch.sigmoid((tc - t_eps) / tau_t)
        # Penalize invalid candidates smoothly instead of hard masking.
        score = tc + (1.0 - vc * t_gate) * (t_far_t + 2.0)
        score = torch.clamp(score, min=-finite_cap, max=finite_cap)
        return -tau_t * torch.logsumexp(-score / tau_t, dim=0)

    return torch.where(t_hit < t_far_t, t_hit, torch.full_like(t_hit, float("inf")))


def _normal_from_sdf(
    p_world: torch.Tensor,
    shape_id: int,
    center_world: torch.Tensor,
    scale_xyz: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    ex = torch.zeros((1,) * (p_world.ndim - 1) + (3,), device=p_world.device, dtype=p_world.dtype)
    ey = torch.zeros_like(ex)
    ez = torch.zeros_like(ex)
    ex[..., 0] = eps
    ey[..., 1] = eps
    ez[..., 2] = eps
    scale_min = torch.min(scale_xyz)
    nx = _sdf_object_world(p_world + ex, shape_id, center_world, scale_xyz, scale_min=scale_min) - _sdf_object_world(
        p_world - ex, shape_id, center_world, scale_xyz, scale_min=scale_min
    )
    ny = _sdf_object_world(p_world + ey, shape_id, center_world, scale_xyz, scale_min=scale_min) - _sdf_object_world(
        p_world - ey, shape_id, center_world, scale_xyz, scale_min=scale_min
    )
    nz = _sdf_object_world(p_world + ez, shape_id, center_world, scale_xyz, scale_min=scale_min) - _sdf_object_world(
        p_world - ez, shape_id, center_world, scale_xyz, scale_min=scale_min
    )
    n = torch.stack([nx, ny, nz], dim=-1)
    return n / torch.clamp(torch.linalg.norm(n, dim=-1, keepdim=True), min=1e-8)


def _render_room(
    cam_pos: torch.Tensor,
    ray_dir: torch.Tensor,
    room_config: RoomConfig,
    floor_rgb: torch.Tensor,
    wall_rgb: torch.Tensor,
    light_pos_world: torch.Tensor,
    ambient: torch.Tensor,
    diffuse: torch.Tensor,
    room_bounds: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    plane_normals: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    eps: torch.Tensor | None = None,
    sky: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, h, w, _ = ray_dir.shape
    dtype = ray_dir.dtype
    device = ray_dir.device

    if room_bounds is None:
        y0 = _as_torch_scalar(room_config.floor_y, device, dtype)
        y1 = _as_torch_scalar(room_config.wall_top_y, device, dtype)
        half_w = _as_torch_scalar(room_config.half_width, device, dtype)
        half_d = _as_torch_scalar(room_config.half_depth, device, dtype)
        x0, x1 = -half_w, half_w
        z0, z1 = -half_d, half_d
    else:
        y0, y1, x0, x1, z0, z1 = room_bounds

    if sky is None:
        dkey = _device_cache_key(device)
        sky = _cached_vec3_tensor(dkey[0], dkey[1], dtype, 0.72, 0.88, 1.0)
    color = sky[None, None, None, :].expand(b, h, w, 3).clone()
    depth = torch.full((b, h, w), float("inf"), device=device, dtype=dtype)
    surface_id = torch.full((b, h, w), -1, device=device, dtype=torch.int64)
    hit_points = torch.zeros((b, h, w, 3), device=device, dtype=dtype)
    if eps is None:
        dkey = _device_cache_key(device)
        eps = _cached_scalar_tensor(dkey[0], dkey[1], dtype, 1e-8)
    if plane_normals is None:
        dkey = _device_cache_key(device)
        n_floor = _cached_vec3_tensor(dkey[0], dkey[1], dtype, 0.0, 1.0, 0.0)
        n_back = _cached_vec3_tensor(dkey[0], dkey[1], dtype, 0.0, 0.0, 1.0)
        n_front = _cached_vec3_tensor(dkey[0], dkey[1], dtype, 0.0, 0.0, -1.0)
        n_left = _cached_vec3_tensor(dkey[0], dkey[1], dtype, 1.0, 0.0, 0.0)
        n_right = _cached_vec3_tensor(dkey[0], dkey[1], dtype, -1.0, 0.0, 0.0)
    else:
        n_floor, n_back, n_front, n_left, n_right = plane_normals

    def safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        den_safe = torch.where(torch.abs(den) > eps, den, torch.where(den >= 0.0, eps, -eps))
        return num / den_safe

    def plane_update(
        t: torch.Tensor,
        p: torch.Tensor,
        valid: torch.Tensor,
        normal: torch.Tensor,
        base_rgb: torch.Tensor,
        surf_id: int,
    ) -> None:
        nonlocal color, depth, surface_id, hit_points
        l = light_pos_world[:, None, None, :] - p
        l = l / torch.clamp(torch.linalg.norm(l, dim=-1, keepdim=True), min=1e-8)
        ndotl = torch.clamp(torch.sum(l * normal[None, None, None, :], dim=-1), min=0.0)
        lit = ambient + diffuse * ndotl
        c = torch.clamp(base_rgb[:, None, None, :] * lit[..., None], 0.0, 1.0)
        closer = valid & (t > 1e-6) & (t < depth)
        depth = torch.where(closer, t, depth)
        color = torch.where(closer[..., None], c, color)
        surf_id_t = torch.tensor(surf_id, device=device, dtype=surface_id.dtype)
        surface_id = torch.where(closer, surf_id_t, surface_id)
        hit_points = torch.where(closer[..., None], p, hit_points)

    # floor
    den = ray_dir[..., 1]
    t = safe_div(y0 - cam_pos[:, None, None, 1], den)
    p = cam_pos[:, None, None, :] + t[..., None] * ray_dir
    valid = (torch.abs(den) > 1e-8) & (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 2] >= z0) & (p[..., 2] <= z1)
    plane_update(t, p, valid, n_floor, floor_rgb, surf_id=0)

    if room_config.draw_back_wall:
        den = ray_dir[..., 2]
        t = safe_div(z0 - cam_pos[:, None, None, 2], den)
        p = cam_pos[:, None, None, :] + t[..., None] * ray_dir
        valid = (torch.abs(den) > 1e-8) & (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)
        plane_update(t, p, valid, n_back, wall_rgb, surf_id=1)
    if room_config.draw_front_wall:
        den = ray_dir[..., 2]
        t = safe_div(z1 - cam_pos[:, None, None, 2], den)
        p = cam_pos[:, None, None, :] + t[..., None] * ray_dir
        valid = (torch.abs(den) > 1e-8) & (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)
        plane_update(t, p, valid, n_front, wall_rgb, surf_id=2)
    if room_config.draw_left_wall:
        den = ray_dir[..., 0]
        t = safe_div(x0 - cam_pos[:, None, None, 0], den)
        p = cam_pos[:, None, None, :] + t[..., None] * ray_dir
        valid = (torch.abs(den) > 1e-8) & (p[..., 2] >= z0) & (p[..., 2] <= z1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)
        plane_update(t, p, valid, n_left, wall_rgb, surf_id=3)
    if room_config.draw_right_wall:
        den = ray_dir[..., 0]
        t = safe_div(x1 - cam_pos[:, None, None, 0], den)
        p = cam_pos[:, None, None, :] + t[..., None] * ray_dir
        valid = (torch.abs(den) > 1e-8) & (p[..., 2] >= z0) & (p[..., 2] <= z1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)
        plane_update(t, p, valid, n_right, wall_rgb, surf_id=4)

    floor_mask = surface_id == 0
    return color, depth, floor_mask, hit_points


def _soft_shadow_floor(
    floor_points: torch.Tensor,  # (B,H,W,3)
    floor_mask: torch.Tensor,    # (B,H,W)
    light_pos_world: torch.Tensor,
    shape_id: int,
    center_world: torch.Tensor,
    scale_xyz: torch.Tensor,
    n_samples: int = 24,
    sigma: float = 0.02,
) -> torch.Tensor:
    if floor_points.numel() == 0:
        return torch.empty(floor_mask.shape, device=floor_points.device, dtype=floor_points.dtype)
    _ = n_samples
    ray = light_pos_world[:, None, None, :] - floor_points
    dist = torch.linalg.norm(ray, dim=-1)
    ray_dir = ray / torch.clamp(dist[..., None], min=1e-8)
    center_world_b = center_world[:, None, None, :]
    scale_xyz_b = scale_xyz[:, None, None, :]
    t_hit = _ray_object_first_hit_t(
        ray_origin_world=floor_points,
        ray_dir_world=ray_dir,
        shape_id=shape_id,
        center_world=center_world_b,
        scale_xyz=scale_xyz_b,
        t_far=100.0,
        t_min=1e-6,
        softmin_tau=max(float(sigma), 1e-4),
    )

    # Smooth window: only intersections between 4% and 96% of light segment occlude.
    start_t = 0.04 * dist
    end_t = 0.96 * dist
    sigma_t = torch.clamp(torch.as_tensor(sigma, device=dist.device, dtype=dist.dtype), min=1e-6)
    occ_exit = torch.sigmoid((end_t - t_hit) / sigma_t)

    # Fill the "donut" artifact: when first hit is before start_t, the segment can still
    # be inside the object at start_t and should cast shadow.
    p_start = floor_points + ray_dir * start_t[..., None]
    scale_min = torch.min(scale_xyz_b)
    d_start = _sdf_object_world(
        p_start,
        shape_id,
        center_world_b,
        scale_xyz_b,
        scale_min=scale_min,
    )
    # Continuous inside weight (no hard branch) to preserve gradients near contact.
    inside_w = torch.sigmoid((-d_start) / sigma_t)
    # Smooth union of hit-based occlusion and start-inside occlusion.
    occ = 1.0 - (1.0 - occ_exit) * (1.0 - inside_w)
    vis = 1.0 - occ
    vis = torch.clamp(vis, 0.0, 1.0)
    return torch.where(floor_mask, vis, torch.ones_like(vis))


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
    orientation_period: float | torch.Tensor = 1.0,
    shadow_strength: float | torch.Tensor = 10.0,
    ssaa_scale: int = 4,
    downsample_prefilter_sigma: float = 0.9,
    downsample_filter_order: Literal["pre", "post"] = "pre",
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

    lighting_config = lighting_config or DEFAULT_LIGHTING_CONFIG
    mesh_resolution_config = mesh_resolution_config or DEFAULT_MESH_RESOLUTION_CONFIG
    camera_config = camera_config or DEFAULT_CAMERA_CONFIG
    room_config = room_config or DEFAULT_ROOM_CONFIG
    object_config = object_config or DEFAULT_OBJECT_CONFIG

    orientation_period_t = _as_torch_scalar(orientation_period, device, dtype)
    period_eps = _cached_scalar_tensor(dkey[0], dkey[1], dtype, 1e-8)
    orientation_period_safe = torch.where(
        torch.abs(orientation_period_t) > period_eps,
        orientation_period_t,
        torch.where(orientation_period_t >= 0.0, period_eps, -period_eps),
    )
    yaw_offset_deg_t = _as_torch_scalar(camera_config.yaw_offset_deg, device, dtype)
    theta = 2.0 * torch.pi * torch.remainder(ori_b / orientation_period_safe, 1.0) + (
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

    meshes = _get_meshes(
        cylinder_radial_steps=mesh_resolution_config.cylinder_radial_steps,
        sphere_lat_steps=mesh_resolution_config.sphere_lat_steps,
        sphere_lon_steps=mesh_resolution_config.sphere_lon_steps,
        capsule_radial_steps=mesh_resolution_config.capsule_radial_steps,
        capsule_hemi_steps=mesh_resolution_config.capsule_hemi_steps,
    )

    mesh = meshes[sid]
    base_scale = _as_torch_shape_vec3(object_config.base_scales, sid, device, dtype)
    base_scale = base_scale * _as_torch_scalar(object_config.global_scale_multiplier, device, dtype)
    scale_g = base_scale[None, :] * size_b[:, None]

    extent_y = _cached_scalar_tensor(dkey[0], dkey[1], dtype, float(np.max(np.abs(mesh.verts[:, 1]))))
    if object_config.use_auto_grounding:
        clearance = _as_torch_shape_scalar(object_config.ground_clearance_by_shape, sid, device, dtype)
        center_y = room_y0 + clearance + extent_y * scale_g[:, 1]
    else:
        center_y = _as_torch_shape_scalar(object_config.center_y_by_shape, sid, device, dtype).expand(bsz)
    center_world_g = torch.stack(
        [
            _as_torch_scalar(object_config.center_x, device, dtype).expand(bsz),
            center_y,
            _as_torch_scalar(object_config.center_z, device, dtype).expand(bsz),
        ],
        dim=-1,
    )

    room_color, room_depth, floor_mask, floor_pts = _render_room(
        cam_pos=cam_pos,
        ray_dir=ray_dir,
        room_config=room_config,
        floor_rgb=floor_rgb,
        wall_rgb=wall_rgb,
        light_pos_world=light_pos_world,
        ambient=ambient,
        diffuse=diffuse,
        room_bounds=room_bounds,
        plane_normals=room_normals,
        eps=room_eps,
        sky=room_sky,
    )

    vis = _soft_shadow_floor(
        floor_points=floor_pts,
        floor_mask=floor_mask,
        light_pos_world=light_pos_world,
        shape_id=sid,
        center_world=center_world_g,
        scale_xyz=scale_g,
        n_samples=24,
        sigma=0.03,
    )
    l = light_pos_world[:, None, None, :] - floor_pts
    l = l / torch.clamp(torch.linalg.norm(l, dim=-1, keepdim=True), min=1e-8)
    ndotl = torch.clamp(l[..., 1], min=0.0)
    lit = ambient + diffuse * ndotl * (1.0 - shadow_s * (1.0 - vis))
    floor_shaded = torch.clamp(floor_rgb[:, None, None, :] * lit[..., None], 0.0, 1.0)
    room_color = torch.where(floor_mask[..., None], floor_shaded, room_color)

    hit_obj, t_obj = _ray_march_object(
        cam_pos=cam_pos,
        ray_dir=ray_dir,
        shape_id=sid,
        center_world=center_world_g,
        scale_xyz=scale_g,
    )
    p_obj = cam_pos[:, None, None, :] + ray_dir * t_obj[..., None]
    n_obj = _normal_from_sdf(
        p_obj,
        sid,
        center_world_g[:, None, None, :],
        scale_g[:, None, None, :],
    )
    l_obj = light_pos_world[:, None, None, :] - p_obj
    l_obj = l_obj / torch.clamp(torch.linalg.norm(l_obj, dim=-1, keepdim=True), min=1e-8)
    ndotl_obj = torch.clamp(torch.sum(n_obj * l_obj, dim=-1), min=0.0)
    obj_lit = ambient + diffuse * ndotl_obj
    obj_color = torch.clamp(obj_rgb[:, None, None, :] * obj_lit[..., None], 0.0, 1.0)
    obj_front = hit_obj & (t_obj < room_depth)
    out = torch.where(obj_front[..., None], obj_color, room_color)

    if ssaa_scale > 1:
        out = _downsample_mean(
            out,
            ssaa_scale,
            prefilter_sigma=downsample_prefilter_sigma,
            filter_order=downsample_filter_order,
        )
    if output_chw:
        out = out.permute(0, 3, 1, 2)
    if bsz == 1:
        return out[0]
    return out

class Render3DShapesModule(nn.Module):
    """nn.Module wrapper around `render_3dshapes_image`.

    Main factors are provided to `forward`; all other renderer settings are fixed at init.
    """

    def __init__(
        self,
        *,
        hue_v: float | torch.Tensor = 0.9,
        orientation_period: float | torch.Tensor = 1.0,
        shadow_strength: float | torch.Tensor = 1.0,
        ssaa_scale: int = 4,
        downsample_prefilter_sigma: float = 0.9,
        downsample_filter_order: Literal["pre", "post"] = "pre",
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
        self.orientation_period = orientation_period
        self.shadow_strength = shadow_strength
        self.ssaa_scale = ssaa_scale
        self.downsample_prefilter_sigma = downsample_prefilter_sigma
        self.downsample_filter_order = downsample_filter_order
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
            orientation_period=self.orientation_period,
            shadow_strength=self.shadow_strength,
            ssaa_scale=self.ssaa_scale,
            downsample_prefilter_sigma=self.downsample_prefilter_sigma,
            downsample_filter_order=self.downsample_filter_order,
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
        height=1.0,           # カメラ高さ
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
        center_y_by_shape=(-0.56, -0.26, -0.40, -0.22),
        ground_clearance_by_shape=(0.0, 0.0, 0.0, 0.0),  # auto_grounding時の床からの余白
        center_x=0.0,
        center_z=0.0,
        global_scale_multiplier=1.0,
        use_auto_grounding=True,
    )

    renderer = Render3DShapesModule(
        hue_v=0.9,
        orientation_period=1.0,
        shadow_strength=0.8,
        ssaa_scale=2,
        image_size=128,
        lighting_config=light_cfg,
        mesh_resolution_config=mesh_res_cfg,
        camera_config=cam_cfg,
        room_config=room_cfg,
        object_config=obj_cfg,
        output_chw=True,
        downsample_prefilter_sigma=0.0,
        downsample_filter_order="post",
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
        # orientation_period=1.0,
        # lighting_config=light_cfg,
        # mesh_resolution_config=mesh_res_cfg,
        # camera_config=cam_cfg,
        # room_config=room_cfg,
        # object_config=obj_cfg,
        # output_chw=True,
        # shadow_strength=10,
        # ssaa_scale=8,
        # image_size=64,
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

    chunk_size = 64

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
