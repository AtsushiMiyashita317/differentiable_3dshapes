from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
from PIL import Image
import torch
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


def _downsample_mean(image_hwc: np.ndarray | torch.Tensor, factor: int) -> np.ndarray | torch.Tensor:
    if factor <= 1:
        return image_hwc
    h, w, c = image_hwc.shape
    if torch.is_tensor(image_hwc):
        # Use area interpolation for better antialiasing quality than block mean.
        x = image_hwc.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        x = F.interpolate(x, size=(h // factor, w // factor), mode="area")
        return x.squeeze(0).permute(1, 2, 0)
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

    m = torch.tensor([0.5, 0.5, 0.5], device=h.device, dtype=h.dtype)
    p0 = c0 - m
    rho2 = torch.sum(p0 * p0, dim=-1, keepdim=True)
    eps = torch.tensor(1e-12, device=h.device, dtype=h.dtype)
    rho = torch.sqrt(torch.clamp(rho2, min=eps))
    u = p0 / rho

    n = torch.tensor([1.0, 1.0, 1.0], device=h.device, dtype=h.dtype) / math.sqrt(3.0)
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
        closest = torch.stack([torch.zeros_like(px), cy, torch.zeros_like(pz)], dim=-1)
        d_local = torch.linalg.norm(p_local - closest, dim=-1) - 1.0

    return d_local * torch.min(scale_xyz)


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
    h, w, _ = ray_dir.shape
    t = torch.full((h, w), 1e-3, device=ray_dir.device, dtype=ray_dir.dtype)
    hit = torch.zeros((h, w), device=ray_dir.device, dtype=torch.bool)

    for _ in range(max_steps):
        p = cam_pos[None, None, :] + ray_dir * t[..., None]
        d = _sdf_object_world(p, shape_id, center_world, scale_xyz)
        active = (~hit) & (t < t_far)
        new_hit = active & (torch.abs(d) < eps)
        hit = hit | new_hit
        step = torch.clamp(d, min=5e-4, max=0.5)
        t = torch.where(active & (~new_hit), t + step, t)

    return hit & (t < t_far), t


def _normal_from_sdf(
    p_world: torch.Tensor,
    shape_id: int,
    center_world: torch.Tensor,
    scale_xyz: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    ex = torch.tensor([eps, 0.0, 0.0], device=p_world.device, dtype=p_world.dtype)
    ey = torch.tensor([0.0, eps, 0.0], device=p_world.device, dtype=p_world.dtype)
    ez = torch.tensor([0.0, 0.0, eps], device=p_world.device, dtype=p_world.dtype)
    nx = _sdf_object_world(p_world + ex, shape_id, center_world, scale_xyz) - _sdf_object_world(
        p_world - ex, shape_id, center_world, scale_xyz
    )
    ny = _sdf_object_world(p_world + ey, shape_id, center_world, scale_xyz) - _sdf_object_world(
        p_world - ey, shape_id, center_world, scale_xyz
    )
    nz = _sdf_object_world(p_world + ez, shape_id, center_world, scale_xyz) - _sdf_object_world(
        p_world - ez, shape_id, center_world, scale_xyz
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w, _ = ray_dir.shape
    dtype = ray_dir.dtype
    device = ray_dir.device

    y0 = _as_torch_scalar(room_config.floor_y, device, dtype)
    y1 = _as_torch_scalar(room_config.wall_top_y, device, dtype)
    half_w = _as_torch_scalar(room_config.half_width, device, dtype)
    half_d = _as_torch_scalar(room_config.half_depth, device, dtype)
    x0, x1 = -half_w, half_w
    z0, z1 = -half_d, half_d

    sky = torch.tensor([0.72, 0.88, 1.0], device=device, dtype=dtype)
    color = sky[None, None, :].expand(h, w, 3).clone()
    depth = torch.full((h, w), float("inf"), device=device, dtype=dtype)
    surface_id = torch.full((h, w), -1, device=device, dtype=torch.int64)
    hit_points = torch.zeros((h, w, 3), device=device, dtype=dtype)

    def safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
        eps = torch.tensor(1e-8, device=device, dtype=dtype)
        den_safe = torch.where(torch.abs(den) > eps, den, torch.where(den >= 0.0, eps, -eps))
        return num / den_safe

    def plane_update(
        den: torch.Tensor,
        t: torch.Tensor,
        p: torch.Tensor,
        valid: torch.Tensor,
        normal: torch.Tensor,
        base_rgb: torch.Tensor,
        surf_id: int,
    ) -> None:
        nonlocal color, depth, surface_id, hit_points
        z_cam = torch.sum((p - cam_pos[None, None, :]) * ray_dir, dim=-1)
        l = light_pos_world[None, None, :] - p
        l = l / torch.clamp(torch.linalg.norm(l, dim=-1, keepdim=True), min=1e-8)
        ndotl = torch.clamp(torch.sum(l * normal[None, None, :], dim=-1), min=0.0)
        lit = ambient + diffuse * ndotl
        c = torch.clamp(base_rgb[None, None, :] * lit[..., None], 0.0, 1.0)
        closer = valid & (t > 1e-6) & (z_cam > 1e-6) & (z_cam < depth)
        depth = torch.where(closer, z_cam, depth)
        color = torch.where(closer[..., None], c, color)
        surface_id = torch.where(closer, torch.full_like(surface_id, surf_id), surface_id)
        hit_points = torch.where(closer[..., None], p, hit_points)

    # floor
    den = ray_dir[..., 1]
    t = safe_div(y0 - cam_pos[1], den)
    p = cam_pos[None, None, :] + t[..., None] * ray_dir
    valid = (torch.abs(den) > 1e-8) & (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 2] >= z0) & (p[..., 2] <= z1)
    plane_update(den, t, p, valid, torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype), floor_rgb, surf_id=0)

    if room_config.draw_back_wall:
        den = ray_dir[..., 2]
        t = safe_div(z0 - cam_pos[2], den)
        p = cam_pos[None, None, :] + t[..., None] * ray_dir
        valid = (torch.abs(den) > 1e-8) & (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)
        plane_update(den, t, p, valid, torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype), wall_rgb, surf_id=1)
    if room_config.draw_front_wall:
        den = ray_dir[..., 2]
        t = safe_div(z1 - cam_pos[2], den)
        p = cam_pos[None, None, :] + t[..., None] * ray_dir
        valid = (torch.abs(den) > 1e-8) & (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)
        plane_update(den, t, p, valid, torch.tensor([0.0, 0.0, -1.0], device=device, dtype=dtype), wall_rgb, surf_id=2)
    if room_config.draw_left_wall:
        den = ray_dir[..., 0]
        t = safe_div(x0 - cam_pos[0], den)
        p = cam_pos[None, None, :] + t[..., None] * ray_dir
        valid = (torch.abs(den) > 1e-8) & (p[..., 2] >= z0) & (p[..., 2] <= z1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)
        plane_update(den, t, p, valid, torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype), wall_rgb, surf_id=3)
    if room_config.draw_right_wall:
        den = ray_dir[..., 0]
        t = safe_div(x1 - cam_pos[0], den)
        p = cam_pos[None, None, :] + t[..., None] * ray_dir
        valid = (torch.abs(den) > 1e-8) & (p[..., 2] >= z0) & (p[..., 2] <= z1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)
        plane_update(den, t, p, valid, torch.tensor([-1.0, 0.0, 0.0], device=device, dtype=dtype), wall_rgb, surf_id=4)

    floor_mask = surface_id == 0
    return color, depth, floor_mask, hit_points


def _soft_shadow_floor(
    floor_points: torch.Tensor,  # (N,3)
    light_pos_world: torch.Tensor,
    shape_id: int,
    center_world: torch.Tensor,
    scale_xyz: torch.Tensor,
    n_samples: int = 24,
    sigma: float = 0.02,
) -> torch.Tensor:
    if floor_points.numel() == 0:
        return torch.empty((0,), device=floor_points.device, dtype=floor_points.dtype)
    s = torch.linspace(0.02, 0.98, n_samples, device=floor_points.device, dtype=floor_points.dtype)
    ray = light_pos_world[None, :] - floor_points
    dist = torch.linalg.norm(ray, dim=-1, keepdim=True)
    ray_dir = ray / torch.clamp(dist, min=1e-8)
    samples = floor_points[:, None, :] + ray_dir[:, None, :] * (dist[:, None, :] * s[None, :, None])
    d = _sdf_object_world(samples.reshape(-1, 3), shape_id, center_world, scale_xyz).reshape(floor_points.shape[0], n_samples)
    occ = torch.sigmoid((-d) / sigma)
    # Visibility in [0,1], stronger than mean-transmittance and closer to hard-shadow behavior.
    vis = 1.0 - occ.max(dim=-1).values
    return torch.clamp(vis, 0.0, 1.0)


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
    image_size: int | Tuple[int, int] = 64,
    lighting_config: LightingConfig | None = None,
    mesh_resolution_config: MeshResolutionConfig | None = None,
    camera_config: CameraConfig | None = None,
    room_config: RoomConfig | None = None,
    object_config: ObjectConfig | None = None,
    output_chw: bool = True,
) -> torch.Tensor:
    """Differentiable torch renderer.

    Gradients are available for tensor inputs in main factors and also scalar/vector
    fields inside camera/lighting/room/object configs when passed as torch tensors.
    """
    refs = [size, orientation, floor_hue, wall_hue, object_hue, hue_v]
    t_ref = next((x for x in refs if torch.is_tensor(x)), None)
    device = t_ref.device if t_ref is not None else torch.device("cpu")
    dtype = t_ref.dtype if t_ref is not None else torch.float32

    lighting_config = lighting_config or DEFAULT_LIGHTING_CONFIG
    mesh_resolution_config = mesh_resolution_config or DEFAULT_MESH_RESOLUTION_CONFIG
    camera_config = camera_config or DEFAULT_CAMERA_CONFIG
    room_config = room_config or DEFAULT_ROOM_CONFIG
    object_config = object_config or DEFAULT_OBJECT_CONFIG

    size_t = _as_torch_scalar(size, device, dtype).clamp(0.4, 1.8)
    ori_t = _as_torch_scalar(orientation, device, dtype)
    floor_h = _as_torch_scalar(floor_hue, device, dtype)
    wall_h = _as_torch_scalar(wall_hue, device, dtype)
    obj_h = _as_torch_scalar(object_hue, device, dtype)
    hue_v_t = _as_torch_scalar(hue_v, device, dtype).clamp(0.0, 1.0)

    orientation_period_t = _as_torch_scalar(orientation_period, device, dtype)
    period_eps = torch.tensor(1e-8, device=device, dtype=dtype)
    orientation_period_safe = torch.where(
        torch.abs(orientation_period_t) > period_eps,
        orientation_period_t,
        torch.where(orientation_period_t >= 0.0, period_eps, -period_eps),
    )
    yaw_offset_deg_t = _as_torch_scalar(camera_config.yaw_offset_deg, device, dtype)
    theta = 2.0 * torch.pi * torch.remainder(ori_t / orientation_period_safe, 1.0) + (
        yaw_offset_deg_t * (torch.pi / 180.0)
    )

    ssaa_scale = int(max(1, ssaa_scale))
    h0, w0 = _parse_image_size(image_size)
    h = h0 * ssaa_scale
    w = w0 * ssaa_scale
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5 - (w - 1) * 0.5) / (48.0 * (w / 64.0))
    ys = -((torch.arange(h, device=device, dtype=dtype) + 0.5 - (h - 1) * 0.5) / (48.0 * (w / 64.0)))
    dx, dy = torch.meshgrid(xs, ys, indexing="xy")

    cam_radius = _as_torch_scalar(camera_config.radius, device, dtype)
    cam_height = _as_torch_scalar(camera_config.height, device, dtype)
    target = _as_torch_vec3(camera_config.target, device, dtype)
    cam_pos = torch.stack([cam_radius * torch.sin(theta), cam_height, cam_radius * torch.cos(theta)])

    forward = (target - cam_pos)
    forward = forward / torch.clamp(torch.linalg.norm(forward), min=1e-8)
    up_hint = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    right = torch.cross(forward, up_hint, dim=0)
    right = right / torch.clamp(torch.linalg.norm(right), min=1e-8)
    up = torch.cross(right, forward, dim=0)
    up = up / torch.clamp(torch.linalg.norm(up), min=1e-8)

    ray_dir = right[None, None, :] * dx[..., None] + up[None, None, :] * dy[..., None] + forward[None, None, :]
    ray_dir = ray_dir / torch.clamp(torch.linalg.norm(ray_dir, dim=-1, keepdim=True), min=1e-8)

    floor_rgb = _hue_to_rgb_constrained(floor_h, hue_v_t)
    wall_rgb = _hue_to_rgb_constrained(wall_h, hue_v_t)
    obj_rgb = _hue_to_rgb_constrained(obj_h, hue_v_t)

    light_base = _as_torch_vec3(lighting_config.light_position_world, device, dtype)
    c = torch.cos(theta)
    s = torch.sin(theta)
    z = torch.tensor(0.0, device=device, dtype=dtype)
    o = torch.tensor(1.0, device=device, dtype=dtype)
    rot = torch.stack(
        [
            torch.stack([c, z, s]),
            torch.stack([z, o, z]),
            torch.stack([-s, z, c]),
        ]
    )
    light_pos_world = rot @ light_base
    ambient = _as_torch_scalar(lighting_config.ambient, device, dtype).clamp(0.0, 1.0)
    diffuse = _as_torch_scalar(lighting_config.diffuse, device, dtype).clamp(0.0, 2.0)

    shape_id = int(max(0, min(3, int(shape))))
    meshes = _get_meshes(
        cylinder_radial_steps=mesh_resolution_config.cylinder_radial_steps,
        sphere_lat_steps=mesh_resolution_config.sphere_lat_steps,
        sphere_lon_steps=mesh_resolution_config.sphere_lon_steps,
        capsule_radial_steps=mesh_resolution_config.capsule_radial_steps,
        capsule_hemi_steps=mesh_resolution_config.capsule_hemi_steps,
    )
    mesh = meshes[shape_id]
    base_scale = _as_torch_shape_vec3(object_config.base_scales, shape_id, device, dtype)
    base_scale = base_scale * _as_torch_scalar(object_config.global_scale_multiplier, device, dtype)
    scale = base_scale * size_t

    extent_y = torch.max(torch.abs(torch.tensor(mesh.verts[:, 1], device=device, dtype=dtype)))
    if object_config.use_auto_grounding:
        clearance = _as_torch_shape_scalar(object_config.ground_clearance_by_shape, shape_id, device, dtype)
        center_y = _as_torch_scalar(room_config.floor_y, device, dtype) + clearance + extent_y * scale[1]
    else:
        center_y = _as_torch_shape_scalar(object_config.center_y_by_shape, shape_id, device, dtype)
    center_world = torch.stack(
        [
            _as_torch_scalar(object_config.center_x, device, dtype),
            center_y,
            _as_torch_scalar(object_config.center_z, device, dtype),
        ]
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
    )

    if floor_mask.any():
        flat_idx = floor_mask.reshape(-1)
        pts = floor_pts.reshape(-1, 3)[flat_idx]
        vis = _soft_shadow_floor(
        floor_points=pts,
        light_pos_world=light_pos_world,
        shape_id=shape_id,
        center_world=center_world,
        scale_xyz=scale,
            n_samples=24,
            sigma=0.03,
        )
        l = light_pos_world[None, :] - pts
        l = l / torch.clamp(torch.linalg.norm(l, dim=-1, keepdim=True), min=1e-8)
        ndotl = torch.clamp(l[:, 1], min=0.0)
        s = _as_torch_scalar(shadow_strength, device, dtype).clamp(0.0, 1.0)
        # Match numpy semantics: ambient is preserved, direct term is reduced by shadow.
        lit = ambient + diffuse * ndotl * (1.0 - s * (1.0 - vis))
        floor_shaded = torch.clamp(floor_rgb[None, :] * lit[:, None], 0.0, 1.0)

        flat_color = room_color.reshape(-1, 3)
        flat_color[flat_idx] = floor_shaded
        room_color = flat_color.reshape(h, w, 3)

    hit_obj, t_obj = _ray_march_object(
        cam_pos=cam_pos,
        ray_dir=ray_dir,
        shape_id=shape_id,
        center_world=center_world,
        scale_xyz=scale,
    )
    p_obj = cam_pos[None, None, :] + ray_dir * t_obj[..., None]
    n_obj = _normal_from_sdf(p_obj.reshape(-1, 3), shape_id, center_world, scale).reshape(h, w, 3)
    l_obj = light_pos_world[None, None, :] - p_obj
    l_obj = l_obj / torch.clamp(torch.linalg.norm(l_obj, dim=-1, keepdim=True), min=1e-8)
    ndotl_obj = torch.clamp(torch.sum(n_obj * l_obj, dim=-1), min=0.0)
    obj_lit = ambient + diffuse * ndotl_obj
    obj_color = torch.clamp(obj_rgb[None, None, :] * obj_lit[..., None], 0.0, 1.0)

    obj_front = hit_obj & (t_obj < room_depth)
    out = torch.where(obj_front[..., None], obj_color, room_color)

    if ssaa_scale > 1:
        out = _downsample_mean(out, ssaa_scale)
    if output_chw:
        out = out.permute(2, 0, 1)
    return out


if __name__ == "__main__":
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

    out_dir = Path("samples")
    out_dir.mkdir(exist_ok=True)
    main_params = dict(
        shape=2,
        size=1.0,
        orientation=0.0,
        floor_hue=0.0,
        wall_hue=0.0,
        object_hue=0.0,
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

    def _render_np(local_params: dict) -> np.ndarray:
        with torch.no_grad():
            return render_3dshapes_image(**local_params).cpu().numpy()

    def _autograd_diff_rgb(local_params: dict, key: str, val: float) -> np.ndarray:
        x = torch.tensor(float(val), dtype=torch.float32)

        def fn(xx: torch.Tensor) -> torch.Tensor:
            p = dict(local_params)
            p[key] = xx
            return render_3dshapes_image(**p)

        _, dimg = torch.autograd.functional.jvp(
            fn,
            (x,),
            (torch.ones_like(x),),
            create_graph=False,
            strict=False,
        )
        diff = dimg.detach().cpu().numpy()
        diff_hwc = np.transpose(diff, (1, 2, 0))  # (H, W, 3)
        scale = float(np.percentile(np.abs(diff_hwc), 99.5))
        if scale < 1e-8:
            scale = 1.0
        # Signed visualization: 0 -> gray(0.5), positive -> brighter, negative -> darker.
        diff_vis = 0.5 + 0.5 * (diff_hwc / scale)
        return np.clip(diff_vis, 0.0, 1.0).astype(np.float32)

    # Save one still image with current main settings.
    still = _render_np(main_params)
    still_path = out_dir / "main_render.png"
    still_hwc = np.transpose(still, (1, 2, 0))
    Image.fromarray((np.clip(still_hwc, 0.0, 1.0) * 255).astype(np.uint8)).save(still_path)
    print("saved:", still_path)

    # Build GIF frames:
    # rows = orientation / size / floor_hue / wall_hue / object_hue, cols = image | differential.
    n_frames = 36
    orientation_vals = np.linspace(0.0, 1.0, n_frames, endpoint=False, dtype=np.float32)
    size_vals = np.linspace(0.7, 1.5, n_frames, endpoint=True, dtype=np.float32)
    color_vals = np.linspace(0.0, 1.0, n_frames, endpoint=False, dtype=np.float32)

    frame_images: list[Image.Image] = []
    for i in range(n_frames):
        frame_rows = []
        sweep_specs = [
            ("orientation", float(orientation_vals[i])),
            ("size", float(size_vals[i])),
            ("floor_hue", float(color_vals[i])),
            ("wall_hue", float(color_vals[i])),
            ("object_hue", float(color_vals[i])),
        ]
        for key, val in sweep_specs:
            p = dict(main_params)
            p[key] = val
            img = _render_np(p)
            img_hwc = np.transpose(img, (1, 2, 0))
            diff_hwc = _autograd_diff_rgb(main_params, key=key, val=val)
            row = np.concatenate([img_hwc, diff_hwc], axis=1)
            frame_rows.append(row)

        frame = np.concatenate(frame_rows, axis=0)
        frame_u8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
        frame_images.append(Image.fromarray(frame_u8))
        print(f"frame {i + 1}/{n_frames} done")

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
