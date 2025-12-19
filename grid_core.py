from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from pykinect2024 import PyKinectRuntime, PyKinect2024

warnings.filterwarnings("ignore", category=FutureWarning)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < eps else (v / n)


@dataclass
class Plane:
    # Plane equation in camera space: n·p + d = 0 (n is unit length)
    n: np.ndarray  # shape (3,)
    d: float

    def flipped(self) -> "Plane":
        return Plane(n=-self.n, d=-self.d)


@dataclass
class GridFrame:
    """
    Defines a 2D coordinate system on the floor:
      - origin_cam: 3D point on the floor plane (meters, camera space)
      - e1: X-axis unit vector on the floor (camera space)
      - e2: Y-axis unit vector on the floor (camera space)
    """
    plane: Plane
    origin_cam: np.ndarray  # shape (3,)
    e1: np.ndarray          # shape (3,)
    e2: np.ndarray          # shape (3,)


class PlaneSmoother:
    """EMA smoothing so the estimated floor plane doesn't jitter."""
    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha
        self._n: Optional[np.ndarray] = None
        self._d: Optional[float] = None

    def reset(self) -> None:
        self._n = None
        self._d = None

    def update(self, plane: Plane) -> Plane:
        n = normalize(plane.n)
        d = float(plane.d)

        if self._n is None:
            self._n = n.copy()
            self._d = d
            return Plane(self._n, self._d)

        # keep direction consistent
        if float(np.dot(self._n, n)) < 0.0:
            n = -n
            d = -d

        self._n = normalize((1.0 - self.alpha) * self._n + self.alpha * n)
        self._d = (1.0 - self.alpha) * self._d + self.alpha * d
        return Plane(self._n, self._d)


def build_plane_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build two orthonormal vectors e1,e2 that lie on the plane.
    e1 = camera +X projected into plane; e2 = n x e1.
    """
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e1 = x_axis - float(np.dot(x_axis, n)) * n
    e1 = normalize(e1)
    if float(np.linalg.norm(e1)) < 1e-6:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        e1 = normalize(z_axis - float(np.dot(z_axis, n)) * n)
    e2 = normalize(np.cross(n, e1))
    return e1, e2


def project_point_to_plane(p: np.ndarray, plane: Plane) -> np.ndarray:
    """
    Orthogonal projection of a 3D point onto the plane along the plane normal.
    """
    # signed distance: n·p + d
    dist = float(np.dot(plane.n, p) + plane.d)
    return p - dist * plane.n


def camera_to_grid_xy(p_cam: np.ndarray, frame: GridFrame) -> Tuple[float, float]:
    """
    Convert a 3D camera-space point on the floor to (x,y) in the grid frame (meters).
    """
    v = p_cam - frame.origin_cam
    x = float(np.dot(v, frame.e1))
    y = float(np.dot(v, frame.e2))
    return x, y


def grid_xy_to_camera(x: float, y: float, frame: GridFrame) -> np.ndarray:
    """
    Convert (x,y) in grid frame to a 3D camera-space point on the floor plane.
    """
    return frame.origin_cam + x * frame.e1 + y * frame.e2


def map_camera_point_to_color_xy(kinect, p_cam: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Project a 3D camera-space point into color pixel coordinates.
    """
    mapper = getattr(kinect, "_mapper", None)
    if mapper is None or not hasattr(mapper, "MapCameraPointToColorSpace"):
        return None
    try:
        cam_pt = PyKinect2024._CameraSpacePoint(float(p_cam[0]), float(p_cam[1]), float(p_cam[2]))
        cp = mapper.MapCameraPointToColorSpace(cam_pt)
        u, v = float(cp.x), float(cp.y)
        if not (np.isfinite(u) and np.isfinite(v)):
            return None
        return u, v
    except Exception:
        return None


def depth_pixel_to_camera_point(kinect, u: int, v: int, depth_mm: int) -> Optional[np.ndarray]:
    """
    Depth pixel (u,v) with depth in mm -> 3D camera-space point (meters).
    """
    mapper = getattr(kinect, "_mapper", None)
    if mapper is None or not hasattr(mapper, "MapDepthPointToCameraSpace"):
        return None
    if depth_mm <= 0:
        return None
    try:
        dp = PyKinect2024._DepthSpacePoint(float(u), float(v))
        cp = mapper.MapDepthPointToCameraSpace(dp, int(depth_mm))
        return np.array([float(cp.x), float(cp.y), float(cp.z)], dtype=np.float64)
    except Exception:
        return None


def map_depth_pixel_to_color_xy(kinect, depth_frame_1d: np.ndarray, depth_w: int, u: int, v: int) -> Optional[Tuple[float, float]]:
    """
    Map a single depth pixel (u,v) to a color pixel using its measured depth.
    This is used to invert color->depth by searching.
    """
    mapper = getattr(kinect, "_mapper", None)
    if mapper is None or not hasattr(mapper, "MapDepthPointToColorSpace"):
        return None

    idx = v * depth_w + u
    depth_mm = int(depth_frame_1d[idx])
    if depth_mm <= 0:
        return None

    try:
        dp = PyKinect2024._DepthSpacePoint(float(u), float(v))
        cp = mapper.MapDepthPointToColorSpace(dp, depth_mm)
        x = float(cp.x)
        y = float(cp.y)
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return x, y
    except Exception:
        return None


def find_depth_pixel_for_color_xy(
    kinect,
    depth_frame_1d: np.ndarray,
    depth_w: int,
    depth_h: int,
    target_color_xy: Tuple[float, float],
    seed_depth_uv: Optional[Tuple[int, int]],
    search_radius: int,
) -> Optional[Tuple[int, int]]:
    """
    Find the depth pixel whose projection into the color image is closest to (target_color_xy).

    We do a coarse-to-fine search to keep it fast.
    """
    tx, ty = float(target_color_xy[0]), float(target_color_xy[1])

    if seed_depth_uv is None:
        seed_u, seed_v = depth_w // 2, depth_h // 2
    else:
        seed_u, seed_v = seed_depth_uv

    def search(center_u: int, center_v: int, radius: int, step: int) -> Optional[Tuple[int, int, float]]:
        umin = max(0, center_u - radius)
        umax = min(depth_w - 1, center_u + radius)
        vmin = max(0, center_v - radius)
        vmax = min(depth_h - 1, center_v + radius)

        best: Optional[Tuple[int, int, float]] = None
        for v in range(vmin, vmax + 1, step):
            for u in range(umin, umax + 1, step):
                uv = map_depth_pixel_to_color_xy(kinect, depth_frame_1d, depth_w, u, v)
                if uv is None:
                    continue
                x, y = uv
                dx = x - tx
                dy = y - ty
                d2 = dx * dx + dy * dy
                if best is None or d2 < best[2]:
                    best = (u, v, d2)
        return best

    b1 = search(seed_u, seed_v, radius=search_radius, step=10)
    if b1 is None:
        return None
    b2 = search(b1[0], b1[1], radius=25, step=4) or b1
    b3 = search(b2[0], b2[1], radius=7, step=1) or b2
    return int(b3[0]), int(b3[1])


# ---------------- Plane fitting (from depth) ----------------
def sample_depth_pixels(
    depth_frame_1d: np.ndarray,
    depth_w: int,
    depth_h: int,
    max_samples: int,
    rng: np.random.Generator,
    roi_x_frac: Tuple[float, float],
    roi_y_frac: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    x0 = int(depth_w * roi_x_frac[0])
    x1 = int(depth_w * roi_x_frac[1])
    y0 = int(depth_h * roi_y_frac[0])
    y1 = int(depth_h * roi_y_frac[1])

    xs = rng.integers(x0, x1, size=max_samples, endpoint=False)
    ys = rng.integers(y0, y1, size=max_samples, endpoint=False)
    idx = ys * depth_w + xs

    z = depth_frame_1d[idx].astype(np.int32)
    valid = z > 0
    xs = xs[valid]
    ys = ys[valid]
    z = z[valid]
    pixels_uv = np.stack([xs, ys], axis=1)
    return pixels_uv, z


def pixels_to_camera_points(kinect, pixels_uv: np.ndarray, depths_mm: np.ndarray) -> np.ndarray:
    mapper = getattr(kinect, "_mapper", None)
    if mapper is None or not hasattr(mapper, "MapDepthPointToCameraSpace"):
        return np.empty((0, 3), dtype=np.float64)

    pts = np.empty((len(depths_mm), 3), dtype=np.float64)
    for i, ((u, v), dmm) in enumerate(zip(pixels_uv, depths_mm)):
        dp = PyKinect2024._DepthSpacePoint(float(u), float(v))
        cp = mapper.MapDepthPointToCameraSpace(dp, int(dmm))
        pts[i, 0] = float(cp.x)
        pts[i, 1] = float(cp.y)
        pts[i, 2] = float(cp.z)
    return pts


def fit_plane_svd(points: np.ndarray) -> Optional[Plane]:
    if points.shape[0] < 3:
        return None
    centroid = points.mean(axis=0)
    X = points - centroid
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    n = normalize(vt[-1])
    d = -float(np.dot(n, centroid))
    return Plane(n=n, d=d)


def ransac_plane(points: np.ndarray, iters: int, thresh_m: float, rng: np.random.Generator) -> Optional[Plane]:
    if points.shape[0] < 100:
        return None

    best_inliers = 0
    best_mask: Optional[np.ndarray] = None
    best_plane: Optional[Plane] = None

    N = points.shape[0]
    for _ in range(iters):
        i0, i1, i2 = rng.integers(0, N, size=3)
        p0, p1, p2 = points[i0], points[i1], points[i2]

        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        if float(np.linalg.norm(n)) < 1e-9:
            continue
        n = normalize(n)
        d = -float(np.dot(n, p0))

        dist = np.abs(points @ n + d)
        mask = dist < thresh_m
        inliers = int(mask.sum())
        if inliers > best_inliers:
            best_inliers = inliers
            best_mask = mask
            best_plane = Plane(n=n, d=d)

    if best_plane is None or best_mask is None or best_inliers < 100:
        return None

    refined = fit_plane_svd(points[best_mask])
    return refined or best_plane


def estimate_floor_plane_from_depth(
    kinect,
    depth_frame_1d: np.ndarray,
    depth_w: int,
    depth_h: int,
    rng: np.random.Generator,
    max_samples: int = 8000,
    iters: int = 140,
    thresh_m: float = 0.015,
    roi_x_frac: Tuple[float, float] = (0.10, 0.90),
    roi_y_frac: Tuple[float, float] = (0.20, 0.95),
) -> Optional[Plane]:
    pixels_uv, depths_mm = sample_depth_pixels(
        depth_frame_1d, depth_w, depth_h, max_samples, rng, roi_x_frac, roi_y_frac
    )
    if len(depths_mm) < 500:
        return None
    pts = pixels_to_camera_points(kinect, pixels_uv, depths_mm)
    if pts.shape[0] < 500:
        return None
    return ransac_plane(pts, iters=iters, thresh_m=thresh_m, rng=rng)


class KinectGridSystem:
    """
    Owns the Kinect runtime and maintains:
      - floor plane (calibrate + lock)
      - grid frame origin (set by mouse click on taped center)
    """
    def __init__(
        self,
        plane_smooth_alpha: float = 0.15,
    ) -> None:
        sources = PyKinect2024.FrameSourceTypes_Color | PyKinect2024.FrameSourceTypes_Depth
        self.kinect = PyKinectRuntime.PyKinectRuntime(sources)

        self.color_w = self.kinect.color_frame_desc.Width
        self.color_h = self.kinect.color_frame_desc.Height
        self.depth_w = self.kinect.depth_frame_desc.Width
        self.depth_h = self.kinect.depth_frame_desc.Height

        self.rng = np.random.default_rng(0)
        self.smoother = PlaneSmoother(alpha=plane_smooth_alpha)

        self.plane: Optional[Plane] = None
        self.plane_locked: bool = False
        self.fit_count: int = 0

        self.last_color_1d: Optional[np.ndarray] = None
        self.last_depth_1d: Optional[np.ndarray] = None

        self.grid_frame: Optional[GridFrame] = None

        # Seeds for mapping (speed)
        self._seed_center_depth_uv: Optional[Tuple[int, int]] = None

    def close(self) -> None:
        try:
            self.kinect.close()
        except Exception:
            pass

    def update_frames(self) -> bool:
        """
        Returns True if we got a new color frame this iteration.
        """
        if not self.kinect.has_new_color_frame():
            return False
        self.last_color_1d = self.kinect.get_last_color_frame()

        if self.kinect.has_new_depth_frame():
            self.last_depth_1d = self.kinect.get_last_depth_frame()
        return True

    def recalibrate_plane(self) -> None:
        self.plane = None
        self.plane_locked = False
        self.fit_count = 0
        self.smoother.reset()

    def try_update_plane(
        self,
        fit_every_n_frames: int,
        frame_idx: int,
        fits_to_lock: int,
        max_samples: int,
        ransac_iters: int,
        inlier_thresh_m: float,
        roi_x_frac: Tuple[float, float],
        roi_y_frac: Tuple[float, float],
    ) -> None:
        if self.plane_locked:
            return
        if self.last_depth_1d is None:
            return
        if frame_idx % fit_every_n_frames != 0:
            return

        new_plane = estimate_floor_plane_from_depth(
            self.kinect,
            self.last_depth_1d,
            self.depth_w,
            self.depth_h,
            self.rng,
            max_samples=max_samples,
            iters=ransac_iters,
            thresh_m=inlier_thresh_m,
            roi_x_frac=roi_x_frac,
            roi_y_frac=roi_y_frac,
        )
        if new_plane is None:
            return

        self.plane = self.smoother.update(new_plane)
        self.fit_count += 1
        if self.fit_count >= fits_to_lock:
            self.plane_locked = True

        # If a grid frame already exists, keep it (plane changes should be stopped after lock anyway).
        # If you recalibrate, you should re-click the grid center.

    def set_grid_center_from_color_click(
        self,
        color_xy: Tuple[float, float],
        search_radius: int = 160,
    ) -> bool:
        """
        Convert a clicked COLOR pixel (where you taped the grid center on the floor)
        into a 3D floor point and set the grid origin there.
        """
        if self.plane is None or self.last_depth_1d is None:
            return False

        # Find the depth pixel that corresponds to this color pixel
        depth_uv = find_depth_pixel_for_color_xy(
            self.kinect,
            self.last_depth_1d,
            self.depth_w,
            self.depth_h,
            target_color_xy=color_xy,
            seed_depth_uv=self._seed_center_depth_uv,
            search_radius=search_radius,
        )
        if depth_uv is None:
            return False
        self._seed_center_depth_uv = depth_uv

        u, v = depth_uv
        depth_mm = int(self.last_depth_1d[v * self.depth_w + u])
        p_cam = depth_pixel_to_camera_point(self.kinect, u, v, depth_mm)
        if p_cam is None:
            return False

        # clicked tape is on floor; still project to plane for robustness
        p_floor = project_point_to_plane(p_cam, self.plane)

        e1, e2 = build_plane_basis(self.plane.n)
        self.grid_frame = GridFrame(plane=self.plane, origin_cam=p_floor, e1=e1, e2=e2)
        return True

    def get_color_bgr(self) -> Optional[np.ndarray]:
        """
        Convert last color frame to BGR image.
        """
        if self.last_color_1d is None:
            return None
        bgra = self.last_color_1d.reshape((self.color_h, self.color_w, 4)).astype(np.uint8, copy=False)
        # Avoid importing cv2 here to keep module dependency light; main_nav handles conversion.
        # But we still return BGRA for main_nav to convert.
        return bgra