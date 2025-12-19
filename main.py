from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from pykinect2024 import PyKinectRuntime, PyKinect2024

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================
# User configuration
# =========================
WINDOW_NAME = "Kinect v2 - Color"

# IMPORTANT:
# We will *not* rely on OpenCV’s window scaling for mouse coordinates.
# Instead, we render a *separately resized* display image at this fixed scale.
# This makes mouse clicks map reliably to the original 1920x1080 color image.
DISPLAY_SCALE = 0.6  # change this to make the window bigger/smaller

GRID_SPACING_M = 0.60
GRID_HALF_EXTENT = 1  # 3x3 grid
POINT_RADIUS = 6

POINT_COLOR = (0, 255, 0)
AXIS_X_COLOR = (0, 0, 255)
AXIS_Y_COLOR = (255, 0, 0)
HUD_COLOR = (255, 255, 255)
CLICK_MARK_COLOR = (255, 0, 255)

# Floor plane estimation (from depth)
MAX_SAMPLES = 8000
RANSAC_ITERS = 140
INLIER_THRESH_M = 0.015
FIT_EVERY_N_FRAMES = 10
PLANE_SMOOTH_ALPHA = 0.15

# Warm-up then lock
CALIBRATION_FITS_TO_LOCK = 25

# Depth ROI (tune if needed)
ROI_X_FRAC = (0.10, 0.90)
ROI_Y_FRAC = (0.20, 0.95)

# Mapping: find depth pixel whose projection to color is closest to a target color pixel
SEARCH_RADIUS_DEFAULT = 160
MANUAL_ORIGIN_UPDATE_EVERY = 10
CENTER_ORIGIN_UPDATE_EVERY = 30
# =========================


def color_frame_to_bgr(frame_1d: np.ndarray, width: int, height: int) -> np.ndarray:
    bgra = frame_1d.reshape((height, width, 4)).astype(np.uint8, copy=False)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < eps else (v / n)


def draw_hud(img: np.ndarray, lines: list[str]) -> None:
    y = 26
    for s in lines:
        cv2.putText(img, s, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, HUD_COLOR, 2, cv2.LINE_AA)
        y += 26


@dataclass
class Plane:
    # n·p + d = 0 (n is unit)
    n: np.ndarray
    d: float

    def flipped(self) -> "Plane":
        return Plane(n=-self.n, d=-self.d)


class PlaneSmoother:
    def __init__(self, alpha: float):
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

        # Keep direction consistent
        if float(np.dot(self._n, n)) < 0.0:
            n = -n
            d = -d

        self._n = normalize((1 - self.alpha) * self._n + self.alpha * n)
        self._d = (1 - self.alpha) * self._d + self.alpha * d
        return Plane(self._n, self._d)


def intersect_ray_plane(ray_dir: np.ndarray, plane: Plane) -> Optional[np.ndarray]:
    denom = float(np.dot(plane.n, ray_dir))
    if abs(denom) < 1e-9:
        return None
    t = -plane.d / denom
    if t <= 0:
        return None
    return t * ray_dir


def build_plane_basis(n: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e1 = x_axis - float(np.dot(x_axis, n)) * n
    e1 = normalize(e1)
    if float(np.linalg.norm(e1)) < 1e-6:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        e1 = normalize(z_axis - float(np.dot(z_axis, n)) * n)
    e2 = normalize(np.cross(n, e1))
    return e1, e2


# ---------------- Kinect mapping helpers ----------------
def map_camera_point_to_color_xy(kinect, p_cam: np.ndarray) -> Optional[Tuple[float, float]]:
    mapper = getattr(kinect, "_mapper", None)
    if mapper is None or not hasattr(mapper, "MapCameraPointToColorSpace"):
        return None
    try:
        cam_pt = PyKinect2024._CameraSpacePoint(float(p_cam[0]), float(p_cam[1]), float(p_cam[2]))
        cp = mapper.MapCameraPointToColorSpace(cam_pt)
        u = float(cp.x)
        v = float(cp.y)
        if not (np.isfinite(u) and np.isfinite(v)):
            return None
        return u, v
    except Exception:
        return None


def depth_pixel_to_camera_ray_dir(kinect, u: int, v: int) -> Optional[np.ndarray]:
    mapper = getattr(kinect, "_mapper", None)
    if mapper is None or not hasattr(mapper, "MapDepthPointToCameraSpace"):
        return None
    try:
        dp = PyKinect2024._DepthSpacePoint(float(u), float(v))
        cam_pt = mapper.MapDepthPointToCameraSpace(dp, 1000)  # 1m arbitrary
        ray = np.array([float(cam_pt.x), float(cam_pt.y), float(cam_pt.z)], dtype=np.float64)
        return normalize(ray)
    except Exception:
        return None


def map_depth_pixel_to_color_xy(
    kinect,
    depth_frame_1d: np.ndarray,
    depth_w: int,
    u: int,
    v: int
) -> Optional[Tuple[float, float]]:
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
    seed_depth_uv: Optional[Tuple[int, int]] = None,
    search_radius: int = SEARCH_RADIUS_DEFAULT,
) -> Optional[Tuple[int, int]]:
    """
    Find the depth pixel whose projection into the color image is closest to the target color pixel.
    Used for:
      - color-center ray
      - manual click ray
    """
    tx, ty = target_color_xy

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

    # Coarse -> medium -> fine
    b1 = search(seed_u, seed_v, radius=search_radius, step=10)
    if b1 is None:
        return None
    b2 = search(b1[0], b1[1], radius=25, step=4) or b1
    b3 = search(b2[0], b2[1], radius=7, step=1) or b2
    return int(b3[0]), int(b3[1])


# ---------------- Plane fitting (RANSAC) ----------------
def sample_depth_pixels(depth_frame_1d: np.ndarray, depth_w: int, depth_h: int, max_samples: int, rng: np.random.Generator):
    x0 = int(depth_w * ROI_X_FRAC[0])
    x1 = int(depth_w * ROI_X_FRAC[1])
    y0 = int(depth_h * ROI_Y_FRAC[0])
    y1 = int(depth_h * ROI_Y_FRAC[1])

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
    best_mask = None
    best_plane = None
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


def estimate_floor_plane_from_depth(kinect, depth_frame_1d: np.ndarray, depth_w: int, depth_h: int, rng: np.random.Generator) -> Optional[Plane]:
    pixels_uv, depths_mm = sample_depth_pixels(depth_frame_1d, depth_w, depth_h, MAX_SAMPLES, rng)
    if len(depths_mm) < 500:
        return None
    pts = pixels_to_camera_points(kinect, pixels_uv, depths_mm)
    if pts.shape[0] < 500:
        return None
    return ransac_plane(pts, iters=RANSAC_ITERS, thresh_m=INLIER_THRESH_M, rng=rng)


# ---------------- App ----------------
class GridApp:
    def __init__(self) -> None:
        sources = PyKinect2024.FrameSourceTypes_Color | PyKinect2024.FrameSourceTypes_Depth
        self.kinect = PyKinectRuntime.PyKinectRuntime(sources)

        self.color_w = self.kinect.color_frame_desc.Width
        self.color_h = self.kinect.color_frame_desc.Height
        self.depth_w = self.kinect.depth_frame_desc.Width
        self.depth_h = self.kinect.depth_frame_desc.Height

        # Create an AUTOSIZE window so OpenCV doesn't scale the displayed image internally.
        # We will show a resized image ourselves.
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

        self.display_w = int(self.color_w * DISPLAY_SCALE)
        self.display_h = int(self.color_h * DISPLAY_SCALE)

        self.rng = np.random.default_rng(0)
        self.smoother = PlaneSmoother(alpha=PLANE_SMOOTH_ALPHA)

        self.plane: Optional[Plane] = None
        self.plane_locked: bool = False
        self.successful_fits: int = 0
        self.frame_idx: int = 0

        self.last_depth_1d: Optional[np.ndarray] = None

        # Default: color-center target
        self.center_target_color_xy = (self.color_w * 0.5, self.color_h * 0.5)
        self.center_seed_depth_uv: Optional[Tuple[int, int]] = None
        self.center_depth_uv: Optional[Tuple[int, int]] = None

        # Manual: user click in color image coordinates (FULL RES)
        self.manual_target_color_xy: Optional[Tuple[float, float]] = None
        self.manual_seed_depth_uv: Optional[Tuple[int, int]] = None
        self.manual_depth_uv: Optional[Tuple[int, int]] = None

        # For HUD
        self.status_msg: str = ""

    def reset_calibration(self) -> None:
        self.plane = None
        self.plane_locked = False
        self.successful_fits = 0
        self.smoother.reset()
        self.status_msg = "Recalibrating floor... (please keep floor visible)"

    # ---- Mouse handling (FIXED) ----
    def _display_to_color_xy(self, x_disp: int, y_disp: int) -> Tuple[float, float]:
        """
        Convert mouse coords from the DISPLAY image to FULL-RES color image coords.
        Because we control the display size ourselves, this mapping is stable.
        """
        x = float(x_disp) / DISPLAY_SCALE
        y = float(y_disp) / DISPLAY_SCALE
        # Clamp to image bounds
        x = float(np.clip(x, 0, self.color_w - 1))
        y = float(np.clip(y, 0, self.color_h - 1))
        return x, y

    def _on_mouse(self, event: int, x: int, y: int, flags: int, userdata) -> None:
        # x,y here are in DISPLAY image pixels (because the window is AUTOSIZE and we show a resized image)
        if event == cv2.EVENT_LBUTTONDOWN:
            cx, cy = self._display_to_color_xy(x, y)

            self.manual_target_color_xy = (cx, cy)
            self.manual_seed_depth_uv = self.manual_depth_uv or self.center_depth_uv or (self.depth_w // 2, self.depth_h // 2)
            self.status_msg = f"Clicked grid center at color pixel ({cx:.1f}, {cy:.1f})"

        if event == cv2.EVENT_RBUTTONDOWN:
            self.manual_target_color_xy = None
            self.manual_seed_depth_uv = None
            self.manual_depth_uv = None
            self.status_msg = "Manual grid center cleared (back to center-of-view)."

    # ---- Updates ----
    def _update_floor_plane(self) -> None:
        if self.last_depth_1d is None or self.plane_locked:
            return
        if self.frame_idx % FIT_EVERY_N_FRAMES != 0:
            return

        new_plane = estimate_floor_plane_from_depth(self.kinect, self.last_depth_1d, self.depth_w, self.depth_h, self.rng)
        if new_plane is None:
            return

        self.plane = self.smoother.update(new_plane)
        self.successful_fits += 1
        if self.successful_fits >= CALIBRATION_FITS_TO_LOCK:
            self.plane_locked = True
            self.status_msg = "Floor locked."

    def _update_depth_pixel_mapping(self) -> None:
        if self.last_depth_1d is None:
            return

        # Center-of-view mapping (color center -> depth pixel)
        if (self.center_depth_uv is None) or (self.frame_idx % CENTER_ORIGIN_UPDATE_EVERY == 0):
            self.center_depth_uv = find_depth_pixel_for_color_xy(
                self.kinect,
                self.last_depth_1d,
                self.depth_w,
                self.depth_h,
                target_color_xy=self.center_target_color_xy,
                seed_depth_uv=self.center_seed_depth_uv,
                search_radius=SEARCH_RADIUS_DEFAULT,
            )
            if self.center_depth_uv is not None:
                self.center_seed_depth_uv = self.center_depth_uv

        # Manual mapping (clicked color pixel -> depth pixel)
        if self.manual_target_color_xy is not None:
            if (self.manual_depth_uv is None) or (self.frame_idx % MANUAL_ORIGIN_UPDATE_EVERY == 0):
                self.manual_depth_uv = find_depth_pixel_for_color_xy(
                    self.kinect,
                    self.last_depth_1d,
                    self.depth_w,
                    self.depth_h,
                    target_color_xy=self.manual_target_color_xy,
                    seed_depth_uv=self.manual_seed_depth_uv,
                    search_radius=SEARCH_RADIUS_DEFAULT,
                )
                if self.manual_depth_uv is not None:
                    self.manual_seed_depth_uv = self.manual_depth_uv

    # ---- Compute origin and draw ----
    def _compute_origin_on_floor(self, plane: Plane) -> Optional[np.ndarray]:
        # Use manual click if available, otherwise use center-of-view
        depth_uv = self.manual_depth_uv if self.manual_target_color_xy is not None else self.center_depth_uv
        if depth_uv is None:
            depth_uv = (self.depth_w // 2, self.depth_h // 2)

        ray_dir = depth_pixel_to_camera_ray_dir(self.kinect, depth_uv[0], depth_uv[1])
        if ray_dir is None:
            return None

        origin = intersect_ray_plane(ray_dir, plane)
        if origin is None:
            origin = intersect_ray_plane(ray_dir, plane.flipped())
            if origin is None:
                return None
        return origin

    def _draw_click_marker(self, img_full: np.ndarray) -> None:
        if self.manual_target_color_xy is None:
            return
        x, y = self.manual_target_color_xy
        xi, yi = int(round(x)), int(round(y))
        h, w = img_full.shape[:2]
        if 0 <= xi < w and 0 <= yi < h:
            cv2.drawMarker(img_full, (xi, yi), CLICK_MARK_COLOR, markerType=cv2.MARKER_CROSS, markerSize=30, thickness=2)

    def _draw_axes_and_grid(self, img_full: np.ndarray, plane: Plane, origin_cam: np.ndarray) -> bool:
        h, w = img_full.shape[:2]
        e1, e2 = build_plane_basis(plane.n)

        o_uv = map_camera_point_to_color_xy(self.kinect, origin_cam)
        if o_uv is None:
            return False
        ox, oy = int(o_uv[0]), int(o_uv[1])

        # origin dot
        cv2.circle(img_full, (ox, oy), 7, (255, 255, 255), -1, cv2.LINE_AA)

        # axes
        axis_len = 1.0
        x_end = origin_cam + axis_len * e1
        y_end = origin_cam + axis_len * e2
        x_uv = map_camera_point_to_color_xy(self.kinect, x_end)
        y_uv = map_camera_point_to_color_xy(self.kinect, y_end)

        if x_uv is not None:
            cv2.arrowedLine(img_full, (ox, oy), (int(x_uv[0]), int(x_uv[1])), AXIS_X_COLOR, 2, cv2.LINE_AA, tipLength=0.03)
            cv2.putText(img_full, "+X", (int(x_uv[0]) + 6, int(x_uv[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, AXIS_X_COLOR, 2, cv2.LINE_AA)

        if y_uv is not None:
            cv2.arrowedLine(img_full, (ox, oy), (int(y_uv[0]), int(y_uv[1])), AXIS_Y_COLOR, 2, cv2.LINE_AA, tipLength=0.03)
            cv2.putText(img_full, "+Y", (int(y_uv[0]) + 6, int(y_uv[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, AXIS_Y_COLOR, 2, cv2.LINE_AA)

        # grid points
        for j in range(-GRID_HALF_EXTENT, GRID_HALF_EXTENT + 1):
            for i in range(-GRID_HALF_EXTENT, GRID_HALF_EXTENT + 1):
                p = origin_cam + (i * GRID_SPACING_M) * e1 + (j * GRID_SPACING_M) * e2
                uv = map_camera_point_to_color_xy(self.kinect, p)
                if uv is None:
                    continue
                u, v = uv
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(img_full, (int(u), int(v)), POINT_RADIUS, POINT_COLOR, -1, cv2.LINE_AA)

        return True

    def run(self) -> int:
        try:
            while True:
                if not self.kinect.has_new_color_frame():
                    time.sleep(0.002)
                    continue

                color_1d = self.kinect.get_last_color_frame()
                img_full = color_frame_to_bgr(color_1d, self.color_w, self.color_h)

                if self.kinect.has_new_depth_frame():
                    self.last_depth_1d = self.kinect.get_last_depth_frame()

                # update plane + mapping
                self._update_floor_plane()
                self._update_depth_pixel_mapping()

                # draw manual click marker
                self._draw_click_marker(img_full)

                # draw axes + grid
                drew = False
                if self.plane is not None and self.last_depth_1d is not None:
                    origin_cam = self._compute_origin_on_floor(self.plane)
                    if origin_cam is not None:
                        drew = self._draw_axes_and_grid(img_full, self.plane, origin_cam)

                # HUD on full-res (will be scaled with the image)
                mode = "MANUAL CLICK" if self.manual_target_color_xy is not None else "CENTER-OF-VIEW"
                hud = [
                    f"Depth: {'OK' if self.last_depth_1d is not None else '---'}   Plane: {'LOCKED' if self.plane_locked else 'CALIBRATING'}   Fits: {self.successful_fits}/{CALIBRATION_FITS_TO_LOCK}",
                    f"Mode: {mode}   Draw: {'OK' if drew else '---'}",
                    "LeftClick set grid center, RightClick clear, r recalibrate, q/ESC quit",
                ]
                if self.status_msg:
                    hud.insert(0, self.status_msg)
                draw_hud(img_full, hud)

                # Create display image at fixed scale (this makes mouse coords reliable)
                img_disp = cv2.resize(img_full, (self.display_w, self.display_h), interpolation=cv2.INTER_AREA)
                cv2.imshow(WINDOW_NAME, img_disp)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("r"):
                    self.reset_calibration()

                self.frame_idx += 1
                time.sleep(0.001)

        finally:
            try:
                self.kinect.close()
            except Exception:
                pass
            cv2.destroyAllWindows()

        return 0


def main() -> int:
    return GridApp().run()


if __name__ == "__main__":
    raise SystemExit(main())