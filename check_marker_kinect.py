import time
import cv2
import numpy as np
from pykinect2024 import PyKinectRuntime, PyKinect2024

CANDIDATE_DICTS = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_50",
    "DICT_ARUCO_ORIGINAL",
]
ALLOWED_IDS: set[int] | None = None  # e.g., {0, 1, 2}; None = any
MIN_MARKER_SIZE_PX = 40
DEBOUNCE_SEC = 0.5
DISPLAY_SCALE = 0.75  # for the shown window; keep full-res for detection

def build_detector(dict_name: str) -> cv2.aruco.ArucoDetector:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    # moderately strict; relax if needed
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 35
    params.adaptiveThreshWinSizeStep = 5
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.03
    params.maxMarkerPerimeterRate = 4.5
    params.minCornerDistanceRate = 0.05
    params.minDistanceToBorder = 3
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = True
    return cv2.aruco.ArucoDetector(d, params)

def filter_by_size(corners: np.ndarray) -> bool:
    pts = corners.reshape(-1, 2)
    side_lengths = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    return np.mean(side_lengths) >= MIN_MARKER_SIZE_PX

def detect_first_hit(detectors: dict[str, cv2.aruco.ArucoDetector], gray: np.ndarray):
    for name, det in detectors.items():
        corners, ids, _ = det.detectMarkers(gray)
        if ids is None:
            continue
        kept_corners, kept_ids = [], []
        for c, mid in zip(corners, ids.flatten()):
            mid = int(mid)
            if ALLOWED_IDS is not None and mid not in ALLOWED_IDS:
                continue
            if not filter_by_size(c):
                continue
            kept_corners.append(c)
            kept_ids.append(mid)
        if kept_ids:
            return name, kept_ids, kept_corners
    return None, [], []

def color_frame_to_bgr(frame_1d: np.ndarray, width: int, height: int) -> np.ndarray:
    bgra = frame_1d.reshape((height, width, 4)).astype(np.uint8, copy=False)
    return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

def main() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not found. Install: pip install opencv-contrib-python")

    detectors = {name: build_detector(name) for name in CANDIDATE_DICTS}

    kinect = PyKinectRuntime.PyKinectRuntime(PyKinect2024.FrameSourceTypes_Color)
    color_w = kinect.color_frame_desc.Width
    color_h = kinect.color_frame_desc.Height

    last_hit = None
    last_time = 0.0

    try:
        while True:
            if not kinect.has_new_color_frame():
                time.sleep(0.002)
                continue

            color_1d = kinect.get_last_color_frame()
            if color_1d is None:
                continue

            frame_bgr = color_frame_to_bgr(color_1d, color_w, color_h)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            dict_name, ids, corners = detect_first_hit(detectors, gray)

            now = time.time()
            hit = (dict_name, tuple(ids))
            if hit != last_hit and now - last_time > DEBOUNCE_SEC:
                if ids:
                    print(f"Detected IDs ({dict_name}): {ids}")
                else:
                    print("No marker detected.")
                last_hit = hit
                last_time = now

            if ids and corners:
                cv2.aruco.drawDetectedMarkers(frame_bgr, corners, np.array(ids, dtype=np.int32))

            disp = cv2.resize(frame_bgr, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
            cv2.imshow("Kinect ArUco Scanner (press q to quit)", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        try:
            kinect.close()
        except Exception:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()