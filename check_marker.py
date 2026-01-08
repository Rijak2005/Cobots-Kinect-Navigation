import cv2
import numpy as np
import time

# Choose the dictionary you actually printed
DICT_NAME = "DICT_4X4_50"
ALLOWED_IDS: set[int] | None = None  # e.g., {0, 1, 2} to whitelist, or None to allow all
MIN_MARKER_SIZE_PX = 40  # reject detections smaller than this (in pixels)

def build_detector(dict_name: str) -> cv2.aruco.ArucoDetector:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    # Stricter settings to reduce false positives
    params.adaptiveThreshWinSizeMin = 5
    params.adaptiveThreshWinSizeMax = 35
    params.adaptiveThreshWinSizeStep = 5
    params.adaptiveThreshConstant = 7
    params.minMarkerPerimeterRate = 0.05
    params.maxMarkerPerimeterRate = 4.0
    params.minCornerDistanceRate = 0.1
    params.minDistanceToBorder = 3
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = True
    return cv2.aruco.ArucoDetector(d, params)

def filter_by_size(corners: np.ndarray) -> bool:
    # corners shape: (4,1,2) -> squeeze to (4,2)
    pts = corners.reshape(-1, 2)
    side_lengths = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
    return np.mean(side_lengths) >= MIN_MARKER_SIZE_PX

def detect_markers(detector: cv2.aruco.ArucoDetector, gray: np.ndarray) -> list[tuple[int, np.ndarray]]:
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    kept: list[tuple[int, np.ndarray]] = []
    for c, mid in zip(corners, ids.flatten()):
        if ALLOWED_IDS is not None and mid not in ALLOWED_IDS:
            continue
        if not filter_by_size(c):
            continue
        kept.append((int(mid), c))
    return kept

def main() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not found. Install: pip install opencv-contrib-python")

    detector = build_detector(DICT_NAME)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    last_ids = None
    last_time = 0.0
    debounce_sec = 0.5

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            markers = detect_markers(detector, gray)
            ids = [m[0] for m in markers]

            for mid, c in markers:
                pts = c.reshape(-1, 2).astype(int)
                cv2.polylines(frame, [pts], True, (0, 255, 0), 2)
                center = np.mean(pts, axis=0)
                top_mid = (pts[0] + pts[1]) / 2.0
                dir_vec = top_mid - center
                arrow_end = center + 1.5 * dir_vec
                cv2.circle(frame, tuple(center.astype(int)), 4, (0, 0, 255), -1)
                cv2.arrowedLine(
                    frame,
                    tuple(center.astype(int)),
                    tuple(arrow_end.astype(int)),
                    (255, 0, 0),
                    2,
                    tipLength=0.25,
                )
                cv2.putText(frame, f"ID: {mid}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            now = time.time()
            if ids != last_ids and now - last_time > debounce_sec:
                if ids:
                    print(f"Detected IDs ({DICT_NAME}): {ids}")
                else:
                    print("No marker detected.")
                last_ids = ids
                last_time = now

            cv2.imshow("ArUco Scanner (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
