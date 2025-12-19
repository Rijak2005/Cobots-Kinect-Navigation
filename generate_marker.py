import cv2
import numpy as np

MARKER_ID = 0
DICT = cv2.aruco.DICT_4X4_50

MARKER_SIZE_PX = 800
MARGIN_PX = 250  # big quiet zone

def main() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco not found. Install: pip install opencv-contrib-python")

    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)

    # IMPORTANT: borderBits=1 gives the required black border around the marker code
    marker = cv2.aruco.generateImageMarker(aruco_dict, MARKER_ID, MARKER_SIZE_PX, borderBits=1)

    canvas = np.full((MARKER_SIZE_PX + 2*MARGIN_PX, MARKER_SIZE_PX + 2*MARGIN_PX), 255, dtype=np.uint8)
    canvas[MARGIN_PX:MARGIN_PX + MARKER_SIZE_PX, MARGIN_PX:MARGIN_PX + MARKER_SIZE_PX] = marker

    out = f"aruco_DICT4X4_50_ID{MARKER_ID}.png"
    cv2.imwrite(out, canvas)
    print(f"Saved: {out}")
    print("Print rules:")
    print("- Print at 100% (disable 'fit to page' if it distorts).")
    print("- Don’t invert colors, don’t use 'save ink' modes.")
    print("- Keep the quiet white border (don’t crop it).")
    print("- Matte paper, flat, no wrinkles.")

if __name__ == "__main__":
    main()
