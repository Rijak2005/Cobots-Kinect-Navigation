# Cobots-Kinect-Navigation

Kinect-based floor-grid tracking and navigation for a cobot/robot setup.

This project uses a Kinect v2 depth camera to:
- estimate the floor plane,
- define a 2D grid on the floor,
- detect an ArUco marker on the robot,
- track the robot in grid coordinates, and
- guide movement to a set of target positions.

## What it does

The main navigation app (`main_nav.py`) does the following:
1. Calibrates the floor plane from depth data.
2. Lets you click the taped grid center in the color image to set the grid origin.
3. Detects the robot with an ArUco marker.
4. Computes the robot pose on the floor plane.
5. Steps through a 3×3 set of grid targets and prints simple movement commands like `turn left`, `turn right`, or `move forward`.

## Main scripts

- `main_nav.py` — full navigation demo with floor calibration, grid overlay, ArUco robot tracking, and target sequencing.
- `main.py` — grid calibration and visualization demo.
- `robot_tracker.py` — ArUco marker tracking and robot pose estimation.
- `grid_core.py` — Kinect depth/color mapping, floor plane fitting, and grid coordinate helpers.
- `navigator.py` — simple target-following logic.
- `check_marker_kinect.py` — marker detection test utility.
- `generate_marker.py` — marker generation utility.

## Requirements

- Python 3.10+ recommended
- Kinect v2 hardware
- `pykinect2024`
- `opencv-contrib-python`
- `numpy`

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running

Start the navigation demo:

```bash
python main_nav.py
```

Or run the grid visualization/calibration demo:

```bash
python main.py
```

## Controls

In the navigation demo:
- `Left click` — set the taped grid center
- `Right click` — clear the grid origin
- `r` — recalibrate the floor plane
- `n` — advance to the next target
- `p` — pause/resume
- `q` or `Esc` — quit

## Notes

The robot tracking is tuned for an ArUco marker setup and is hardcoded in `main_nav.py` for a marker ID of `871` and the `DICT_4X4_1000` dictionary.
