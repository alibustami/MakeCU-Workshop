# MakeCU PyBullet Workshop

This project loads the `clpai_12dof_0905.urdf` robot into a PyBullet simulation. The script can run headless or with the PyBullet GUI.

## Requirements
- Python 3.11 or newer
- `pybullet` (installed automatically via `pip install .`)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install .
```

## Usage
Run the script from the project root:
```bash
python -m src.main
```

### GUI mode
Add the `--gui` flag to open the PyBullet visualizer:
```bash
python -m src.main --gui
```

### Simulation controls
- `--steps`: number of simulation steps (default: 2400)
- `--timestep`: duration of each simulation step in seconds (default: `1/240`)

### Forward motion
Use the forward-motion options to command the robot base to walk:
- `--walk-mode`: `time` (default) walks for a duration, `distance` walks until the travelled distance is reached.
- `--walk-amount`: seconds or meters depending on the mode (default: 3).
- `--walk-speed`: forward speed in m/s (default: 1.0).

Example: move forward for three meters in the GUI.
```bash
python -m src.main --gui --walk-mode distance --walk-amount 3
```

### Rotation motion
After the forward walk completes, the simulation can execute an in-place turn that mimics the ROS `cmd_vel` angular.z command:
- `--turn-angle`: yaw rotation in degrees (positive turns counter-clockwise, default: 90). Set to `0` to skip the turn.
- `--turn-speed`: yaw rotation speed in degrees per second (default: 45).

Example: walk for two seconds then rotate 45 degrees right at 60 deg/s.
```bash
python -m src.main --gui --walk-amount 2 --turn-angle -45 --turn-speed 60
```

### Keyboard teleoperation
Control the robot live from the PyBullet GUI using the arrow keys:
```bash
python -m src.main_keyboard --gui
```

Controls:
- Up Arrow: walk forward
- Down Arrow: walk backward
- Left Arrow: rotate counter-clockwise
- Right Arrow: rotate clockwise
- Space: stop the current motion and clear the queue
- Esc or `q`: exit

You can tune the distance and speed per key press with `--forward-distance` and `--forward-speed`, and adjust rotation behavior with `--turn-angle` and `--turn-speed`. Use `--queue-limit` to cap how many motions can be queued.

### Example
```bash
python -m src.main --gui --steps 1200 --timestep 0.01
```

## Troubleshooting
- If you see `ModuleNotFoundError: No module named 'pybullet'`, ensure the virtual environment is active and dependencies are installed (`pip install .`).
