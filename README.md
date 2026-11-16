# FormCheck

A webcam gym form tracker: it watches your pose, counts your reps, and tells you when a rep doesn't go deep enough. Built on MediaPipe pose estimation + OpenCV.

## What it does

- Tracks 33 body landmarks per frame (`formcheck/pose.py`, MediaPipe Tasks PoseLandmarker - works on mediapipe 1.x).
- Computes the joint angle that matters for the exercise (knee angle for squats, elbow angle for push-ups) from raw pixel coordinates (`formcheck/angles.py`).
- Runs a small state machine over that angle stream (`formcheck/exercises.py`): a rep only counts when you go below the down threshold and come back above the up threshold. If the exercise has a depth bar (hips below knees, chest near the floor) and you don't reach it, the rep still counts but is flagged with a form cue, and clean reps are tracked separately.

## Quick start

```bash
git clone https://github.com/saanvigunu/FormCheck.git
cd FormCheck
pip3 install -r requirements.txt
python3 app.py                 # squats by default
python3 app.py --exercise push-up
```

The pose model (~5 MB) downloads itself on first run - just be online once.

Controls while running: `e` switch exercise, `r` reset the count, `q` quit.

For squats, stand side-on to the camera (it tracks your left side) and step back far enough that your whole body is in frame.

## How a rep counts

```
standing (knee ~175°) -> descend past 140° -> bottom out -> rise past 160° = 1 rep
```

A squat only counts *clean* if the bottom reached 100° at the knee. Shallower than that and you'll see "go deeper - hips below the knees". Push-ups work the same way on the elbow angle (down past 100°, clean below 90°).

## Layout

- `formcheck/angles.py` - joint-angle math, pure functions
- `formcheck/exercises.py` - exercise definitions + the rep-counting state machine
- `formcheck/pose.py` - MediaPipe pose tracking + skeleton overlay
- `app.py` - the webcam loop
- `tests/` - 11 tests over the angle math and every rep-counter path (clean, shallow, partial, multi-rep)

## Tests

```bash
python3 -m unittest discover -s tests -v
```

No camera needed - the tests drive the state machine with scripted angle sequences.

## Requirements

Python 3.9+, opencv, mediapipe >= 0.10.14, numpy. That's it.
