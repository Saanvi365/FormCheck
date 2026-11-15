"""Pose tracking with the MediaPipe Tasks PoseLandmarker API.

Works on mediapipe >= 0.10.14 including 1.x (the legacy mp.solutions API was
removed in 1.0). The pose_landmarker_lite.task model downloads automatically
on first run.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

NUM_LANDMARKS = 33

MODEL_PATH = os.path.join("models", "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# Minimal skeleton for overlay drawing (BlazePose 33-point topology, the
# body parts relevant to gym movements).
POSE_CONNECTIONS = [
    (11, 12),                       # shoulders
    (11, 13), (13, 15),             # left arm
    (12, 14), (14, 16),             # right arm
    (11, 23), (12, 24), (23, 24),   # torso
    (23, 25), (25, 27),             # left leg
    (24, 26), (26, 28),             # right leg
]


def ensure_model(path=MODEL_PATH):
    """Download the pose landmarker model if it is not already on disk."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading pose landmarker model to {path} (one time, ~5 MB)...")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def draw_pose(frame, landmarks, color=(80, 200, 255)):
    points = [(int(x), int(y)) for x, y in landmarks]
    for a, b in POSE_CONNECTIONS:
        cv2.line(frame, points[a], points[b], (70, 70, 70), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, color, -1)
    return frame


class PoseTracker:
    """Detect one person's 33 pose landmarks per frame."""

    def __init__(self, detection_confidence=0.6, tracking_confidence=0.6,
                 model_path=MODEL_PATH):
        ensure_model(model_path)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = PoseLandmarker.create_from_options(options)

    def process(self, frame, draw=True):
        """Run detection on a BGR frame.

        Returns (landmarks, annotated_frame) where landmarks is a list of
        33 (x, y) pixel tuples, or None when no person is in frame.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(time.monotonic() * 1000)
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        landmarks = None
        if result.pose_landmarks:
            pose = result.pose_landmarks[0]
            h, w, _ = frame.shape
            landmarks = [(lm.x * w, lm.y * h) for lm in pose]
            if draw:
                draw_pose(frame, landmarks)
        return landmarks, frame

    def close(self):
        self._landmarker.close()
