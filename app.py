"""FormCheck live demo: count reps and call out form from your webcam.

Usage:
    python3 app.py [--exercise squat|push-up]

Controls: e = switch exercise, r = reset count, q = quit.
"""

import argparse

import cv2

from formcheck.exercises import EXERCISES, RepCounter
from formcheck.pose import PoseTracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exercise", default="squat",
                        choices=sorted(EXERCISES), help="exercise to track")
    args = parser.parse_args()

    names = sorted(EXERCISES)
    counter = RepCounter(EXERCISES[args.exercise])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("could not open webcam")
    tracker = PoseTracker()
    print(f"tracking {counter.exercise.name}. e = switch exercise, "
          "r = reset, q = quit")

    flash = ""          # transient on-screen message
    flash_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror

            landmarks, frame = tracker.process(frame)

            if landmarks is not None:
                try:
                    angle = counter.angle_from_landmarks(landmarks)
                except ValueError:
                    angle = None
                if angle is not None:
                    event = counter.update(angle)
                    if event in ("rep", "shallow"):
                        flash = ("rep!" if event == "rep"
                                 else "rep counted - " + counter.last_cue)
                        flash_frames = 45

                    cv2.putText(
                        frame,
                        f"{counter.exercise.name}: {angle:.0f} deg",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0), 2,
                    )
            else:
                cv2.putText(frame, "step back so your whole body is in frame",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255), 2)

            color = (0, 255, 0) if not counter.last_cue else (0, 165, 255)
            cv2.putText(
                frame,
                f"reps: {counter.reps}  (clean: {counter.clean_reps})",
                (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
            )
            if counter.last_cue:
                cv2.putText(frame, counter.last_cue, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            if flash_frames > 0:
                cv2.putText(frame, flash, (10, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                flash_frames -= 1

            cv2.imshow("FormCheck", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                counter = RepCounter(counter.exercise)
            if key == ord("e"):
                i = (names.index(counter.exercise.name) + 1) % len(names)
                counter = RepCounter(EXERCISES[names[i]])
                print(f"switched to {counter.exercise.name}")
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nsession: {counter.reps} {counter.exercise.name} reps "
          f"({counter.clean_reps} clean)")


if __name__ == "__main__":
    main()
