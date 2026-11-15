"""Exercise definitions and the rep-counting state machine. Pure python.

The counter watches one joint angle through a rep: it must dip below
down_threshold (the hard part of the rep) and come back above up_threshold
(the start position). If the exercise has a min_depth, a rep that never gets
deep enough still counts but is flagged shallow with a form cue.
"""

from .angles import (
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_ANKLE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    joint_angle,
)


class Exercise:
    def __init__(self, name, joints, down_threshold, up_threshold,
                 min_depth=None, cue=""):
        self.name = name
        self.joints = joints            # (a, b, c) landmark indices, angle at b
        self.down_threshold = down_threshold
        self.up_threshold = up_threshold
        self.min_depth = min_depth      # angle must reach this low to count clean
        self.cue = cue                  # form cue shown for shallow reps


SQUAT = Exercise(
    name="squat",
    joints=(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE),   # knee angle
    down_threshold=140,
    up_threshold=160,
    min_depth=100,
    cue="go deeper - hips below the knees",
)

PUSHUP = Exercise(
    name="push-up",
    joints=(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST),  # elbow angle
    down_threshold=100,
    up_threshold=155,
    min_depth=90,
    cue="chest closer to the floor",
)

EXERCISES = {e.name: e for e in (SQUAT, PUSHUP)}


class RepCounter:
    """Counts reps of one exercise from a stream of joint angles."""

    def __init__(self, exercise):
        self.exercise = exercise
        self.state = "up"
        self.reps = 0
        self.clean_reps = 0
        self._deepest = None            # lowest angle seen this rep
        self.last_cue = ""

    def update(self, angle):
        """Feed one frame's joint angle. Returns an event string or None:

        'rep'        - clean rep counted
        'shallow'    - rep counted but too shallow (see last_cue)
        'descending' - started the downward phase
        """
        ex = self.exercise
        if self.state == "up":
            if angle < ex.down_threshold:
                self.state = "down"
                self._deepest = angle
                return "descending"
            return None

        # state == down
        self._deepest = min(self._deepest, angle)
        if angle > ex.up_threshold:
            self.state = "up"
            self.reps += 1
            if ex.min_depth is not None and self._deepest > ex.min_depth:
                self.last_cue = ex.cue
                return "shallow"
            self.clean_reps += 1
            self.last_cue = ""
            return "rep"
        return None

    def angle_from_landmarks(self, landmarks):
        a, b, c = self.exercise.joints
        return joint_angle(landmarks, a, b, c)
