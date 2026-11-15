"""Joint-angle math on pose landmarks. Pure functions, no dependencies.

Landmarks are (x, y) pixel tuples. Angles come back in degrees, 0-180.
"""


def angle_at(a, b, c):
    """Angle ABC in degrees: the joint angle at b between points a and c."""
    import math

    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    dot = bax * bcx + bay * bcy
    mag_ba = math.hypot(bax, bay)
    mag_bc = math.hypot(bcx, bcy)
    if mag_ba < 1e-9 or mag_bc < 1e-9:
        raise ValueError("degenerate landmarks (zero-length segment)")
    cosine = max(-1.0, min(1.0, dot / (mag_ba * mag_bc)))
    return math.degrees(math.acos(cosine))


# BlazePose landmark indices for the joints FormCheck tracks.
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def joint_angle(landmarks, a_idx, b_idx, c_idx):
    """Angle at landmark b given three landmark indices."""
    return angle_at(landmarks[a_idx], landmarks[b_idx], landmarks[c_idx])
