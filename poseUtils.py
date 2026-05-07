#!/usr/bin/env python3
"""Shared pose detection utilities for gestureControl modules.

Provides finger state classification, spread computation, spread constraint
checking, dark frame detection, and landmark skeleton drawing.
"""

import collections


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5),
    (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
    (10, 11), (11, 12), (9, 13), (13, 14),
    (14, 15), (15, 16), (13, 17), (0, 17),
    (17, 18), (18, 19), (19, 20),
]


def fingerStates(landmarks, handLabel):
    """Return [thumb, index, middle, ring, pinky] booleans (True = extended).

    Finger extended: tip.y < pip.y (tip is higher; y increases downward).
    Thumb: horizontal comparison, direction depends on handedness.

    After cv2.flip(frame, 1), MediaPipe 'Left' == user's right hand,
    so the thumb direction check is inverted relative to the raw label.
    """
    index = landmarks[8].y < landmarks[6].y
    middle = landmarks[12].y < landmarks[10].y
    ring = landmarks[16].y < landmarks[14].y
    pinky = landmarks[20].y < landmarks[18].y
    isRight = handLabel == "Right"
    thumb = landmarks[4].x > landmarks[3].x if isRight else landmarks[4].x < landmarks[3].x
    return [thumb, index, middle, ring, pinky]


def computeFingerSpreads(landmarks):
    """Return normalized tip-to-tip gaps for the four adjacent finger pairs.

    Gaps are divided by the wrist(0) to middle-MCP(9) reference length so the
    values are scale-invariant with camera distance.
    """
    def d(a, b):
        dx = landmarks[a].x - landmarks[b].x
        dy = landmarks[a].y - landmarks[b].y
        return (dx * dx + dy * dy) ** 0.5

    refLen = d(0, 9)
    if refLen < 1e-6:
        return {"thumbIndex": 0.0, "indexMiddle": 0.0, "middleRing": 0.0, "ringPinky": 0.0}
    return {
        "thumbIndex":  d(4,  8)  / refLen,
        "indexMiddle": d(8,  12) / refLen,
        "middleRing":  d(12, 16) / refLen,
        "ringPinky":   d(16, 20) / refLen,
    }


DEFAULT_SPREAD_THRESHOLD = 0.20


def checkSpreadConstraint(value, constraint, threshold):
    """Return True if value satisfies the spread constraint.

    constraint: None (don't care) | "close" | "apart" | float (custom min gap)
    """
    if constraint is None:
        return True
    if isinstance(constraint, float):
        return value >= constraint
    if constraint == "apart":
        return value >= threshold
    if constraint == "close":
        return value < threshold
    return True


def classifyPose(landmarks, handLabel, poses, spreadThreshold=DEFAULT_SPREAD_THRESHOLD):
    """Match current finger states and spread constraints against the pose list.

    Each pose specifies True/False/None per finger (None = don't care) plus
    optional spread constraints ("close"/"apart"/float) for adjacent finger pairs.
    The first pose whose constraints all match is returned.
    Define more specific poses (more constraints) before general ones in config.
    """
    states = fingerStates(landmarks, handLabel)
    spreads = computeFingerSpreads(landmarks)
    for pose in poses:
        fingerConstraints = [pose.thumb, pose.index, pose.middle, pose.ring, pose.pinky]
        if not all(c is None or c == s for c, s in zip(fingerConstraints, states)):
            continue
        spreadConstraints = [
            (spreads["thumbIndex"],  pose.spreadThumbIndex),
            (spreads["indexMiddle"], pose.spreadIndexMiddle),
            (spreads["middleRing"],  pose.spreadMiddleRing),
            (spreads["ringPinky"],   pose.spreadRingPinky),
        ]
        if all(checkSpreadConstraint(v, c, spreadThreshold) for v, c in spreadConstraints):
            return pose.name
    return None


class DarkFrameDetector:
    """Adaptive filter for IR cameras that interleave dark calibration frames.

    IR cameras (e.g. BRIO Windows Hello) alternate between an illuminated frame
    and a dark calibration frame. Ambient IR (e.g. sunlight) raises the dark
    frame baseline above any fixed threshold, so we split on the valley between
    the two brightness clusters instead of a hardcoded number.
    """

    _WARMUP_THRESHOLD = 20.0
    _WINDOW = 30
    _MIN_SPREAD = 5.0
    _SPLIT_FRAC = 0.4

    def __init__(self):
        self._recent = collections.deque(maxlen=self._WINDOW)

    def isDark(self, mean):
        """Return True if the frame appears to be a dark calibration frame."""
        self._recent.append(mean)
        if len(self._recent) < 10:
            return mean < self._WARMUP_THRESHOLD
        lo = min(self._recent)
        hi = max(self._recent)
        if hi - lo < self._MIN_SPREAD:
            return False
        return mean < lo + (hi - lo) * self._SPLIT_FRAC


def drawLandmarks(frame, landmarks, connections=HAND_CONNECTIONS):
    """Draw the hand skeleton onto the frame."""
    import cv2
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)
