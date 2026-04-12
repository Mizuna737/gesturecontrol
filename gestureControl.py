#!/usr/bin/env python3
"""
gestureControl.py — Webcam gesture engine.

Reads trigger bindings from triggers.toml, detects hand gestures via
MediaPipe, and emits named D-Bus signals. No actions are executed here —
that is the responsibility of gestureControl-actions.py or any other
subscriber on the session bus.

D-Bus interface : org.gesturecontrol.Engine  at  /org/gesturecontrol
Signals emitted :
  GestureFired(name: s, hand: s)
  ContinuousUpdate(name: s, hand: s, value: d)
  ContinuousEnd(name: s, hand: s)
  SequenceProgress(name: s, hand: s, step: i, total: i)

Usage:
  python gestureControl.py [--input 0] [--config PATH] [--debug]

  --input   camera device index or path (default: 0)
  --config  path to triggers.toml (default: ~/.config/gestureControl/triggers.toml)
  --debug   show OpenCV window with landmarks and gesture labels
"""

import sys
import os
_VENV = os.path.expanduser("~/.local/share/gesturecontrol/venv/bin/python3")
if sys.executable != _VENV and os.path.exists(_VENV):
    os.execv(_VENV, [_VENV] + sys.argv)

import argparse
import json
import signal
import subprocess
import sys
import time
import collections
import threading
import tomllib
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import mediapipe as mp
import dbus
import dbus.service
import dbus.mainloop.glib

# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_PATH = (
    Path.home() / ".local" / "share" / "gesturecontrol" / "hand_landmarker.task"
)
DEFAULT_CONFIG = Path.home() / ".config" / "gesturecontrol" / "triggers.toml"

DBUS_NAME = "org.gesturecontrol"
DBUS_PATH = "/org/gesturecontrol"
DBUS_IFACE = "org.gesturecontrol.Engine"

# ── Dark-frame detector ────────────────────────────────────────────────────────

class DarkFrameDetector:
    """Adaptive filter for IR cameras that interleave dark calibration frames.

    IR cameras (e.g. BRIO Windows Hello) alternate between an illuminated frame
    and a dark calibration frame. Ambient IR (e.g. sunlight) raises the dark
    frame baseline above any fixed threshold, so we split on the valley between
    the two brightness clusters instead of a hardcoded number.
    """

    _WARMUP_THRESHOLD = 20.0  # fallback until the rolling window fills
    _WINDOW = 30              # frames to keep in the rolling window
    _MIN_SPREAD = 5.0         # minimum lo/hi spread to trust the split
    _SPLIT_FRAC = 0.4         # threshold = lo + spread * frac

    def __init__(self):
        self._recent = collections.deque(maxlen=self._WINDOW)

    def isDark(self, mean):
        self._recent.append(mean)
        if len(self._recent) < 10:
            return mean < self._WARMUP_THRESHOLD
        lo = min(self._recent)
        hi = max(self._recent)
        if hi - lo < self._MIN_SPREAD:
            return False  # no bimodal split visible; don't filter anything
        return mean < lo + (hi - lo) * self._SPLIT_FRAC

# MediaPipe Tasks API aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Landmark index pairs for drawing the hand skeleton
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]

DEBUG = False

# ── Config dataclasses ─────────────────────────────────────────────────────────


@dataclass
class PoseTrigger:
    hand: str  # "right", "left", or "either"
    shape: str  # pose name, e.g. "ONE", "FIST"
    dwellMs: int  # ms the pose must be held before firing


@dataclass
class SwipeTrigger:
    hand: str
    direction: str  # "LEFT_SWIPE" or "RIGHT_SWIPE" (normalised from config)
    minDisplacement: float  # normalised x displacement required


@dataclass
class SequenceTrigger:
    hand: str
    steps: list  # ordered pose names, e.g. ["FIST", "THUMBS_UP"]
    windowMs: int  # max ms between first and last step completion
    stepDwellMs: int  # ms each individual step must be held to register


@dataclass
class ContinuousTrigger:
    hand: str
    metric: str  # "pinch_distance", "hand_height", "hand_x", "finger_spread"
    activeWhile: (
        str | None
    )  # pose name that must be held to activate this trigger; None = no same-hand pose requirement
    valueRange: tuple  # (low, high) raw sensor range to normalise across [0, 1]


@dataclass
class ChordTrigger:
    left: str  # pose name required on left hand simultaneously
    right: str  # pose name required on right hand simultaneously
    dwellMs: int


@dataclass
class PoseDefinition:
    name: str
    thumb: bool | None = None  # None = don't care
    index: bool | None = None
    middle: bool | None = None
    ring: bool | None = None
    pinky: bool | None = None


@dataclass
class Binding:
    name: str
    trigger: object  # one of the trigger types above
    requireLeft: str | None = (
        None  # left hand must hold this pose (cross-hand condition)
    )
    requireRight: str | None = (
        None  # right hand must hold this pose (cross-hand condition)
    )


# ── Config loading ─────────────────────────────────────────────────────────────


def parseRange(raw):
    """Convert a [min, max] list from config into a (float, float) tuple."""
    return (float(raw[0]), float(raw[1]))


def parseTrigger(d, defaultDwellMs):
    """Build the appropriate trigger dataclass from a raw config dict."""
    kind = d["type"]
    hand = d.get("hand", "right")

    if kind == "pose":
        return PoseTrigger(
            hand=hand,
            shape=d["shape"],
            dwellMs=d.get("dwell_ms", defaultDwellMs),
        )
    if kind == "swipe":
        # Config uses "left"/"right"; normalise to internal "LEFT_SWIPE"/"RIGHT_SWIPE"
        direction = d["direction"].upper() + "_SWIPE"
        return SwipeTrigger(
            hand=hand,
            direction=direction,
            minDisplacement=d.get("min_displacement", 0.15),
        )
    if kind == "sequence":
        return SequenceTrigger(
            hand=hand,
            steps=d["steps"],
            windowMs=d["window_ms"],
            stepDwellMs=d.get("step_dwell_ms", 100),
        )
    if kind == "continuous":
        return ContinuousTrigger(
            hand=hand,
            metric=d["metric"],
            activeWhile=d.get("active_while"),  # None = no same-hand pose requirement
            valueRange=parseRange(d["range"]) if "range" in d else (0.0, 1.0),
        )
    if kind == "chord":
        return ChordTrigger(
            left=d["left"],
            right=d["right"],
            dwellMs=d.get("dwell_ms", defaultDwellMs),
        )
    raise ValueError(f"Unknown trigger type: {kind!r}")


def parsePose(d):
    """Build a PoseDefinition from a raw config dict. Absent fingers default to None (don't care)."""
    return PoseDefinition(
        name=d["name"],
        thumb=d.get("thumb"),
        index=d.get("index"),
        middle=d.get("middle"),
        ring=d.get("ring"),
        pinky=d.get("pinky"),
    )


def loadConfig(path):
    """Load triggers.toml. Returns (settings dict, list of PoseDefinition, list of Binding)."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    settings = raw.get("settings", {})
    defaultDwellMs = settings.get("dwell_ms", 200)
    poses = [parsePose(p) for p in raw.get("poses", [])]
    bindings = [
        Binding(
            name=item["name"],
            trigger=parseTrigger(item["trigger"], defaultDwellMs),
            requireLeft=item.get("require_left"),
            requireRight=item.get("require_right"),
        )
        for item in raw.get("bindings", [])
    ]
    return settings, poses, bindings


# ── Pose detection ─────────────────────────────────────────────────────────────


def fingerStates(landmarks, handLabel):
    """Return [thumb, index, middle, ring, pinky] booleans (True = extended).

    Finger extended: tip.y < pip.y (tip is higher; y increases downward).
    Thumb: horizontal comparison, direction depends on handedness.

    After cv2.flip(frame, 1), MediaPipe 'Left' == user's right hand,
    so the thumb direction check is inverted relative to the raw label.
    """
    lm = landmarks
    index = lm[8].y < lm[6].y
    middle = lm[12].y < lm[10].y
    ring = lm[16].y < lm[14].y
    pinky = lm[20].y < lm[18].y
    isRight = handLabel == "Right"
    thumb = lm[4].x > lm[3].x if isRight else lm[4].x < lm[3].x
    return [thumb, index, middle, ring, pinky]


def classifyPose(landmarks, handLabel, poses):
    """Match current finger states against the user-defined pose list.

    Each pose specifies True/False/None per finger; None means don't care.
    The first pose whose constraints all match is returned.
    Define more specific poses (more constraints) before general ones in config.
    """
    states = fingerStates(landmarks, handLabel)
    for pose in poses:
        constraints = [pose.thumb, pose.index, pose.middle, pose.ring, pose.pinky]
        if all(c is None or c == s for c, s in zip(constraints, states)):
            return pose.name
    return None


# ── Continuous metrics ─────────────────────────────────────────────────────────


def measureMetric(landmarks, metric):
    """Compute a single raw float for a named metric from hand landmarks."""
    if metric == "pinch_distance":
        dx = landmarks[4].x - landmarks[8].x
        dy = landmarks[4].y - landmarks[8].y
        return (dx * dx + dy * dy) ** 0.5
    if metric == "hand_height":
        # Invert y so that raising the hand produces a higher value
        return 1.0 - landmarks[0].y
    if metric == "hand_x":
        return landmarks[0].x
    if metric == "finger_spread":
        xs = [landmarks[i].x for i in [4, 8, 12, 16, 20]]
        return max(xs) - min(xs)
    if metric == "angle":
        # Vector from wrist (0) to middle-finger MCP (9) gives stable palm direction.
        # Cosine of that vector w.r.t. horizontal: -1 = pointing left, +1 = pointing right.
        # Shifted to [0.0, 1.0] so 0 = left, 1 = right.
        dx = landmarks[9].x - landmarks[0].x
        dy = landmarks[9].y - landmarks[0].y
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            return 0.5
        return (dx / length + 1.0) / 2.0
    return 0.0


def measureAllMetrics(landmarks):
    """Return a dict of all supported metric values for a hand."""
    return {
        m: measureMetric(landmarks, m)
        for m in ("pinch_distance", "hand_height", "hand_x", "finger_spread", "angle")
    }


def normalizeMetric(value, valueRange):
    """Clamp and map a raw metric value through valueRange to [0.0, 1.0]."""
    low, high = valueRange
    if high == low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


# ── Primitive detectors ────────────────────────────────────────────────────────


class DwellDebouncer:
    """Fire a gesture after it has been held continuously for dwellMs.

    Resets any time the input changes. Re-arming after fire requires the
    gesture to change and come back (prevents continuous re-fire on hold).
    """

    def __init__(self, dwellMs):
        self.dwellMs = dwellMs
        self.current = None
        self.since = 0.0
        self.lastFired = None

    def update(self, gesture):
        """Return the gesture name when dwell threshold is met, else None."""
        now = time.monotonic()
        if gesture != self.current:
            self.current = gesture
            self.since = now
            self.lastFired = None
            return None
        if gesture is not None and gesture != self.lastFired:
            if (now - self.since) * 1000 >= self.dwellMs:
                self.lastFired = gesture
                return gesture
        return None

    def reset(self):
        self.current = None
        self.since = 0.0
        self.lastFired = None


class SwipeDetector:
    """Detect left/right swipes from index tip x-position history.

    Fires when total horizontal displacement within the sliding windowMs
    exceeds minDisplacement. A cooldown prevents double-firing.
    Frame is already flipped: x increasing = hand moving right on screen.
    """

    def __init__(self, minDisplacement=0.15, windowMs=300, cooldownMs=800):
        self.minDisplacement = minDisplacement
        self.windowMs = windowMs
        self.cooldownMs = cooldownMs
        self.history = []  # [(timestamp_ms, x), ...]
        self.lastFiredAt = 0

    def update(self, indexTipX, timestampMs):
        """Return 'LEFT_SWIPE', 'RIGHT_SWIPE', or None."""
        self.history.append((timestampMs, indexTipX))
        cutoff = timestampMs - self.windowMs
        self.history = [(t, x) for t, x in self.history if t >= cutoff]

        if timestampMs - self.lastFiredAt < self.cooldownMs:
            return None
        if len(self.history) < 2:
            return None

        displacement = self.history[-1][1] - self.history[0][1]
        if abs(displacement) >= self.minDisplacement:
            self.lastFiredAt = timestampMs
            self.history.clear()
            return "RIGHT_SWIPE" if displacement > 0 else "LEFT_SWIPE"
        return None

    def reset(self):
        self.history.clear()


class MotionFilter:
    """Suppress static pose classification when the hand is moving rapidly.

    Tracks wrist (landmark 0) over windowMs. If displacement exceeds
    maxDisplacement, isMoving() returns True and poses should be ignored.
    This prevents a swipe from also registering as ONE/TWO/etc.
    """

    def __init__(self, maxDisplacement=0.04, windowMs=150):
        self.maxDisplacement = maxDisplacement
        self.windowMs = windowMs
        self.history = []  # [(timestamp_ms, x, y), ...]

    def update(self, x, y, timestampMs):
        self.history.append((timestampMs, x, y))
        cutoff = timestampMs - self.windowMs
        self.history = [(t, px, py) for t, px, py in self.history if t >= cutoff]

    def isMoving(self):
        if len(self.history) < 2:
            return False
        dx = self.history[-1][1] - self.history[0][1]
        dy = self.history[-1][2] - self.history[0][2]
        return (dx * dx + dy * dy) ** 0.5 > self.maxDisplacement

    def reset(self):
        self.history.clear()


# ── Per-hand processor ─────────────────────────────────────────────────────────


@dataclass
class HandFrameResult:
    """All processed output for one hand in one frame."""

    pose: str | None  # pose name if static and recognised; None if moving/absent
    swipe: str | None  # swipe event this frame, or None
    metrics: dict  # all metric values, always computed
    rawPose: str | None  # pose before motion suppression (debug only)
    isMoving: bool
    fingers: list | None = None  # [thumb, index, middle, ring, pinky] booleans


class HandProcessor:
    """Bundles SwipeDetector, MotionFilter, and pose classification
    for a single hand. Stateful across frames."""

    def __init__(self, poses):
        self.poses = poses
        self.swipeDetector = SwipeDetector()
        self.motionFilter = MotionFilter()

    def update(self, landmarks, mpLabel, timestampMs):
        """Process one frame for this hand and return a HandFrameResult."""
        self.motionFilter.update(landmarks[0].x, landmarks[0].y, timestampMs)
        swipe = self.swipeDetector.update(landmarks[8].x, timestampMs)
        rawPose = classifyPose(landmarks, mpLabel, self.poses)
        # Suppress static poses while the hand is in motion
        pose = None if self.motionFilter.isMoving() else rawPose
        metrics = measureAllMetrics(landmarks)
        return HandFrameResult(
            pose=pose,
            swipe=swipe,
            metrics=metrics,
            rawPose=rawPose,
            isMoving=self.motionFilter.isMoving(),
            fingers=fingerStates(landmarks, mpLabel),
        )

    def reset(self):
        """Call when this hand leaves the frame to clear stale history."""
        self.swipeDetector.reset()
        self.motionFilter.reset()


# ── Sequence tracker ───────────────────────────────────────────────────────────


class SequenceTracker:
    """Tracks progress through a single SequenceTrigger's ordered steps.

    Each step must be held for stepDwellMs to register. The overall
    sequence must complete within windowMs of the first step firing.
    Returns (stepsCompleted, done) on each update call.
    """

    def __init__(self, steps, windowMs, stepDwellMs):
        self.steps = steps
        self.windowMs = windowMs
        self.stepDwellMs = stepDwellMs
        self.stepDebouncer = DwellDebouncer(stepDwellMs)
        self.currentStep = 0
        self.firstStepTimeMs = None  # set when step 0 fires; drives timeout

    def update(self, pose, timestampMs):
        """Return (stepsCompleted: int, done: bool).

        stepsCompleted is 0 if no step fired this frame, otherwise the
        running count of completed steps (1..N). done is True on the
        frame when the final step completes.
        """
        # Expire the entire sequence if the window has elapsed since step 0
        if self.firstStepTimeMs is not None:
            if timestampMs - self.firstStepTimeMs > self.windowMs:
                self.reset()

        # Only feed the expected pose into the debouncer; anything else resets it
        expected = self.steps[self.currentStep]
        debouncerInput = pose if pose == expected else None
        if not self.stepDebouncer.update(debouncerInput):
            return 0, False

        # This step confirmed — record timing and advance
        if self.currentStep == 0:
            self.firstStepTimeMs = timestampMs
        self.currentStep += 1
        stepsCompleted = self.currentStep

        if self.currentStep >= len(self.steps):
            self.reset()
            return stepsCompleted, True

        # Prepare debouncer for the next step
        self.stepDebouncer.reset()
        return stepsCompleted, False

    def reset(self):
        self.currentStep = 0
        self.firstStepTimeMs = None
        self.stepDebouncer.reset()


# ── Continuous tracker ─────────────────────────────────────────────────────────


class ContinuousTracker:
    """Tracks a ContinuousTrigger: active while activeWhile pose is held.

    Returns (normalizedValue, justEnded) each frame.
    normalizedValue is None when the trigger is inactive.
    justEnded is True on the single frame the trigger deactivates.
    """

    def __init__(self, trigger):
        self.trigger = trigger
        self.active = False

    def update(self, pose, metrics, timestampMs, enabled=True):
        shouldBeActive = enabled and (
            self.trigger.activeWhile is None or pose == self.trigger.activeWhile
        )

        # Deactivation: emit one ContinuousEnd then go idle
        if self.active and not shouldBeActive:
            self.active = False
            return None, True

        if shouldBeActive:
            self.active = True
            raw = metrics.get(self.trigger.metric, 0.0)
            value = normalizeMetric(raw, self.trigger.valueRange)
            return value, False

        return None, False


# ── Trigger matching ───────────────────────────────────────────────────────────


@dataclass
class BindingState:
    """Mutable per-binding state initialised at startup."""

    binding: Binding
    debouncer: object = None  # DwellDebouncer — pose and chord bindings
    sequenceTracker: object = None  # SequenceTracker — sequence bindings
    continuousTracker: object = None  # ContinuousTracker — continuous bindings


def buildBindingState(binding, defaultDwellMs):
    """Create a BindingState with the correct state object for the trigger type."""
    t = binding.trigger
    state = BindingState(binding=binding)
    if isinstance(t, PoseTrigger):
        state.debouncer = DwellDebouncer(t.dwellMs)
    elif isinstance(t, SequenceTrigger):
        state.sequenceTracker = SequenceTracker(t.steps, t.windowMs, t.stepDwellMs)
    elif isinstance(t, ContinuousTrigger):
        state.continuousTracker = ContinuousTracker(t)
    elif isinstance(t, ChordTrigger):
        state.debouncer = DwellDebouncer(t.dwellMs)
    return state


def getPoseForHand(handData, hand):
    """Resolve the pose for a trigger's hand field, including 'either'."""
    if hand == "either":
        return (handData.get("right") or handData.get("left") or _emptyResult()).pose
    result = handData.get(hand)
    return result.pose if result else None


def getSwipeForHand(handData, hand):
    """Resolve the swipe event for a trigger's hand field, including 'either'."""
    if hand == "either":
        return (handData.get("right") or handData.get("left") or _emptyResult()).swipe
    result = handData.get(hand)
    return result.swipe if result else None


def _emptyResult():
    """Sentinel HandFrameResult used when a hand is absent."""
    return HandFrameResult(
        pose=None, swipe=None, metrics={}, rawPose=None, isMoving=False
    )


class TriggerMatcher:
    """Matches per-frame hand data against all configured bindings
    and calls the publisher when triggers fire."""

    def __init__(self, bindings, publisher, defaultDwellMs):
        self.publisher = publisher
        self.states = [buildBindingState(b, defaultDwellMs) for b in bindings]

    def update(self, handData, timestampMs):
        """Process one frame. handData = {side: HandFrameResult}."""
        for state in self.states:
            self._processBinding(state, handData, timestampMs)

    def _checkConditions(self, binding, handData):
        """Return True if all require_left / require_right constraints are met."""
        if binding.requireLeft is not None:
            leftPose = (handData.get("left") or _emptyResult()).pose
            if leftPose != binding.requireLeft:
                return False
        if binding.requireRight is not None:
            rightPose = (handData.get("right") or _emptyResult()).pose
            if rightPose != binding.requireRight:
                return False
        return True

    def _isSingleHand(self, binding):
        """True if this binding should be suppressed when both hands are visible.

        ChordTrigger is inherently two-handed. Bindings with requireLeft /
        requireRight are deliberately conditioned on both hands. SequenceTrigger
        requires deliberate ordered steps and must not be suppressed — doing so
        would prevent any sequence from firing when both hands are in frame.
        Everything else is a single-hand trigger that should be suppressed while
        both hands are visible, to prevent accidental firing during chord gestures.
        """
        return (
            not isinstance(binding.trigger, (ChordTrigger, SequenceTrigger))
            and binding.requireLeft is None
            and binding.requireRight is None
        )

    def _processBinding(self, state, handData, timestampMs):
        b = state.binding
        t = b.trigger
        conditionsMet = self._checkConditions(b, handData)
        suppress = len(handData) == 2 and self._isSingleHand(b)

        # Continuous triggers must always update the tracker (even when inactive)
        # so that ContinuousEnd fires correctly on deactivation.
        if isinstance(t, ContinuousTrigger):
            result = handData.get(t.hand) or _emptyResult()
            enabled = conditionsMet and not suppress
            value, ended = state.continuousTracker.update(
                result.pose, result.metrics, timestampMs, enabled=enabled
            )
            if value is not None:
                self.publisher.continuousUpdate(b.name, t.hand, value)
            if ended:
                self.publisher.continuousEnd(b.name, t.hand)
            return

        if not conditionsMet:
            return

        if isinstance(t, PoseTrigger):
            # Pass None when suppressed so the debouncer timer resets; otherwise
            # a dwell built up during a chord could fire the moment one hand leaves.
            pose = None if suppress else getPoseForHand(handData, t.hand)
            debouncerInput = pose if pose == t.shape else None
            if state.debouncer.update(debouncerInput):
                self.publisher.gestureFired(b.name, t.hand)

        elif isinstance(t, SwipeTrigger):
            if not suppress and getSwipeForHand(handData, t.hand) == t.direction:
                self.publisher.gestureFired(b.name, t.hand)

        elif isinstance(t, SequenceTrigger):
            # Pass None when suppressed so in-progress sequences don't advance.
            pose = None if suppress else getPoseForHand(handData, t.hand)
            stepsCompleted, done = state.sequenceTracker.update(pose, timestampMs)
            if stepsCompleted:
                self.publisher.sequenceProgress(
                    b.name, t.hand, stepsCompleted, len(t.steps)
                )
            if done:
                self.publisher.gestureFired(b.name, t.hand)

        elif isinstance(t, ChordTrigger):
            leftResult = handData.get("left") or _emptyResult()
            rightResult = handData.get("right") or _emptyResult()
            chordHeld = leftResult.pose == t.left and rightResult.pose == t.right
            if state.debouncer.update("chord" if chordHeld else None):
                self.publisher.gestureFired(b.name, "both")


# ── D-Bus publisher ────────────────────────────────────────────────────────────


class GestureEngineService(dbus.service.Object):
    """D-Bus object that broadcasts gesture signals on the session bus.

    Method bodies are empty — dbus-python intercepts the decorated call
    and broadcasts the signal automatically.
    """

    def __init__(self, bus):
        super().__init__(bus, DBUS_PATH)

    @dbus.service.signal(DBUS_IFACE, signature="ss")
    def GestureFired(self, name, hand):
        pass

    @dbus.service.signal(DBUS_IFACE, signature="ssd")
    def ContinuousUpdate(self, name, hand, value):
        pass

    @dbus.service.signal(DBUS_IFACE, signature="ss")
    def ContinuousEnd(self, name, hand):
        pass

    @dbus.service.signal(DBUS_IFACE, signature="ssii")
    def SequenceProgress(self, name, hand, step, total):
        pass


class GesturePublisher:
    """Owns the D-Bus connection and service object.

    No mainloop is needed: @dbus.service.signal emits synchronously when
    called, so there's nothing to dispatch. A running mainloop is only
    required when receiving incoming method calls, which this service never does.
    """

    def __init__(self):
        # DBusGMainLoop must be set before the first SessionBus() call
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()
        bus.request_name(DBUS_NAME)
        self._service = GestureEngineService(bus)

    def gestureFired(self, name, hand):
        print(f"[signal] GestureFired       {name}  ({hand})")
        self._service.GestureFired(name, hand)

    def continuousUpdate(self, name, hand, value):
        self._service.ContinuousUpdate(name, hand, float(value))

    def continuousEnd(self, name, hand):
        print(f"[signal] ContinuousEnd      {name}  ({hand})")
        self._service.ContinuousEnd(name, hand)

    def sequenceProgress(self, name, hand, step, total):
        print(f"[signal] SequenceProgress   {name}  {step}/{total}  ({hand})")
        self._service.SequenceProgress(name, hand, step, total)

    def stop(self):
        pass


# ── Debug overlay ──────────────────────────────────────────────────────────────


def drawLandmarks(frame, landmarks):
    """Draw the hand skeleton onto the frame."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5, (255, 255, 255), -1)


def renderDebugOverlay(frame, handData):
    """Overlay pose labels and motion indicators for each visible hand."""
    parts = []
    for side in ("right", "left"):
        result = handData.get(side)
        if result is None:
            continue
        label = f"{'~' if result.isMoving else ''}{result.rawPose or '---'}"
        if result.swipe or result.pose:
            label += f" -> {result.swipe or result.pose}"
        parts.append(f"{side[0].upper()}: {label}")
    if parts:
        cv2.putText(
            frame,
            "  ".join(parts),
            (10, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 80),
            2,
        )


# ── Main loop ──────────────────────────────────────────────────────────────────


def notifyError(title, body):
    """Send a desktop notification for a config error (non-blocking)."""
    subprocess.Popen(["notify-send", "-u", "critical", "-t", "0", title, body])


# ── Config hot-reload ──────────────────────────────────────────────────────────


class ConfigWatcher:
    """Polls a file's mtime in a background thread and sets a flag on change."""

    def __init__(self, path):
        self._path = Path(path)
        self._mtime = self._currentMtime()
        self._changed = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _currentMtime(self):
        try:
            return self._path.stat().st_mtime
        except OSError:
            return 0.0

    def _poll(self):
        while True:
            time.sleep(1.0)
            mtime = self._currentMtime()
            if mtime != self._mtime:
                self._mtime = mtime
                self._changed.set()

    def pollChanged(self):
        """Return True (once) if the file changed since the last call."""
        if self._changed.is_set():
            self._changed.clear()
            return True
        return False


def calibrateMetric(metric, camInput, hand="either", countdown=3, sampleSecs=2):
    """Sample a raw metric value from the live camera and copy the result to clipboard.

    Counts down `countdown` seconds so the user can hold their pose, then
    samples for `sampleSecs` seconds and reports min/avg/max. The average is
    copied to the clipboard so it can be pasted directly into triggers.toml.
    """
    VALID_METRICS = ("pinch_distance", "hand_height", "hand_x", "finger_spread", "angle")
    if metric not in VALID_METRICS:
        print(f"ERROR: unknown metric '{metric}'. Choose from: {', '.join(VALID_METRICS)}", file=sys.stderr)
        sys.exit(1)

    print(f"Calibrating '{metric}' (hand={hand})")
    print("Hold your pose...")

    for i in range(countdown, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1.0)
    print("Sampling!", flush=True)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    cap = openCamera(camInput)
    detector = DarkFrameDetector()
    samples = []
    deadline = time.monotonic() + sampleSecs

    with HandLandmarker.create_from_options(options) as landmarker:
        while time.monotonic() < deadline:
            ret, frame = cap.read()
            if not ret:
                break
            if detector.isDark(frame.mean()):
                continue
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mpImage, ts)
            for i, handedness in enumerate(result.handedness):
                mpLabel = handedness[0].category_name  # "Left" or "Right"
                side = "right" if mpLabel == "Left" else "left"
                if hand != "either" and side != hand:
                    continue
                samples.append(measureMetric(result.hand_landmarks[i], metric))

    cap.release()

    if not samples:
        print("No hand detected during sampling — try again.", file=sys.stderr)
        sys.exit(1)

    lo   = min(samples)
    hi   = max(samples)
    avg  = sum(samples) / len(samples)
    print(f"  samples : {len(samples)}")
    print(f"  min     : {lo:.4f}")
    print(f"  avg     : {avg:.4f}  ← copied to clipboard")
    print(f"  max     : {hi:.4f}")

    try:
        subprocess.run(["xclip", "-selection", "clipboard"],
                       input=f"{avg:.4f}".encode(), check=True)
    except FileNotFoundError:
        try:
            subprocess.run(["wl-copy"], input=f"{avg:.4f}".encode(), check=True)
        except FileNotFoundError:
            print("(clipboard copy failed — xclip and wl-copy not found)", file=sys.stderr)


def buildHandLandmarker():
    """Create and return a MediaPipe HandLandmarker in VIDEO mode."""
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def openCamera(camInput, width=None, height=None, fps=None, fmt=None):
    """Open the camera, apply format settings, and return the VideoCapture object.

    fmt: explicit fourcc string (e.g. "MJPG", "YUYV", "GREY"). If omitted,
    MJPG is only forced when width/height/fps are also set (RGB camera path).
    IR cameras that don't support MJPG should leave fmt and resolution unset.
    """
    try:
        camIndex = int(camInput)
    except ValueError:
        camIndex = camInput
    cap = cv2.VideoCapture(camIndex, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camInput!r}", file=sys.stderr)
        sys.exit(1)
    if fmt:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt[:4].upper()))
    elif width or height or fps:
        # Only force MJPG when resolution/fps settings are being applied.
        # IR cameras (fixed native format, no resolution overrides) must not
        # have MJPG forced — the set() silently fails and can destabilise the driver.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[init] camera {actual_w}x{actual_h} @ {actual_fps:.0f} fps")
    return cap


def buildHandMap(result):
    """Map detected hands to user-side keys ("right"/"left").

    After cv2.flip(frame, 1): MediaPipe "Left" label == user's right hand.
    """
    handMap = {}
    for i in range(len(result.hand_landmarks)):
        mpLabel = result.handedness[i][0].category_name
        side = "right" if mpLabel == "Left" else "left"
        handMap[side] = (result.hand_landmarks[i], mpLabel)
    return handMap


def processFrame(frame, landmarker, processors, matcher, timestampMs, darkFrameDetector,
                 streamServer=None):
    """Run detection, trigger matching, and optional debug rendering for one frame."""
    frame = cv2.flip(frame, 1)
    if frame.ndim == 2:  # greyscale input (e.g. IR camera)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    # IR cameras (e.g. BRIO Windows Hello) interleave dark calibration frames.
    # Ambient IR (e.g. sunlight) raises the dark-frame baseline unpredictably,
    # so we use an adaptive split on the bimodal brightness distribution.
    if darkFrameDetector.isDark(frame.mean()):
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mpImage, timestampMs)
    handMap = buildHandMap(result)

    handData = {}
    for side, processor in (
        ("right", processors["right"]),
        ("left", processors["left"]),
    ):
        if side in handMap:
            lm, mpLabel = handMap[side]
            handData[side] = processor.update(lm, mpLabel, timestampMs)
        else:
            processor.reset()

    matcher.update(handData, timestampMs)

    shouldAnnotate = DEBUG or streamServer is not None
    if shouldAnnotate:
        for side in handData:
            lm, _ = handMap[side]
            drawLandmarks(frame, lm)
        renderDebugOverlay(frame, handData)

    if DEBUG:
        cv2.imshow("gestureControl", frame)

    if streamServer is not None:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        hands = {
            side: {"fingers": r.fingers, "pose": r.pose}
            for side, r in handData.items()
        }
        streamServer.publish(buf.tobytes(), hands)


# ── Stream server ──────────────────────────────────────────────────────────────


class StreamServer:
    """Serves a local MJPEG stream and SSE hand-state feed on 127.0.0.1.

    Used by gestureControl-config to display the live feed without opening
    the camera a second time. Runs in a daemon thread; zero overhead when
    nobody is connected.
    """

    def __init__(self, port):
        self._port  = port
        self._lock  = threading.Lock()
        self._frame = None
        self._hands = {}
        server = ThreadingHTTPServer(("127.0.0.1", port), self._makeHandler())
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[stream] serving on http://127.0.0.1:{port}")

    def publish(self, jpegBytes, hands):
        with self._lock:
            self._frame = jpegBytes
            self._hands = hands

    def _makeHandler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # suppress per-request output

            def do_GET(self):
                if self.path == "/stream":
                    self._serveStream()
                elif self.path == "/state":
                    self._serveState()
                elif self.path == "/snapshot":
                    self._serveSnapshot()
                else:
                    self.send_response(404)
                    self.end_headers()

            def _serveSnapshot(self):
                with server._lock:
                    frame = server._frame
                if frame is None:
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(frame)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(frame)

            def _serveStream(self):
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                try:
                    while True:
                        with server._lock:
                            frame = server._frame
                        if frame:
                            self.wfile.write(
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n"
                                + frame + b"\r\n"
                            )
                            self.wfile.flush()
                        time.sleep(1 / 30)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _serveState(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                try:
                    while True:
                        with server._lock:
                            data = {"hands": dict(server._hands)}
                        self.wfile.write(
                            f"data: {json.dumps(data)}\n\n".encode()
                        )
                        self.wfile.flush()
                        time.sleep(0.1)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        return Handler


def main():
    global DEBUG

    parser = argparse.ArgumentParser(description="Gesture engine — emits D-Bus signals")
    parser.add_argument(
        "--input",
        default=None,
        help="Camera index or path (overrides config camera setting)",
    )
    parser.add_argument(
        "--config", default=str(DEFAULT_CONFIG), help="Path to triggers.toml"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Show OpenCV window with landmarks"
    )
    parser.add_argument(
        "--stream-port", type=int, default=7071, metavar="PORT",
        help="Port for the local MJPEG/SSE stream server (default: 7071)",
    )
    parser.add_argument(
        "--no-stream", action="store_true",
        help="Disable the local stream server",
    )
    args = parser.parse_args()
    DEBUG = args.debug

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}", file=sys.stderr)
        print("Run gestureControl-setup.sh to download it.", file=sys.stderr)
        sys.exit(1)

    settings, poses, bindings = loadConfig(args.config)
    defaultDwellMs = settings.get("dwell_ms", 200)
    print(f"Loaded {len(poses)} pose(s), {len(bindings)} binding(s) from {args.config}")
    configWatcher = ConfigWatcher(args.config)

    print("[init] connecting to D-Bus...")
    publisher = GesturePublisher()
    print("[init] D-Bus ready")
    processors = {"right": HandProcessor(poses), "left": HandProcessor(poses)}
    matcher = TriggerMatcher(bindings, publisher, defaultDwellMs)
    darkFrameDetector = DarkFrameDetector()

    camInput = args.input if args.input is not None else settings.get("camera", 0)
    print("[init] opening camera...")
    cap = openCamera(
        camInput,
        width=settings.get("width"),
        height=settings.get("height"),
        fps=settings.get("fps"),
        fmt=settings.get("format"),
    )
    print("[init] camera opened")

    streamServer = None if args.no_stream else StreamServer(args.stream_port)

    # cap.read() is a blocking C call — Python's KeyboardInterrupt won't fire
    # until it returns. Use a stop flag set by the signal handler instead.
    stopFlag = threading.Event()
    signal.signal(signal.SIGINT, lambda s, f: stopFlag.set())

    print("[init] loading MediaPipe model...")
    try:
        with buildHandLandmarker() as landmarker:
            print("[init] model ready — entering loop")
            frameCount = 0
            while not stopFlag.is_set():
                if configWatcher.pollChanged():
                    try:
                        settings, poses, bindings = loadConfig(args.config)
                        defaultDwellMs = settings.get("dwell_ms", 200)
                        processors = {
                            "right": HandProcessor(poses),
                            "left": HandProcessor(poses),
                        }
                        matcher = TriggerMatcher(bindings, publisher, defaultDwellMs)
                        print(
                            f"[config] Reloaded: {len(poses)} pose(s), {len(bindings)} binding(s)"
                        )
                    except Exception as e:
                        print(
                            f"[config] Reload failed — keeping old config: {e}",
                            file=sys.stderr,
                        )
                        notifyError("gestureControl config error", str(e))

                ret, frame = cap.read()
                if not ret:
                    print("ERROR: Lost camera feed.", file=sys.stderr)
                    break
                frameCount += 1
                if frameCount <= 3:
                    print(f"[init] frame {frameCount} shape={frame.shape}")
                timestampMs = int(time.monotonic() * 1000)
                processFrame(frame, landmarker, processors, matcher, timestampMs, darkFrameDetector,
                             streamServer=streamServer)
                if cv2.waitKey(1) & 0xFF == ord("q") and DEBUG:
                    break
    finally:
        cap.release()
        publisher.stop()
        if DEBUG:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
