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
import math
import signal
import subprocess
import sys
import time
import threading
import tomllib
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import mediapipe as mp
from poseUtils import (
    HAND_CONNECTIONS,
    DarkFrameDetector,
    DEFAULT_SPREAD_THRESHOLD,
    checkSpreadConstraint,
    classifyPose,
    computeFingerSpreads,
    drawLandmarks,
    fingerStates,
)
import numpy as np
import onnxruntime as ort
import dbus
import dbus.service
import dbus.mainloop.glib
from gi.repository import GLib

# ── ONNX session helper ────────────────────────────────────────────────────────

_onnxCudaNotified = False


def _createOnnxSession(modelPath, providers=None):
    """Load an ONNX model, preferring GPU via CUDA. Falls back to CPU gracefully."""
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(str(modelPath), providers=providers)
    except Exception as e:
        if "CPUExecutionProvider" in providers:
            print(
                f"[gesture] ONNX {Path(modelPath).name}: CUDA unavailable, falling back to CPU — {e}",
                file=sys.stderr,
            )
            sess = ort.InferenceSession(str(modelPath), providers=["CPUExecutionProvider"])
        else:
            raise
    selected = sess.get_providers()[0]
    print(f"[gesture] ONNX {Path(modelPath).name} loaded via {selected}", file=sys.stderr)
    return sess


# ── ONNX hand-landmark shim ────────────────────────────────────────────────────

@dataclass
class NormalizedLandmark:
    x: float
    y: float
    z: float


class _HandednessEntry:
    __slots__ = ("categoryName",)
    def __init__(self, name):
        self.categoryName = name


class _DetectionResult:
    __slots__ = ("handLandmarks", "handedness")
    def __init__(self, handLandmarks, handedness):
        self.handLandmarks = handLandmarks
        self.handedness = handedness


class HandLandmarkerONNX:
    _PALM_INPUT_SIZE = np.array([192, 192])   # wh — matches model input_1 shape
    _LANDMARK_INPUT_SIZE = np.array([224, 224])
    _SCORE_THRESHOLD = 0.5
    _NMS_THRESHOLD = 0.3
    _LANDMARK_CONF_THRESHOLD = 0.5

    _ANCHOR_STRIDES = [8, 16]
    _ANCHOR_COUNTS = [2, 6]

    def __init__(self, palmModelPath, landmarkModelPath):
        self._palmSess = _createOnnxSession(palmModelPath)
        self._landmarkSess = _createOnnxSession(landmarkModelPath)
        self._palmInputName = self._palmSess.get_inputs()[0].name
        self._landmarkInputName = self._landmarkSess.get_inputs()[0].name
        # output[0] = box+landmark deltas [1,2016,18], output[1] = scores [1,2016,1]
        self._palmOutBox = self._palmSess.get_outputs()[0].name
        self._palmOutScore = self._palmSess.get_outputs()[1].name
        # output[0]=landmarks[1,63], output[1]=conf[1,1], output[2]=handedness[1,1]
        self._landOutLm = self._landmarkSess.get_outputs()[0].name
        self._landOutConf = self._landmarkSess.get_outputs()[1].name
        self._landOutHand = self._landmarkSess.get_outputs()[2].name
        self._anchors = self._buildAnchors()
        if "CUDAExecutionProvider" in self._palmSess.get_providers():
            self._palmBinding     = self._palmSess.io_binding()
            self._landmarkBinding = self._landmarkSess.io_binding()
            self._useCudaBinding  = True
        else:
            self._useCudaBinding = False

    def _buildAnchors(self):
        anchors = []
        inputH, inputW = self._PALM_INPUT_SIZE[1], self._PALM_INPUT_SIZE[0]
        anchorConfig = [
            (8,  2),
            (16, 6),
        ]
        for stride, count in anchorConfig:
            gridH = int(np.ceil(inputH / stride))
            gridW = int(np.ceil(inputW / stride))
            for y in range(gridH):
                for x in range(gridW):
                    cx = (x + 0.5) / gridW
                    cy = (y + 0.5) / gridH
                    for _ in range(count):
                        anchors.append([cx, cy])
        return np.array(anchors, dtype=np.float32)

    def _preprocessPalm(self, image):
        h, w = image.shape[:2]
        ih, iw = int(self._PALM_INPUT_SIZE[1]), int(self._PALM_INPUT_SIZE[0])
        ratio = min(iw / w, ih / h)
        rw, rh = int(w * ratio), int(h * ratio)
        resized = cv2.resize(image, (rw, rh))
        padTop = (ih - rh) // 2
        padLeft = (iw - rw) // 2
        canvas = np.zeros((ih, iw, 3), dtype=np.float32)
        canvas[padTop:padTop+rh, padLeft:padLeft+rw] = resized.astype(np.float32) / 255.0
        blob = canvas[np.newaxis]
        padBias = np.array([padLeft, padTop], dtype=np.float32)
        return blob, ratio, padBias, np.array([w, h], dtype=np.float32)

    def _runPalmDetection(self, image):
        blob, ratio, padBias, origWH = self._preprocessPalm(image)
        if self._useCudaBinding:
            b = self._palmBinding
            b.bind_cpu_input(self._palmInputName, blob)
            b.bind_output(self._palmOutBox,   "cuda")
            b.bind_output(self._palmOutScore, "cuda")
            self._palmSess.run_with_iobinding(b)
            outs     = b.get_outputs()
            rawBox   = outs[0].numpy()
            rawScore = outs[1].numpy()
        else:
            rawBox, rawScore = self._palmSess.run(
                [self._palmOutBox, self._palmOutScore],
                {self._palmInputName: blob}
            )
        # rawBox: [1,2016,18] — first 4 are cx,cy,w,h deltas; rest are landmark deltas
        # rawScore: [1,2016,1]
        boxDelta = rawBox[0, :, :4]       # (2016, 4)
        landmarkDelta = rawBox[0, :, 4:]  # (2016, 14) → 7 palm landmarks × 2
        scores = rawScore[0, :, 0].astype(np.float64)
        scores = 1.0 / (1.0 + np.exp(-scores))

        scale = max(origWH)
        cxyDelta = boxDelta[:, :2] / self._PALM_INPUT_SIZE
        whDelta = boxDelta[:, 2:] / self._PALM_INPUT_SIZE
        xy1 = (cxyDelta - whDelta / 2 + self._anchors) * scale
        xy2 = (cxyDelta + whDelta / 2 + self._anchors) * scale
        boxes = np.concatenate([xy1, xy2], axis=1)
        boxes -= [padBias[0], padBias[1], padBias[0], padBias[1]]

        keepIdx = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self._SCORE_THRESHOLD, self._NMS_THRESHOLD, top_k=5
        )
        if len(keepIdx) == 0:
            return np.empty((0, 19))
        keepIdx = np.array(keepIdx).flatten()
        selectedScore = scores[keepIdx]
        selectedBox = boxes[keepIdx]
        selectedLm = landmarkDelta[keepIdx].reshape(-1, 7, 2)
        selectedLm = selectedLm / self._PALM_INPUT_SIZE
        selectedAnchors = self._anchors[keepIdx]
        for idx, lm in enumerate(selectedLm):
            lm += selectedAnchors[idx]
        selectedLm *= scale
        selectedLm -= padBias
        return np.c_[selectedBox.reshape(-1, 4), selectedLm.reshape(-1, 14), selectedScore.reshape(-1, 1)]

    def _cropAndPadFromPalm(self, image, palmBbox, forRotation=False):
        whPalm = palmBbox[1] - palmBbox[0]
        if forRotation:
            shiftVec = np.array([0.0, 0.0]) * whPalm
            enlargeScale = 4.0
        else:
            shiftVec = np.array([0.0, -0.4]) * whPalm
            enlargeScale = 3.0
        palmBbox = palmBbox + shiftVec
        centerPalm = np.sum(palmBbox, axis=0) / 2
        newHalfSize = (palmBbox[1] - palmBbox[0]) * enlargeScale / 2
        palmBbox = np.array([centerPalm - newHalfSize, centerPalm + newHalfSize]).astype(np.int32)
        palmBbox[:, 0] = np.clip(palmBbox[:, 0], 0, image.shape[1])
        palmBbox[:, 1] = np.clip(palmBbox[:, 1], 0, image.shape[0])
        crop = image[palmBbox[0][1]:palmBbox[1][1], palmBbox[0][0]:palmBbox[1][0], :]
        if forRotation:
            sideLen = int(np.linalg.norm(crop.shape[:2]))
        else:
            sideLen = max(crop.shape[:2])
        padH = sideLen - crop.shape[0]
        padW = sideLen - crop.shape[1]
        left = padW // 2; top = padH // 2
        right = padW - left; bottom = padH - top
        crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        bias = palmBbox[0] - np.array([left, top], dtype=np.int32)
        return crop, palmBbox, bias

    def _preprocessLandmark(self, image, palm):
        palmIdxBase = 0
        palmIdxMiddle = 2
        padBias = np.array([0, 0], dtype=np.int32)
        palmBbox = palm[0:4].reshape(2, 2)
        image, palmBbox, bias = self._cropAndPadFromPalm(image, palmBbox, forRotation=True)
        padBias += bias
        palmBbox = palmBbox - padBias
        palmLandmarks = palm[4:18].reshape(7, 2) - padBias
        p1 = palmLandmarks[palmIdxBase]
        p2 = palmLandmarks[palmIdxMiddle]
        radians = np.pi / 2 - np.arctan2(-(p2[1] - p1[1]), p2[0] - p1[0])
        radians = radians - 2 * np.pi * np.floor((radians + np.pi) / (2 * np.pi))
        angle = np.rad2deg(radians)
        centerPalm = np.sum(palmBbox, axis=0) / 2
        rotMat = cv2.getRotationMatrix2D(tuple(centerPalm.astype(float)), angle, 1.0)
        rotatedImage = cv2.warpAffine(image, rotMat, (image.shape[1], image.shape[0]))
        homoCoord = np.c_[palmLandmarks, np.ones(palmLandmarks.shape[0])]
        rotatedLm = np.array([homoCoord @ rotMat[0], homoCoord @ rotMat[1]])
        rotatedBbox = np.array([np.amin(rotatedLm, axis=1), np.amax(rotatedLm, axis=1)])
        crop, rotatedBbox, _ = self._cropAndPadFromPalm(rotatedImage, rotatedBbox)
        iw, ih = int(self._LANDMARK_INPUT_SIZE[0]), int(self._LANDMARK_INPUT_SIZE[1])
        blob = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        return blob[np.newaxis], rotatedBbox, angle, rotMat, padBias

    def _runLandmarkRegression(self, image, palm):
        blob, rotatedBbox, angle, rotMat, padBias = self._preprocessLandmark(image, palm)
        if self._useCudaBinding:
            b = self._landmarkBinding
            b.bind_cpu_input(self._landmarkInputName, blob)
            b.bind_output(self._landOutLm,   "cuda")
            b.bind_output(self._landOutConf, "cuda")
            b.bind_output(self._landOutHand, "cuda")
            self._landmarkSess.run_with_iobinding(b)
            outs    = b.get_outputs()
            lmRaw   = outs[0].numpy()
            confRaw = outs[1].numpy()
            handRaw = outs[2].numpy()
        else:
            lmRaw, confRaw, handRaw = self._landmarkSess.run(
                [self._landOutLm, self._landOutConf, self._landOutHand],
                {self._landmarkInputName: blob}
            )
        conf = float(confRaw[0][0])
        if conf < self._LANDMARK_CONF_THRESHOLD:
            return None, None
        handScore = float(handRaw[0][0])  # >0.5 → right hand in unflipped frame
        landmarks = lmRaw[0].reshape(21, 3)
        iw, ih = int(self._LANDMARK_INPUT_SIZE[0]), int(self._LANDMARK_INPUT_SIZE[1])
        bboxWH = (rotatedBbox[1] - rotatedBbox[0]).astype(float)
        landmarks[:, 0] = landmarks[:, 0] / iw * bboxWH[0] + rotatedBbox[0][0]
        landmarks[:, 1] = landmarks[:, 1] / ih * bboxWH[1] + rotatedBbox[0][1]
        center = np.sum(rotatedBbox, axis=0) / 2
        invAngle = -angle
        invMat = cv2.getRotationMatrix2D(tuple(center.astype(float)), invAngle, 1.0)
        homo = np.c_[landmarks[:, :2], np.ones(21)]
        derotXY = np.c_[homo @ invMat[0], homo @ invMat[1]]
        landmarks[:, :2] = derotXY + padBias
        return landmarks, handScore

    def detect_for_video(self, frameOrMpImage, timestampMs):
        if hasattr(frameOrMpImage, 'numpy_view'):
            frame = frameOrMpImage.numpy_view()  # mp.Image → numpy (RGB)
        else:
            frame = cv2.cvtColor(frameOrMpImage, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        palms = self._runPalmDetection(frame)
        handLandmarksList = []
        handednessList = []
        for palm in palms:
            landmarks, handScore = self._runLandmarkRegression(frame, palm)
            if landmarks is None:
                continue
            lmList = [
                NormalizedLandmark(x=float(landmarks[j, 0]) / w,
                                   y=float(landmarks[j, 1]) / h,
                                   z=float(landmarks[j, 2]))
                for j in range(21)
            ]
            # handScore from model: >0.5 = right hand (unflipped). After cv2.flip
            # in processFrame the labels invert, matching MediaPipe convention.
            handName = "Right" if handScore > 0.5 else "Left"
            handLandmarksList.append(lmList)
            handednessList.append([_HandednessEntry(handName)])
        return _DetectionResult(handLandmarksList, handednessList)


# ── Constants ──────────────────────────────────────────────────────────────────

MODEL_PATH = (
    Path.home() / ".local" / "share" / "gesturecontrol" / "hand_landmarker.task"
)
POSE_MODEL_PATH = MODEL_PATH.parent / "pose_landmarker_lite.task"
PERSON_DETECT_MODEL_PATH = MODEL_PATH.parent / "person_detection_mediapipe_2023mar.onnx"
DEFAULT_CONFIG = Path.home() / ".config" / "gesturecontrol" / "triggers.toml"

DBUS_NAME = "org.gesturecontrol"
DBUS_PATH = "/org/gesturecontrol"
DBUS_IFACE = "org.gesturecontrol.Engine"

# MediaPipe Tasks API aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

DEBUG = False

# ── Config dataclasses ─────────────────────────────────────────────────────────


@dataclass
class PoseTrigger:
    hand: str
    shape: str
    dwellMs: int

    @classmethod
    def parse(cls, d, defaultDwellMs):
        return cls(hand=d.get("hand", "right"), shape=d["shape"],
                   dwellMs=d.get("dwellMs", d.get("dwell_ms", defaultDwellMs)))

    def buildState(self):
        return BindingState(debouncer=DwellDebouncer(self.dwellMs))

    def process(self, bState, handData, timestampMs, publisher, name, condsMet, suppress, gracePeriodMs=0):
        pose = None if suppress else getPoseForHand(handData, self.hand)
        if pose is None and self.shape is not None:
            if gracePeriodMs > 0 and timestampMs - bState.graceTimeMs < gracePeriodMs and bState.graceValue == self.shape:
                pose = bState.graceValue
        if bState.debouncer.update(pose if pose == self.shape else None):
            publisher.gestureFired(name, self.hand)


@dataclass
class SwipeTrigger:
    hand: str
    direction: str
    minDisplacement: float

    @classmethod
    def parse(cls, d, defaultDwellMs):
        return cls(hand=d.get("hand", "right"),
                   direction=d["direction"].upper() + "_SWIPE",
                   minDisplacement=d.get("minDisplacement", d.get("min_displacement", 0.15)))

    def buildState(self):
        return BindingState()

    def process(self, bState, handData, timestampMs, publisher, name, condsMet, suppress, gracePeriodMs=0):
        if not suppress and getSwipeForHand(handData, self.hand) == self.direction:
            publisher.gestureFired(name, self.hand)


@dataclass
class SequenceTrigger:
    hand: str
    steps: list
    windowMs: int
    stepDwellMs: int

    @classmethod
    def parse(cls, d, defaultDwellMs):
        return cls(hand=d.get("hand", "right"), steps=d["steps"],
                   windowMs=d.get("windowMs", d.get("window_ms", 3000)),
                   stepDwellMs=d.get("stepDwellMs", d.get("step_dwell_ms", 100)))

    def buildState(self):
        return BindingState(sequenceTracker=SequenceTracker(
            self.steps, self.windowMs, self.stepDwellMs))

    def process(self, bState, handData, timestampMs, publisher, name, condsMet, suppress, gracePeriodMs=0):
        pose = None if suppress else getPoseForHand(handData, self.hand)
        if pose is None and gracePeriodMs > 0 and timestampMs - bState.graceTimeMs < gracePeriodMs:
            pose = bState.graceValue
        stepsCompleted, done = bState.sequenceTracker.update(pose, timestampMs)
        if stepsCompleted:
            publisher.sequenceProgress(name, self.hand, stepsCompleted, len(self.steps))
        if done:
            publisher.gestureFired(name, self.hand)


@dataclass
class ContinuousTrigger:
    hand: str
    metric: str
    valueRange: tuple
    hysteresis: float = 0.04

    @classmethod
    def parse(cls, d, defaultDwellMs):
        return cls(hand=d.get("hand", "right"), metric=d["metric"],
                    valueRange=parseRange(d["range"]) if "range" in d else (0.0, 1.0),
                    hysteresis=float(d.get("hysteresis", 0.04)))

    def buildState(self):
        return BindingState(continuousTracker=ContinuousTracker(self))

    def process(self, bState, handData, timestampMs, publisher, name, condsMet, suppress, gracePeriodMs=0):
        result = handData.get(self.hand) or _emptyResult()
        tracker = bState.continuousTracker
        wasActive = tracker.active
        metrics = result.metrics
        if not metrics and gracePeriodMs > 0 and timestampMs - bState.graceTimeMs < gracePeriodMs and bState.graceValue is not None:
            metrics = {self.metric: bState.graceValue}
        value, ended = tracker.update(
            metrics, timestampMs, enabled=condsMet and not suppress)
        if not wasActive and tracker.active:
            publisher.continuousStart(name, self.hand)
            publisher.awaitSlotConfig(name, timeoutMs=50)
        if value is not None:
            bState.graceValue = value
            bState.graceTimeMs = timestampMs
            publisher.continuousUpdate(name, self.hand, publisher.applySlotConfig(name, tracker, value))
        if ended:
            publisher.continuousEnd(name, self.hand)


@dataclass
class ChordTrigger:
    left: str
    right: str
    dwellMs: int

    @classmethod
    def parse(cls, d, defaultDwellMs):
        return cls(left=d["left"], right=d["right"],
                   dwellMs=d.get("dwellMs", d.get("dwell_ms", defaultDwellMs)))

    def buildState(self):
        return BindingState(debouncer=DwellDebouncer(self.dwellMs))

    def process(self, bState, handData, timestampMs, publisher, name, condsMet, suppress, gracePeriodMs=0):
        leftResult  = handData.get("left")  or _emptyResult()
        rightResult = handData.get("right") or _emptyResult()
        chordHeld = leftResult.pose == self.left and rightResult.pose == self.right
        if bState.debouncer.update("chord" if chordHeld else None):
            publisher.gestureFired(name, "both")


@dataclass
class SequencedContinuousTrigger:
    """Continuous trigger gated behind a prefix sequence.

    The prefix sequence must complete before the continuous phase activates.
    When the continuous phase ends (condsMet goes false or hand leaves frame),
    the prefix resets and must be repeated.
    """
    hand: str
    prefixSteps: list
    prefixWindowMs: int
    prefixStepDwellMs: int
    metric: str
    valueRange: tuple
    hysteresis: float = 0.04

    @classmethod
    def parse(cls, d, defaultDwellMs):
        return cls(
            hand=d.get("hand", "right"),
            prefixSteps=d.get("prefixSteps", d.get("prefix_steps", [])),
            prefixWindowMs=d.get("prefixWindowMs", d.get("prefix_window_ms", 1500)),
            prefixStepDwellMs=d.get("prefixDwellMs", d.get("prefix_dwell_ms", defaultDwellMs)),
            metric=d["metric"],
            valueRange=parseRange(d["range"]) if "range" in d else (0.0, 1.0),
            hysteresis=float(d.get("hysteresis", 0.04)),
        )

    def buildState(self):
        return BindingState(
            sequenceTracker=SequenceTracker(
                self.prefixSteps, self.prefixWindowMs, self.prefixStepDwellMs),
            continuousTracker=ContinuousTracker(self),
        )

    def process(self, bState, handData, timestampMs, publisher, name, condsMet, suppress, gracePeriodMs=0):
        if not bState.prefixComplete:
            pose = None if suppress else getPoseForHand(handData, self.hand)
            if pose is None and gracePeriodMs > 0 and timestampMs - bState.graceTimeMs < gracePeriodMs:
                pose = bState.graceValue
            stepsCompleted, done = bState.sequenceTracker.update(pose, timestampMs)
            if stepsCompleted:
                publisher.sequenceProgress(name, self.hand, stepsCompleted, len(self.prefixSteps))
            if done:
                bState.prefixComplete = True
            return

        lastPose = self.prefixSteps[-1]
        currentPose = getPoseForHand(handData, self.hand)
        if currentPose != lastPose:
            if bState.continuousTracker.active:
                publisher.continuousEnd(name, self.hand)
            bState.prefixComplete = False
            bState.sequenceTracker.reset()
            return

        result = handData.get(self.hand) or _emptyResult()
        tracker = bState.continuousTracker
        wasActive = tracker.active
        metrics = result.metrics
        if not metrics and gracePeriodMs > 0 and timestampMs - bState.graceTimeMs < gracePeriodMs and bState.graceValue is not None:
            metrics = {self.metric: bState.graceValue}
        value, ended = tracker.update(
            metrics, timestampMs, enabled=condsMet and not suppress)
        if not wasActive and tracker.active:
            publisher.continuousStart(name, self.hand)
            publisher.awaitSlotConfig(name, timeoutMs=50)
        if value is not None:
            bState.graceValue = value
            bState.graceTimeMs = timestampMs
            publisher.continuousUpdate(name, self.hand, publisher.applySlotConfig(name, tracker, value))
        if ended:
            publisher.continuousEnd(name, self.hand)
            bState.prefixComplete = False
            bState.sequenceTracker.reset()


@dataclass
class PoseDefinition:
    name: str
    thumb: bool | None = None
    index: bool | None = None
    middle: bool | None = None
    ring: bool | None = None
    pinky: bool | None = None
    spreadThumbIndex: str | float | None = None
    spreadIndexMiddle: str | float | None = None
    spreadMiddleRing: str | float | None = None
    spreadRingPinky: str | float | None = None


@dataclass
class Binding:
    name: str
    trigger: object
    requirePoses: list = None


# ── Config loading ─────────────────────────────────────────────────────────────


def parseRange(raw):
    """Convert a [min, max] list from config into a (float, float) tuple."""
    return (float(raw[0]), float(raw[1]))


_TRIGGER_CLASSES = {
    "pose":                  PoseTrigger,
    "swipe":                 SwipeTrigger,
    "sequence":              SequenceTrigger,
    "continuous":            ContinuousTrigger,
    "chord":                 ChordTrigger,
    "sequencedContinuous":  SequencedContinuousTrigger,
    "sequenced_continuous": SequencedContinuousTrigger,
}


def parseTrigger(d, defaultDwellMs):
    """Dispatch to the appropriate trigger class's parse() classmethod."""
    kind = d["type"]
    cls = _TRIGGER_CLASSES.get(kind)
    if cls is None:
        raise ValueError(f"Unknown trigger type: {kind!r}")
    return cls.parse(d, defaultDwellMs)


def parsePose(d):
    """Build a PoseDefinition from a raw config dict. Absent fields default to None (don't care)."""
    def spreadVal(key):
        v = d.get(key)
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        return v  # "close" or "apart"

    return PoseDefinition(
        name=d["name"],
        thumb=d.get("thumb"),
        index=d.get("index"),
        middle=d.get("middle"),
        ring=d.get("ring"),
        pinky=d.get("pinky"),
        spreadThumbIndex=spreadVal("spreadThumbIndex"),
        spreadIndexMiddle=spreadVal("spreadIndexMiddle"),
        spreadMiddleRing=spreadVal("spreadMiddleRing"),
        spreadRingPinky=spreadVal("spreadRingPinky"),
    )


def loadConfig(path):
    """Load triggers.toml. Returns (settings dict, list of PoseDefinition, list of Binding)."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    settings = raw.get("settings", {})
    defaultDwellMs = settings.get("dwellMs", settings.get("dwell_ms", 200))
    gracePeriodMs = settings.get("gracePeriodMs", settings.get("grace_period_ms", 0))
    poses = [parsePose(p) for p in raw.get("poses", [])]
    bindings = [
        Binding(
            name=item["name"],
            trigger=parseTrigger(item["trigger"], defaultDwellMs),
            requirePoses=item.get("require", []),
        )
        for item in raw.get("bindings", [])
    ]
    # spreadThreshold can be tuned in [settings]; defaults to DEFAULT_SPREAD_THRESHOLD
    spreadThresholdVal = settings.get("spreadThreshold", settings.get("spread_threshold", None))
    settings.setdefault("spreadThreshold", spreadThresholdVal or DEFAULT_SPREAD_THRESHOLD)
    presence = raw.get("presence", {})
    return settings, poses, bindings, presence


# ── Continuous metrics ─────────────────────────────────────────────────────────


def measureMetric(landmarks, metric):
    """Compute a single raw float for a named metric from hand landmarks."""
    if metric in ("pinchDistance", "pinch_distance"):
        dx = landmarks[4].x - landmarks[8].x
        dy = landmarks[4].y - landmarks[8].y
        return (dx * dx + dy * dy) ** 0.5
    if metric in ("handHeight", "hand_height"):
        # Invert y so that raising the hand produces a higher value
        return 1.0 - landmarks[0].y
    if metric in ("handX", "hand_x"):
        return landmarks[0].x
    if metric in ("fingerSpread", "finger_spread"):
        xs = [landmarks[i].x for i in [4, 8, 12, 16, 20]]
        return max(xs) - min(xs)
    if metric == "angle":
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
        for m in ("pinchDistance", "pinch_distance", "handHeight", "hand_height",
        "handX", "hand_x", "fingerSpread", "finger_spread", "angle")
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
        self.history = deque()  # [(timestamp_ms, x), ...]
        self.lastFiredAt = 0

    def update(self, indexTipX, timestampMs):
        """Return 'LEFT_SWIPE', 'RIGHT_SWIPE', or None."""
        self.history.append((timestampMs, indexTipX))
        cutoff = timestampMs - self.windowMs
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

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
        self.history = deque()  # [(timestamp_ms, x, y), ...]

    def update(self, x, y, timestampMs):
        self.history.append((timestampMs, x, y))
        cutoff = timestampMs - self.windowMs
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()

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

    pose: str | None
    swipe: str | None
    metrics: dict
    rawPose: str | None
    isMoving: bool
    fingers: list | None = None
    spreads: dict | None = None


class HandProcessor:
    """Bundles SwipeDetector, MotionFilter, and pose classification
    for a single hand. Stateful across frames."""

    def __init__(self, poses, spreadThreshold=DEFAULT_SPREAD_THRESHOLD):
        self.poses = poses
        self.spreadThreshold = spreadThreshold
        self.swipeDetector = SwipeDetector()
        self.motionFilter = MotionFilter()

    def update(self, landmarks, mpLabel, timestampMs):
        """Process one frame for this hand and return a HandFrameResult."""
        self.motionFilter.update(landmarks[0].x, landmarks[0].y, timestampMs)
        swipe = self.swipeDetector.update(landmarks[8].x, timestampMs)
        rawPose = classifyPose(landmarks, mpLabel, self.poses, self.spreadThreshold)
        pose = None if self.motionFilter.isMoving() else rawPose
        metrics = measureAllMetrics(landmarks)
        spreads = computeFingerSpreads(landmarks)
        return HandFrameResult(
            pose=pose,
            swipe=swipe,
            metrics=metrics,
            rawPose=rawPose,
            isMoving=self.motionFilter.isMoving(),
            fingers=fingerStates(landmarks, mpLabel),
            spreads=spreads,
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
        self.firstStepTimeMs = None

    def update(self, pose, timestampMs):
        """Return (stepsCompleted: int, done: bool).

        stepsCompleted is 0 if no step fired this frame, otherwise the
        running count of completed steps (1..N). done is True on the
        frame when the final step completes.
        """
        if self.firstStepTimeMs is not None:
            if timestampMs - self.firstStepTimeMs > self.windowMs:
                self.reset()

        expected = self.steps[self.currentStep]
        debouncerInput = pose if pose == expected else None
        if not self.stepDebouncer.update(debouncerInput):
            return 0, False

        if self.currentStep == 0:
            self.firstStepTimeMs = timestampMs
        self.currentStep += 1
        stepsCompleted = self.currentStep

        if self.currentStep >= len(self.steps):
            self.reset()
            return stepsCompleted, True

        self.stepDebouncer.reset()
        return stepsCompleted, False

    def reset(self):
        self.currentStep = 0
        self.firstStepTimeMs = None
        self.stepDebouncer.reset()


# ── Continuous tracker ─────────────────────────────────────────────────────────


class ContinuousTracker:
    """Tracks a ContinuousTrigger: active while binding conditions are met.

    Returns (normalizedValue, justEnded) each frame.
    normalizedValue is None when the trigger is inactive.
    justEnded is True on the single frame the trigger deactivates.
    """

    def __init__(self, trigger):
        self.trigger = trigger
        self.active = False
        self.currentSlot = None

    def update(self, metrics, timestampMs, enabled=True):
        shouldBeActive = enabled

        if self.active and not shouldBeActive:
            self.active = False
            self.currentSlot = None
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

    binding: Binding = None
    debouncer: object = None
    sequenceTracker: object = None
    continuousTracker: object = None
    prefixComplete: bool = False
    gracePose: str = None
    graceValue: float = None
    graceTimeMs: float = 0.0

    def getWithGrace(self, value, timestampMs, gracePeriodMs):
        """Return value with grace period handling for None detection loss."""
        if value is not None:
            self.gracePose = value
            self.graceValue = value
            self.graceTimeMs = timestampMs
            return value
        if gracePeriodMs > 0 and timestampMs - self.graceTimeMs < gracePeriodMs:
            return self.graceValue
        return None


def buildBindingState(binding, defaultDwellMs):
    """Create a BindingState by delegating to the trigger's buildState() method."""
    bState = binding.trigger.buildState()
    bState.binding = binding
    return bState


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

    def __init__(self, bindings, publisher, defaultDwellMs, gracePeriodMs=0):
        self.publisher = publisher
        self.gracePeriodMs = gracePeriodMs
        self.states = [buildBindingState(b, defaultDwellMs) for b in bindings]

    def update(self, handData, timestampMs):
        """Process one frame. handData = {side: HandFrameResult}."""
        for state in self.states:
            self._processBinding(state, handData, timestampMs)

    def _checkConditions(self, binding, handData):
        """Return True if all require conditions are met."""
        for req in (binding.requirePoses or []):
            hand = req["hand"]
            pose = req["pose"]
            if hand == "either":
                leftPose  = (handData.get("left")  or _emptyResult()).pose
                rightPose = (handData.get("right") or _emptyResult()).pose
                if leftPose != pose and rightPose != pose:
                    return False
            else:
                actual = (handData.get(hand) or _emptyResult()).pose
                if actual != pose:
                    return False
        return True

    def _isSingleHand(self, binding):
        """True if this binding should be suppressed when both hands are visible.

        ChordTrigger is inherently two-handed. Bindings with requirePoses are
        deliberately conditioned on specific hand poses. SequenceTrigger requires
        deliberate ordered steps and must not be suppressed — doing so would
        prevent any sequence from firing when both hands are in frame.
        Everything else is a single-hand trigger that should be suppressed while
        both hands are visible, to prevent accidental firing during chord gestures.
        """
        return (
            not isinstance(binding.trigger, (ChordTrigger, SequenceTrigger))
            and not binding.requirePoses
        )

    def _processBinding(self, bState, handData, timestampMs):
        b = bState.binding
        conditionsMet = self._checkConditions(b, handData)
        suppress = len(handData) == 2 and self._isSingleHand(b)

        if not isinstance(b.trigger, ContinuousTrigger) and not conditionsMet:
            return

        b.trigger.process(bState, handData, timestampMs, self.publisher, b.name,
                          conditionsMet, suppress, self.gracePeriodMs)


# ── D-Bus publisher ────────────────────────────────────────────────────────────


class GestureEngineService(dbus.service.Object):
    """D-Bus object that broadcasts gesture signals and receives method calls.

    Signals are emitted synchronously via dbus-python decorators.
    RegisterSlots is a method that any subscriber can call to configure
    hysteresis-based slot mapping for a continuous binding.
    """

    def __init__(self, bus, publisher):
        super().__init__(bus, DBUS_PATH)
        self._publisher = publisher

    @dbus.service.signal(DBUS_IFACE, signature="ss")
    def GestureFired(self, name, hand):
        pass

    @dbus.service.signal(DBUS_IFACE, signature="ss")
    def ContinuousStart(self, name, hand):
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

    @dbus.service.method(DBUS_IFACE, in_signature="si", out_signature="")
    def RegisterSlots(self, name, slots):
        """Configure slot-based mapping for a continuous binding.

        Any subscriber can call this via D-Bus to tell the engine to map the
        raw [0, 1] continuous value to discrete 1..slots indices before emitting
        ContinuousUpdate. The hysteresis amount is read from the binding's
        triggers.toml config. Call before or immediately after ContinuousStart;
        reset is automatic on ContinuousEnd.

        Args:
            name:  binding name (matches triggers.toml)
            slots: number of discrete slots (0 = raw passthrough)
        """
        self._publisher.registerSlots(str(name), int(slots))


class GesturePublisher:
    """Owns the D-Bus connection, service object, and slot-mapping state.

    Signals are emitted synchronously from the camera loop thread.
    RegisterSlots method calls arrive on the GLib mainloop thread and update
    _slotRegistry; awaitSlotConfig blocks the camera loop briefly on gesture
    start so the subscriber's RegisterSlots call can arrive before the first
    ContinuousUpdate is emitted.
    """

    def __init__(self):
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        bus = dbus.SessionBus()
        bus.request_name(DBUS_NAME)
        self._slotRegistry = {}
        self._slotEvents = {}
        self._service = GestureEngineService(bus, self)
        self._gloop = GLib.MainLoop()
        threading.Thread(target=self._gloop.run, daemon=True).start()

    def registerSlots(self, name, slots):
        """Called by GestureEngineService.RegisterSlots on the GLib thread."""
        self._slotRegistry[name] = slots
        event = self._slotEvents.get(name)
        if event:
            event.set()
        print(f"[slots]  RegisterSlots       {name}  slots={slots}")

    def awaitSlotConfig(self, name, timeoutMs=50):
        """Block until RegisterSlots arrives for `name`, or timeoutMs elapses.

        Called on the camera thread immediately after ContinuousStart so the
        subscriber has a chance to call RegisterSlots before the first value
        is emitted. Returns immediately if the binding is already configured
        (e.g. registered at startup).
        """
        if name in self._slotRegistry:
            return
        event = threading.Event()
        self._slotEvents[name] = event
        event.wait(timeout=timeoutMs / 1000.0)
        self._slotEvents.pop(name, None)

    def applySlotConfig(self, name, tracker, value):
        """Map raw [0,1] value to a slot index using hysteresis if configured.

        Slots are set by the subscriber via RegisterSlots. Hysteresis is read
        from the binding's trigger config (triggers.toml). Mutates
        tracker.currentSlot to track state across frames. Returns the slot
        index as a float (1.0..N.0), or the raw value if no slot config is
        registered for this binding.
        """
        slots = self._slotRegistry.get(name)
        if slots is None or slots <= 0:
            return value
        hysteresis = getattr(tracker.trigger, "hysteresis", 0.04)
        if tracker.currentSlot is None:
            tracker.currentSlot = max(1, min(slots, math.floor(value * slots) + 1))
        else:
            lower = (tracker.currentSlot - 1) / slots - hysteresis
            upper =  tracker.currentSlot      / slots + hysteresis
            if value < lower:
                tracker.currentSlot = max(1, tracker.currentSlot - 1)
            elif value > upper:
                tracker.currentSlot = min(slots, tracker.currentSlot + 1)
        return float(tracker.currentSlot)

    def gestureFired(self, name, hand):
        print(f"[signal] GestureFired       {name}  ({hand})")
        self._service.GestureFired(name, hand)

    def continuousStart(self, name, hand):
        print(f"[signal] ContinuousStart    {name}  ({hand})")
        self._service.ContinuousStart(name, hand)

    def continuousUpdate(self, name, hand, value):
        self._service.ContinuousUpdate(name, hand, float(value))

    def continuousEnd(self, name, hand):
        print(f"[signal] ContinuousEnd      {name}  ({hand})")
        self._service.ContinuousEnd(name, hand)

    def sequenceProgress(self, name, hand, step, total):
        print(f"[signal] SequenceProgress   {name}  {step}/{total}  ({hand})")
        self._service.SequenceProgress(name, hand, step, total)

    def stop(self):
        self._gloop.quit()


# ── Debug overlay ──────────────────────────────────────────────────────────────


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
    VALID_METRICS = ("pinchDistance", "pinch_distance", "handHeight", "hand_height",
                     "handX", "hand_x", "fingerSpread", "finger_spread", "angle")
    if metric not in VALID_METRICS:
        print(f"ERROR: unknown metric '{metric}'. Choose from: {', '.join(VALID_METRICS)}", file=sys.stderr)
        sys.exit(1)

    print(f"Calibrating '{metric}' (hand={hand})")
    print("Hold your pose...")

    for i in range(countdown, 0, -1):
        print(f"  {i}...", flush=True)
        time.sleep(1.0)
    print("Sampling!", flush=True)

    landmarker = buildHandLandmarker()
    cap = openCamera(camInput)
    detector = DarkFrameDetector()
    samples = []
    deadline = time.monotonic() + sampleSecs

    while time.monotonic() < deadline:
        ret, frame = cap.read()
        if not ret:
            break
        if detector.isDark(frame.mean()):
            continue
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        ts = int(time.monotonic() * 1000)
        result = landmarker.detect_for_video(frame, ts)
        for i, handedness in enumerate(result.handedness):
            mpLabel = handedness[0].categoryName
            side = "right" if mpLabel == "Left" else "left"
            if hand != "either" and side != hand:
                continue
            samples.append(measureMetric(result.handLandmarks[i], metric))

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


class _PoseDetectResult:
    __slots__ = ("poseLandmarks",)
    def __init__(self, poseLandmarks):
        self.poseLandmarks = poseLandmarks


class PoseLandmarkerONNX:
    def __init__(self, modelPath, minConfidence=0.2):
        self._minConfidence = minConfidence
        self._landmarker = PoseLandmarker.create_from_options(
            PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(modelPath)),
                running_mode=VisionRunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=minConfidence,
                min_pose_presence_confidence=minConfidence,
            )
        )
        print(f"[presence] PoseLandmarker loaded from {modelPath.name} (minConf={minConfidence})", file=sys.stderr)

    def detect_for_video(self, frameBgr, timestampMs):
        frameRgb = cv2.cvtColor(frameBgr, cv2.COLOR_BGR2RGB)
        mpImg = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRgb)
        result = self._landmarker.detect_for_video(mpImg, timestampMs)
        landmarks = result.pose_landmarks if result.pose_landmarks else []
        return _PoseDetectResult(landmarks)


def buildHandLandmarker():
    palmModelPath = Path.home() / ".local/share/gesturecontrol/palm_detection_mediapipe.onnx"
    landmarkModelPath = Path.home() / ".local/share/gesturecontrol/handpose_estimation_mediapipe.onnx"
    lm = HandLandmarkerONNX(palmModelPath, landmarkModelPath)
    if "CUDAExecutionProvider" not in lm._palmSess.get_providers():
        global _onnxCudaNotified
        if not _onnxCudaNotified:
            _onnxCudaNotified = True
            notifyError(
                "gestureControl",
                "Hand tracking running on CPU — GPU (CUDA) not available. Performance may be reduced.",
            )
    return lm


def buildPoseLandmarker(minConfidence=0.5):
    taskModelPath = Path.home() / ".local/share/gesturecontrol/pose_landmarker_lite.task"
    if not taskModelPath.exists():
        print(f"[presence] pose model not found at {taskModelPath}", file=sys.stderr)
        return None
    pm = PoseLandmarkerONNX(taskModelPath, minConfidence)
    return pm


def openCamera(camInput, width=None, height=None, fmt=None):
    """Open the camera, apply format settings, and return the VideoCapture object.

    fmt: explicit fourcc string (e.g. "MJPG", "YUYV", "GREY"). If omitted,
    MJPG is only forced when width/height are also set (RGB camera path).
    IR cameras that don't support MJPG should leave fmt and resolution unset.
    Returns None on failure.
    """
    try:
        camIndex = int(camInput)
    except ValueError:
        camIndex = camInput
    cap = cv2.VideoCapture(camIndex, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {camInput!r}", file=sys.stderr)
        return None
    if fmt:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt[:4].upper()))
    elif width or height:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    actualW = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actualH = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actualFps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[init] camera {actualW}x{actualH} @ {actualFps:.0f} fps (native)")
    return cap


def buildHandMap(result):
    """Map detected hands to user-side keys ("right"/"left").

    After cv2.flip(frame, 1): MediaPipe "Left" label == user's right hand.
    """
    handMap = {}
    for i in range(len(result.handLandmarks)):
        mpLabel = result.handedness[i][0].categoryName
        side = "right" if mpLabel == "Left" else "left"
        handMap[side] = (result.handLandmarks[i], mpLabel)
    return handMap


def processFrame(frame, landmarker, processors, matcher, timestampMs,
                 streamServer=None):
    """Run detection, trigger matching, and optional debug rendering for one frame."""
    frame = cv2.flip(frame, 1)
    if frame.ndim == 2:  # greyscale input (e.g. IR camera)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    result = landmarker.detect_for_video(frame, timestampMs)
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
        self._presence = {}
        server = ThreadingHTTPServer(("127.0.0.1", port), self._makeHandler())
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[stream] serving on http://127.0.0.1:{port}")

    def publish(self, jpegBytes, hands):
        with self._lock:
            self._frame = jpegBytes
            self._hands = hands

    def setPresence(self, p):
        with self._lock:
            self._presence = p

    def _makeHandler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

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
                            data = {
                                "hands": dict(server._hands),
                                "presence": dict(server._presence),
                            }
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

    settings, poses, bindings, presence = loadConfig(args.config)
    defaultDwellMs = settings.get("dwellMs", settings.get("dwell_ms", 200))
    gracePeriodMs = settings.get("gracePeriodMs", settings.get("grace_period_ms", 0))
    spreadThreshold = settings.get("spreadThreshold", settings.get("spread_threshold", DEFAULT_SPREAD_THRESHOLD))
    print(f"Loaded {len(poses)} pose(s), {len(bindings)} binding(s) from {args.config}")
    configWatcher = ConfigWatcher(args.config)

    print("[init] connecting to D-Bus...")
    publisher = GesturePublisher()
    print("[init] D-Bus ready")
    processors = {
        "right": HandProcessor(poses, spreadThreshold),
        "left":  HandProcessor(poses, spreadThreshold),
    }
    matcher = TriggerMatcher(bindings, publisher, defaultDwellMs, gracePeriodMs)
    darkFrameDetector = DarkFrameDetector()

    camInput = args.input if args.input is not None else settings.get("camera", 0)
    print("[init] opening camera...")
    cap = openCamera(
        camInput,
        width=settings.get("width"),
        height=settings.get("height"),
        fmt=settings.get("format"),
    )
    if cap is None:
        sys.exit(1)
    print("[init] camera opened")

    streamServer = None if args.no_stream else StreamServer(args.stream_port)

    stopFlag = threading.Event()
    signal.signal(signal.SIGINT, lambda s, f: stopFlag.set())

    print("[init] loading hand landmarker model...")
    landmarker = buildHandLandmarker()
    poseLandmarker = None
    try:
        print("[init] model ready — entering loop")
        frameCount = 0
        _camKeys = {"camera", "width", "height", "format"}
        inferenceFps = settings.get("fps") or 0
        inferenceInterval = (1.0 / inferenceFps) if inferenceFps > 0 else 0
        lastInferenceTime = 0.0
        presenceEnabled = presence.get("enabled", False)
        presenceIdleSecs = presence.get("idleSeconds", 300)
        presenceThreshold = presence.get("motionThreshold", 5.0)
        presenceCheckHz = presence.get("checkHz", 2)
        presenceInterval = 1.0 / presenceCheckHz if presenceCheckHz > 0 else 0.5
        presencePoseDetection = presence.get("poseDetection", False)
        presencePoseMinConf = presence.get("poseMinConfidence", 0.5)
        presencePoseCheckMode = presence.get("poseCheckMode", "fallback")
        presencePauseHands = presence.get("pauseHandsWhenAbsent", True)
        useMotionDetection = presence.get("useMotionDetection", True)
        presenceRef = None
        presenceLastCheck = 0.0
        presenceLastActive = time.monotonic()
        presenceBlanked = False
        poseLandmarker = buildPoseLandmarker(presencePoseMinConf) if presencePoseDetection else None
        if presenceEnabled:
            print(f"[presence] enabled — idle blanks screen after {presenceIdleSecs}s")
        while not stopFlag.is_set():
            if configWatcher.pollChanged():
                try:
                    prevSettings = settings
                    settings, poses, bindings, presence = loadConfig(args.config)
                    defaultDwellMs = settings.get("dwellMs", settings.get("dwell_ms", 200))
                    gracePeriodMs = settings.get("gracePeriodMs", settings.get("grace_period_ms", 0))
                    spreadThreshold = settings.get("spreadThreshold", settings.get("spread_threshold", DEFAULT_SPREAD_THRESHOLD))
                    processors = {
                        "right": HandProcessor(poses, spreadThreshold),
                        "left":  HandProcessor(poses, spreadThreshold),
                    }
                    matcher = TriggerMatcher(bindings, publisher, defaultDwellMs, gracePeriodMs)
                    inferenceFps = settings.get("fps") or 0
                    inferenceInterval = (1.0 / inferenceFps) if inferenceFps > 0 else 0
                    presenceEnabled = presence.get("enabled", False)
                    presenceIdleSecs = presence.get("idleSeconds", 300)
                    presenceThreshold = presence.get("motionThreshold", 5.0)
                    presenceCheckHz = presence.get("checkHz", 2)
                    presenceInterval = 1.0 / presenceCheckHz if presenceCheckHz > 0 else 0.5
                    newPoseDetection = presence.get("poseDetection", False)
                    newPoseMinConf = presence.get("poseMinConfidence", 0.5)
                    presencePoseCheckMode = presence.get("poseCheckMode", "fallback")
                    presencePauseHands = presence.get("pauseHandsWhenAbsent", True)
                    newUseMotionDetection = presence.get("useMotionDetection", True)
                    if newUseMotionDetection != useMotionDetection:
                        useMotionDetection = newUseMotionDetection
                    if newPoseDetection != presencePoseDetection or newPoseMinConf != presencePoseMinConf:
                        if poseLandmarker is not None:
                            poseLandmarker = None
                        if newPoseDetection:
                            poseLandmarker = buildPoseLandmarker(newPoseMinConf)
                    presencePoseDetection = newPoseDetection
                    presencePoseMinConf = newPoseMinConf
                    changedCamKeys = {
                        k for k in _camKeys
                        if prevSettings.get(k) != settings.get(k)
                    }
                    if changedCamKeys:
                        oldVals = {k: prevSettings.get(k) for k in changedCamKeys}
                        newVals = {k: settings.get(k) for k in changedCamKeys}
                        print(f"[config] Camera settings changed {oldVals} → {newVals}, reopening camera")
                        newCamInput = args.input if args.input is not None else settings.get("camera", 0)
                        newCap = openCamera(
                            newCamInput,
                            width=settings.get("width"),
                            height=settings.get("height"),
                            fmt=settings.get("format"),
                        )
                        if newCap is not None:
                            cap.release()
                            cap = newCap
                        else:
                            print("[config] Camera reopen failed — keeping existing capture", file=sys.stderr)
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
            if darkFrameDetector is not None and darkFrameDetector.isDark(frame.mean()):
                continue
            now = time.monotonic()
            if presenceEnabled and (now - presenceLastCheck) >= presenceInterval:
                presenceLastCheck = now
                motionDetected = False
                poseDetected = False

                if useMotionDetection:
                    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                       if len(frame.shape) == 3 else frame,
                                       (64, 64)).astype(float)
                    if presenceRef is None:
                        presenceRef = small
                    diff = float(abs(small - presenceRef).mean())
                    # slow drift tracker so lighting changes don't trigger motion
                    presenceRef = 0.95 * presenceRef + 0.05 * small
                    motionDetected = diff > presenceThreshold

                if poseLandmarker is not None and (
                    useMotionDetection and (presencePoseCheckMode == "always" or not motionDetected)
                    or not useMotionDetection
                ):
                    timestampMs = int(now * 1000)
                    poseResult = poseLandmarker.detect_for_video(frame, timestampMs)
                    if poseResult and poseResult.poseLandmarks:
                        poseDetected = True

                motionDetected = motionDetected or poseDetected

                if motionDetected:
                    presenceLastActive = now
                    if presenceBlanked:
                        subprocess.Popen(["xset", "dpms", "force", "on"])
                        presenceBlanked = False
                        if presencePauseHands and landmarker is None:
                            print("[presence] screen on — rebuilding hand landmarker")
                            landmarker = buildHandLandmarker()
                elif not presenceBlanked and (now - presenceLastActive) > presenceIdleSecs:
                    subprocess.Popen(["xset", "dpms", "force", "off"])
                    presenceBlanked = True
                    if presencePauseHands and landmarker is not None:
                        print("[presence] screen off — releasing hand landmarker")
                        landmarker = None

            if streamServer is not None and presenceEnabled:
                streamServer.setPresence({
                    "enabled": presenceEnabled,
                    "motionDetected": motionDetected if presenceEnabled else False,
                    "poseDetected": poseDetected if presenceEnabled else False,
                    "useMotionDetection": useMotionDetection,
                    "screenBlanked": presenceBlanked,
                    "timeSinceLastActive": int((now - presenceLastActive) * 1000),
                    "idleSeconds": presenceIdleSecs,
                    "poseDetection": presencePoseDetection,
                    "checkHz": presenceCheckHz,
                })

            if inferenceInterval == 0 or (now - lastInferenceTime) >= inferenceInterval:
                if landmarker is not None:
                    lastInferenceTime = now
                    timestampMs = int(now * 1000)
                    processFrame(frame, landmarker, processors, matcher, timestampMs,
                                 streamServer=streamServer)
            if cv2.waitKey(1) & 0xFF == ord("q") and DEBUG:
                break
    finally:
        if landmarker is not None:
            landmarker = None
        if poseLandmarker is not None:
            poseLandmarker = None
        cap.release()
        publisher.stop()
        if DEBUG:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
