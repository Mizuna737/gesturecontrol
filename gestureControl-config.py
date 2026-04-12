#!/usr/bin/env python3
"""
gestureControl-config.py — Web-based configuration UI for gestureControl.

Serves a local web app for editing triggers.toml and actions.toml.
Streams a live camera feed with MediaPipe pose-detection overlay.

Usage:
  python gestureControl-config.py [--port 7070] [--config DIR] [--input INDEX]

  --port    HTTP port (default: 7070)
  --config  config directory containing triggers.toml and actions.toml
            (default: ~/.config/gestureControl)
  --input   camera index or path; overrides the camera setting in triggers.toml
"""

import sys
import os
_VENV = os.path.expanduser("~/.local/share/gesturecontrol/venv/bin/python3")
if sys.executable != _VENV and os.path.exists(_VENV):
    os.execv(_VENV, [_VENV] + sys.argv)

import argparse
import atexit
import collections
import json
import threading
import time
import tomllib
import webbrowser
from pathlib import Path

import cv2
import mediapipe as mp
from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

# ── Paths ──────────────────────────────────────────────────────────────────────

MODEL_PATH          = Path.home() / ".local" / "share" / "gesturecontrol" / "hand_landmarker.task"
DEFAULT_CONFIG      = Path.home() / ".config" / "gesturecontrol"
UI_DIR              = Path(__file__).parent / "gestureControl-config-ui"
ENGINE_STREAM_PORT  = 7071  # must match gestureControl.py --stream-port default
LOCK_FILE           = Path.home() / ".local" / "share" / "gesturecontrol" / "config-ui.pid"

# ── MediaPipe aliases ──────────────────────────────────────────────────────────

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

# ── Dark-frame detector ────────────────────────────────────────────────────────

class DarkFrameDetector:
    _WARMUP_THRESHOLD = 20.0
    _WINDOW           = 30
    _MIN_SPREAD       = 5.0
    _SPLIT_FRAC       = 0.4

    def __init__(self):
        self._recent = collections.deque(maxlen=self._WINDOW)

    def isDark(self, mean):
        self._recent.append(mean)
        if len(self._recent) < 10:
            return mean < self._WARMUP_THRESHOLD
        lo = min(self._recent)
        hi = max(self._recent)
        if hi - lo < self._MIN_SPREAD:
            return False
        return mean < lo + (hi - lo) * self._SPLIT_FRAC


# ── Pose helpers ───────────────────────────────────────────────────────────────

def fingerStates(landmarks, handLabel):
    lm     = landmarks
    index  = lm[8].y  < lm[6].y
    middle = lm[12].y < lm[10].y
    ring   = lm[16].y < lm[14].y
    pinky  = lm[20].y < lm[18].y
    isRight = handLabel == "Right"
    thumb  = lm[4].x > lm[3].x if isRight else lm[4].x < lm[3].x
    return [thumb, index, middle, ring, pinky]


def classifyPose(landmarks, handLabel, poses):
    """Match finger states against the pose list; returns first match name or None."""
    states = fingerStates(landmarks, handLabel)
    for pose in poses:
        constraints = [
            pose.get("thumb"), pose.get("index"), pose.get("middle"),
            pose.get("ring"),  pose.get("pinky"),
        ]
        if all(c is None or c == s for c, s in zip(constraints, states)):
            return pose["name"]
    return None


# ── Shared camera state ────────────────────────────────────────────────────────

class CameraState:
    """Thread-safe container for the latest annotated frame and hand state."""

    def __init__(self):
        self._lock  = threading.Lock()
        self._frame = None   # latest JPEG bytes
        self._hands = {}     # {side: {"fingers": [T,I,M,R,P], "pose": str|None}}
        self._error = None   # string if camera failed to open

    def setFrame(self, jpegBytes):
        with self._lock:
            self._frame = jpegBytes

    def setHands(self, hands):
        with self._lock:
            self._hands = hands

    def setError(self, msg):
        with self._lock:
            self._error = msg

    def getFrame(self):
        with self._lock:
            return self._frame

    def getHandState(self):
        with self._lock:
            return {"hands": dict(self._hands), "error": self._error}


# ── Camera thread ──────────────────────────────────────────────────────────────

def cameraThread(camInput, cfgDir, state, stopEvent):
    """Background thread: reads camera frames, runs MediaPipe, publishes state."""
    triggersPath = Path(cfgDir) / "triggers.toml"

    def loadPoses():
        try:
            with open(triggersPath, "rb") as f:
                return tomllib.load(f).get("poses", [])
        except Exception:
            return []

    if not MODEL_PATH.exists():
        state.setError(f"Model not found at {MODEL_PATH}\nRun gestureControl-setup.sh first.")
        return

    try:
        camIndex = int(camInput)
    except (ValueError, TypeError):
        camIndex = camInput

    cap = cv2.VideoCapture(camIndex, cv2.CAP_V4L2)
    if not cap.isOpened():
        state.setError(
            f"Cannot open camera {camInput!r}\n"
            "Is the gesture engine already running?"
        )
        return

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    darkDetector  = DarkFrameDetector()
    poses         = loadPoses()
    nextPoseReload = time.monotonic() + 2.0

    with HandLandmarker.create_from_options(options) as landmarker:
        while not stopEvent.is_set():
            now = time.monotonic()
            if now >= nextPoseReload:
                poses         = loadPoses()
                nextPoseReload = now + 2.0

            ret, frame = cap.read()
            if not ret:
                state.setError("Camera read failed — device disconnected?")
                break

            frame = cv2.flip(frame, 1)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if darkDetector.isDark(frame.mean()):
                continue

            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mpImg = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            tsMs  = int(time.monotonic() * 1000)
            result = landmarker.detect_for_video(mpImg, tsMs)

            hands = {}
            for i, handedness in enumerate(result.handedness):
                mpLabel = handedness[0].category_name
                side    = "right" if mpLabel == "Left" else "left"
                lm      = result.hand_landmarks[i]
                fingers = fingerStates(lm, mpLabel)
                pose    = classifyPose(lm, mpLabel, poses)
                hands[side] = {"fingers": fingers, "pose": pose}

                h, w = frame.shape[:2]
                pts  = [(int(l.x * w), int(l.y * h)) for l in lm]
                for a, b in HAND_CONNECTIONS:
                    cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
                for pt in pts:
                    cv2.circle(frame, pt, 4, (255, 255, 255), -1)

                label = pose or "---"
                yPos  = 36 if side == "right" else 66
                cv2.putText(frame, f"{side[0].upper()}: {label}",
                            (10, yPos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 80), 2)

            state.setHands(hands)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            state.setFrame(buf.tobytes())

    cap.release()


# ── TOML serialization ─────────────────────────────────────────────────────────

def _tomlVal(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_tomlVal(i) for i in v) + "]"
    raise ValueError(f"Cannot TOML-serialize {v!r}")


def _inlineTable(d):
    parts = [f"{k} = {_tomlVal(v)}" for k, v in d.items() if v is not None]
    return "{ " + ", ".join(parts) + " }"


def serializeTriggersTOML(data):
    lines = []

    settings = data.get("settings", {})
    if settings:
        lines.append("[settings]")
        for k, v in settings.items():
            lines.append(f"{k} = {_tomlVal(v)}")
        lines.append("")

    for pose in data.get("poses", []):
        lines.append("[[poses]]")
        lines.append(f'name = "{pose["name"]}"')
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            val = pose.get(finger)
            if val is not None:
                lines.append(f"{finger} = {_tomlVal(val)}")
        lines.append("")

    for binding in data.get("bindings", []):
        lines.append("[[bindings]]")
        lines.append(f'name = "{binding["name"]}"')
        if binding.get("require_left"):
            lines.append(f'require_left = "{binding["require_left"]}"')
        if binding.get("require_right"):
            lines.append(f'require_right = "{binding["require_right"]}"')

        t  = binding["trigger"]
        td = {k: v for k, v in t.items() if v is not None}

        lines.append(f"trigger = {_inlineTable(td)}")
        lines.append("")

    return "\n".join(lines)


def serializeActionsTOML(data):
    lines = []
    for binding in data.get("bindings", []):
        lines.append("[[bindings]]")
        lines.append(f'signal = "{binding["signal"]}"')
        lines.append(f"action = {_inlineTable(binding['action'])}")
        if binding.get("on_end"):
            lines.append(f"on_end = {_inlineTable(binding['on_end'])}")
        lines.append("")
    return "\n".join(lines)


# ── Single-instance lock ───────────────────────────────────────────────────────

def _acquireLock():
    """Return True if this process is the sole running instance."""
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
            os.kill(pid, 0)   # signal 0: probe without sending
            return False      # another instance is alive
        except (ProcessLookupError, PermissionError, ValueError):
            pass              # stale lock — overwrite it
    LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(str(os.getpid()))
    return True


def _releaseLock():
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


# ── GTK/WebKit2 window ────────────────────────────────────────────────────────

def _openInWindow(port):
    """Open the config UI in a standalone GTK window using WebKit2."""
    import gi
    gi.require_version("Gtk", "3.0")
    gi.require_version("WebKit2", "4.1")
    from gi.repository import Gtk, WebKit2, GLib

    win = Gtk.Window(title="gestureControl — Config")
    win.set_default_size(1200, 760)
    win.connect("destroy", Gtk.main_quit)

    webview = WebKit2.WebView()
    settings = webview.get_settings()
    settings.set_property(
        "hardware-acceleration-policy",
        WebKit2.HardwareAccelerationPolicy.NEVER,
    )
    win.add(webview)
    win.show_all()

    def _loadWhenReady():
        import urllib.request
        url = f"http://127.0.0.1:{port}/"
        for _ in range(40):
            try:
                urllib.request.urlopen(url, timeout=0.3).close()
                webview.load_uri(url)
                return False
            except Exception:
                time.sleep(0.15)
        webview.load_uri(url)   # last-ditch attempt
        return False

    GLib.idle_add(_loadWhenReady)
    Gtk.main()


# ── Flask app ──────────────────────────────────────────────────────────────────

app             = Flask(__name__, static_folder=None)
cameraState     = CameraState()
configDir       = DEFAULT_CONFIG
useEngineStream = False   # set in main() after checking engine availability


@app.route("/")
def index():
    return send_from_directory(UI_DIR, "index.html")


@app.route("/<path:filename>")
def staticFile(filename):
    resp = send_from_directory(UI_DIR, filename)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/stream")
def videoStream():
    if useEngineStream:
        def proxyStream():
            import urllib.request
            with urllib.request.urlopen(
                f"http://127.0.0.1:{ENGINE_STREAM_PORT}/stream"
            ) as resp:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    yield chunk
        return Response(
            stream_with_context(proxyStream()),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    def generate():
        while True:
            frame = cameraState.getFrame()
            if frame:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(1 / 30)

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/state")
def sseState():
    if useEngineStream:
        def proxyState():
            import urllib.request
            with urllib.request.urlopen(
                f"http://127.0.0.1:{ENGINE_STREAM_PORT}/state"
            ) as resp:
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    yield line.decode()
        return Response(
            stream_with_context(proxyState()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    def generate():
        while True:
            data = json.dumps(cameraState.getHandState())
            yield f"data: {data}\n\n"
            time.sleep(0.1)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/config")
def getConfig():
    triggersPath = Path(configDir) / "triggers.toml"
    actionsPath  = Path(configDir) / "actions.toml"

    triggersRaw = {}
    actionsRaw  = {}
    if triggersPath.exists():
        with open(triggersPath, "rb") as f:
            triggersRaw = tomllib.load(f)
    if actionsPath.exists():
        with open(actionsPath, "rb") as f:
            actionsRaw = tomllib.load(f)

    return jsonify({
        "settings": triggersRaw.get("settings", {}),
        "poses":    triggersRaw.get("poses", []),
        "triggers": triggersRaw.get("bindings", []),
        "actions":  actionsRaw.get("bindings", []),
    })


@app.route("/api/config/triggers", methods=["POST"])
def saveTriggers():
    data    = request.get_json()
    outPath = Path(configDir) / "triggers.toml"
    payload = {
        "settings": data.get("settings", {}),
        "poses":    data.get("poses", []),
        "bindings": data.get("triggers", []),
    }
    try:
        outPath.write_text(serializeTriggersTOML(payload))
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@app.route("/api/config/actions", methods=["POST"])
def saveActions():
    data    = request.get_json()
    outPath = Path(configDir) / "actions.toml"
    payload = {"bindings": data.get("actions", [])}
    try:
        outPath.write_text(serializeActionsTOML(payload))
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


# ── Main ───────────────────────────────────────────────────────────────────────

def engineStreamAvailable():
    """Return True if the gesture engine's stream server is reachable."""
    import urllib.request
    try:
        urllib.request.urlopen(
            f"http://127.0.0.1:{ENGINE_STREAM_PORT}/state", timeout=0.5
        ).close()
        return True
    except Exception:
        return False


def main():
    global configDir, useEngineStream

    parser = argparse.ArgumentParser(description="gestureControl configuration UI")
    parser.add_argument("--port",   type=int, default=7070)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG),
                        help="Config directory (contains triggers.toml + actions.toml)")
    parser.add_argument("--input",  default=None,
                        help="Camera index or path (overrides triggers.toml)")
    parser.add_argument("--window", action="store_true",
                        help="Open in a standalone GTK window instead of the browser")
    args = parser.parse_args()

    if not _acquireLock():
        print("gestureControl-config is already running.", file=sys.stderr)
        sys.exit(0)
    atexit.register(_releaseLock)

    configDir = Path(args.config)

    if engineStreamAvailable():
        useEngineStream = True
        print(f"[camera] gesture engine detected — proxying stream from port {ENGINE_STREAM_PORT}")
    else:
        useEngineStream = False
        camInput = args.input
        if camInput is None:
            trigPath = configDir / "triggers.toml"
            if trigPath.exists():
                with open(trigPath, "rb") as f:
                    camInput = tomllib.load(f).get("settings", {}).get("camera", 0)
            else:
                camInput = 0
        stopEvent = threading.Event()
        threading.Thread(
            target=cameraThread,
            args=(camInput, configDir, cameraState, stopEvent),
            daemon=True,
        ).start()
        print("[camera] starting own camera thread")

    print(f"gestureControl-config  →  http://localhost:{args.port}")

    if args.window:
        threading.Thread(
            target=lambda: app.run(host="127.0.0.1", port=args.port, threaded=True),
            daemon=True,
        ).start()
        _openInWindow(args.port)
    else:
        def openBrowser():
            time.sleep(1.2)
            webbrowser.open(f"http://localhost:{args.port}")
        threading.Thread(target=openBrowser, daemon=True).start()
        app.run(host="127.0.0.1", port=args.port, threaded=True)


if __name__ == "__main__":
    main()
