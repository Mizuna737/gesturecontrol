# gestureControl

Hand gesture recognition for Linux. Point a webcam at yourself, define poses, and map them to any action — keystrokes, shell commands, media controls, or anything else you can script.

Built on custom ONNX hand-pose models (palm detection + landmark regression), communicates over D-Bus, and stays out of your way as a system tray icon.

---

## Features

- **Custom ONNX hand tracking** — palm detection + 21-landmark regression pipeline, GPU-accelerated via CUDA when available
- **6 trigger types** — pose, swipe, sequence, chord, continuous, and sequenced-continuous (pose-gated continuous)
- **Context-aware actions** — optionally scope bindings to a specific focused window's WM_CLASS
- **Presence detection** — auto-blank screen after idle, wake on motion/pose detection; pauses hand tracking while screen is off
- **IR camera support** — adaptive dark-frame detection for cameras that interleave calibration frames
- **Hot-reload config** — edit `triggers.toml` and `actions.toml` live, no restart needed
- **Local stream server** — MJPEG feed + SSE hand-state feed for the config UI (zero extra camera overhead)
- **Systemd user services** — engine and action daemon managed as services, tray icon for control
- **Config UI** — web-based editor with live camera preview, pose detection overlay, and camera device picker

---

## Install

Use the AUR helper of your choice:

```bash
paru -S gesturecontrol
```

On first launch, the tray icon triggers a setup wizard — Python venv, pip dependencies, and the ONNX hand-pose models (~30 MB). Takes about a minute.

---

## Usage

Launch **Gesture Control** from your app launcher, or run:

```bash
gesturecontrol-tray &
```

The tray icon gives you start/stop/restart controls and opens the config UI on click. The engine and action daemon are managed as systemd user services:

```bash
systemctl --user enable --now gestureControl.service
systemctl --user enable --now gestureControl-actions.service
```

The tray icon autostarts on login via XDG autostart.

### Command-line flags

| Script                   | Flags                                                                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| `gestureControl`         | `--input DEVICE` override camera, `--config PATH` custom triggers.toml, `--debug` OpenCV overlay, `--stream-port N` / `--no-stream` |
| `gestureControl-config`  | `--port N` server port, `--config DIR` config directory, `--input DEVICE`, `--window` open in GTK window                            |
| `gestureControl-actions` | `--config PATH` custom actions.toml                                                                                                 |
| `gestureControl-tray`    | no flags                                                                                                                            |

---

## Configuration

The config UI (accessible from the tray) lets you define everything visually with a live camera preview:

- **Poses** — define finger states (which fingers are extended) and give them names
- **Triggers** — bind poses to events: one-shot, held continuous, chord (two hands), sequence, swipe, or sequenced-continuous
- **Actions** — map trigger signals to commands, keypresses, or scaled values; optionally scoped to a specific focused window

Config files live at `~/.config/gesturecontrol/`:

| File            | Purpose                                           |
| --------------- | ------------------------------------------------- |
| `triggers.toml` | Poses, trigger bindings, camera/presence settings |
| `actions.toml`  | What to do when a gesture signal fires            |

Changes are picked up live — no restart needed.

### Trigger types

```toml
# pose — hold a finger configuration for dwellMs
[[bindings]]
name = "fist_toggle"
trigger = { type = "pose", hand = "right", shape = "FIST", dwellMs = 200 }

# swipe — hand moves in a direction beyond min_displacement
[[bindings]]
name = "browser_back"
trigger = { type = "swipe", hand = "right", direction = "left", minDisplacement = 0.3 }

# sequence — ordered poses held in a window
[[bindings]]
name = "play_pause"
trigger = { type = "sequence", hand = "left", steps = ["FIST", "FIVE"], windowMs = 3000, stepDwellMs = 1000 }

# chord — two hands hold specific poses simultaneously
[[bindings]]
name = "chord_action"
trigger = { type = "chord", left = "FIST", right = "FIST", dwellMs = 500 }

# continuous — emit live values while conditions hold
[[bindings]]
name = "set_volume"
trigger = { type = "continuous", hand = "left", metric = "angle", range = [0, 1] }

# sequenced-continuous — require a prefix sequence before continuous phase
[[bindings]]
name = "zoom_control"
trigger = {
  type = "sequencedContinuous", hand = "right",
  prefixSteps = ["THUMBS_UP"], prefixWindowMs = 1500, prefixDwellMs = 200,
  metric = "pinchDistance", range = [0.05, 0.3], hysteresis = 0.04
}
```

Trigger fields:

| Field             | Type                                                                                | Description                                                                        |
| ----------------- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `hand`            | `"left"`, `"right"`, `"either"`                                                     | Which hand to observe                                                              |
| `shape`           | string                                                                              | Pose name (pose trigger)                                                           |
| `direction`       | `"LEFT_SWIPE"` or `"RIGHT_SWIPE"` (set automatically from `direction` config value) | Swipe direction                                                                    |
| `steps`           | string[]                                                                            | Ordered pose names (sequence)                                                      |
| `windowMs`        | int                                                                                 | Time window for all steps to complete                                              |
| `stepDwellMs`     | int                                                                                 | Minimum hold time per step                                                         |
| `metric`          | string                                                                              | Continuous metric: `pinchDistance`, `handHeight`, `handX`, `fingerSpread`, `angle` |
| `range`           | [float, float]                                                                      | Map raw metric values to [0.0, 1.0]                                                |
| `minDisplacement` | float                                                                               | Fraction of frame width for swipe detection                                        |
| `dwellMs`         | int                                                                                 | Hold duration before firing (pose/chord)                                           |
| `hysteresis`      | float                                                                               | Slot-transition buffer for continuous (default: 0.04)                              |

### Pose definitions with spread constraints

```toml
# Poses can include spread constraints between adjacent fingers
[[poses]]
name = "OPEN_PALM"
thumb = true
index = true
middle = true
ring = true
pinky = true
spreadThumbIndex = "apart"
spreadIndexMiddle = "apart"
spreadMiddleRing = "apart"
spreadRingPinky = "apart"

# Custom numeric thresholds (float)
[[poses]]
name = "spread_victory"
thumb = true
index = true
middle = true
ring = false
pinky = false
spreadThumbIndex = 0.5
spreadIndexMiddle = 0.3
```

Spread constraint values: `"close"` (below threshold), `"apart"` (above threshold), or a float (custom minimum gap).

### Modifier bindings (require)

A binding can require one or more poses to be held before it fires. This is useful for "modifier" gestures:

```toml
# Only swipe when left hand holds THREE
[[bindings]]
name = "browser_back"
require = [{ hand = "left", pose = "THREE" }]
trigger = { type = "swipe", hand = "right", direction = "left" }
```

### Swipe triggers

```toml
[[bindings]]
name = "browser_back"
trigger = { type = "swipe", hand = "right", direction = "left", minDisplacement = 0.3 }
```

Valid directions: `left`, `right`, `up`, `down`. `min_displacement` is a fraction of the frame width.

### Context-aware actions

Action bindings accept an optional `context` field — a WM_CLASS substring. When set, the action only fires if that string appears in the focused window's class name:

```toml
# actions.toml
[[bindings]]
signal = "browser_back"
context = "qutebrowser"
action = { type = "key", key = "shift+h" }
```

Requires `xdotool` (listed under optional dependencies).

### Presence detection

Control idle screen-blanking and wake behavior:

```toml
[presence]
enabled = true
idleSeconds = 300          # blank screen after 5 min idle
checkHz = 2                # presence check frequency
motionThreshold = 5.0      # frame-diff sensitivity
poseDetection = false      # use pose landmark model for presence
poseCheckMode = "fallback" # "always" or "fallback" (motion only first)
useMotionDetection = true  # enable frame-diff presence check
pauseHandsWhenAbsent = true# release hand landmarker while screen is blanked
```

### Camera settings

```toml
[settings]
camera = 0                 # device index or /dev/video path
fps = 30                   # inference rate (0 = every frame)
width = 640
height = 480
format = "GREY"            # V4L2 format: "MJPG", "YUYV", "GREY", etc.
spreadThreshold = 0.20     # finger spread classify threshold
dwellMs = 200              # default dwell time for pose/chord triggers
gracePeriodMs = 200        # hold detected pose briefly after it disappears
```

### Continuous metrics

| Metric                             | Description                                                    |
| ---------------------------------- | -------------------------------------------------------------- |
| `pinchDistance` / `pinch_distance` | Distance between thumb tip and index tip                       |
| `handHeight` / `hand_height`       | Normalized Y position (inverted — raising hand = higher value) |
| `handX` / `hand_x`                 | Normalized X position                                          |
| `fingerSpread` / `finger_spread`   | Max - min of finger tip X coordinates                          |
| `angle`                            | Wrist-to-middle-finger angle mapped to [0, 1]                  |

---

## Actions

Actions are defined in `actions.toml`. Three action types:

```toml
# Execute a command directly
[[bindings]]
signal = "play_pause"
action = { type = "exec", cmd = ["playerctl", "play-pause"] }

# Shell command with {value} template (continuous triggers)
[[bindings]]
signal = "set_volume"
action = { type = "execScaled", template = "pactl set-sink-volume @DEFAULT_SINK@ {value}" }

# Key press via xdotool
[[bindings]]
signal = "browser_back"
action = { type = "key", key = "shift+h" }
```

Actions can have an `onEnd` handler that fires when a continuous trigger ends:

```toml
[[bindings]]
signal = "zoom_control"
action = { type = "execScaled", template = "wmctrl -r :ACTIVE: -e 0,0,0,{value},1024" }
onEnd = { type = "exec", cmd = ["notify-send", "Zoom ended"] }
```

---

## Architecture

Four components, all optional beyond the engine:

| Component                   | Role                                                                      |
| --------------------------- | ------------------------------------------------------------------------- |
| `gestureControl.py`         | Engine — reads webcam, runs ONNX hand-pose models, emits D-Bus signals    |
| `gestureControl-actions.py` | Action daemon — listens on D-Bus, executes configured actions             |
| `gestureControl-config.py`  | Config UI — Flask server + live camera preview (or proxies engine stream) |
| `gestureControl-tray.py`    | Tray icon — service controls + config UI launcher + first-run setup       |

D-Bus interface: `org.gesturecontrol.Engine` at `/org/gesturecontrol`

Signals emitted:

| Signal             | Signature | Description                                                   |
| ------------------ | --------- | ------------------------------------------------------------- |
| `GestureFired`     | `(ss)`    | One-shot trigger fired: binding name, hand                    |
| `ContinuousStart`  | `(ss)`    | Continuous phase began                                        |
| `ContinuousUpdate` | `(ssd)`   | Live value update: binding name, hand, normalized value       |
| `ContinuousEnd`    | `(ss)`    | Continuous phase ended                                        |
| `SequenceProgress` | `(ssii)`  | Sequence step completed: binding name, hand, step, total      |
| `RegisterSlots`    | `(si)`    | Method call to configure slot mapping for continuous triggers |

Any program can subscribe to these signals independently of the bundled action daemon.

---

## Dependencies

**Required** (installed automatically via pacman):

- `python >= 3.11`, `python-dbus`, `python-gobject`, `gtk3`, `webkit2gtk`, `curl`

**Pip** (installed automatically on first run):

- `mediapipe >= 0.10.30`, `opencv-python >= 4.9`, `flask >= 3.0`, `pillow >= 10.0`

**Optional**:

- `xdotool` — key action support and context-aware window matching
- `libnotify` — sequence progress notifications
- `playerctl` — media control actions
- `v4l2-utils` — camera device naming in config UI

---

## Development

Clone and edit directly:

```bash
git clone https://github.com/Mizuna737/gesturecontrol
cd gesturecontrol
```

To test changes without reinstalling, point systemd at your local copy:

```bash
# Override the installed service temporarily
systemctl --user edit gestureControl.service
# Add:
# [Service]
# ExecStart=
# ExecStart=%h/.local/share/gesturecontrol/venv/bin/python3 /path/to/your/gestureControl.py
```

### Project structure

| File / Directory            | Purpose                                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------------- |
| `gestureControl.py`         | Engine — ONNX hand-pose detection, trigger matching, D-Bus signals, stream server                 |
| `gestureControl-actions.py` | Action daemon — D-Bus signal consumer, command/keypress execution                                 |
| `gestureControl-config.py`  | Config UI backend — Flask server, TOML I/O, camera thread                                         |
| `gestureControl-tray.py`    | Tray icon — GTK app, service management, first-run setup                                          |
| `gestureControl-config-ui/` | Frontend — `index.html`, `app.js`, `style.css`                                                    |
| `poseUtils.py`              | Shared pose utilities — finger states, spread computation, dark-frame detection, landmark drawing |
| `systemd/`                  | Systemd user service units                                                                        |
| `config/`                   | Example `triggers.toml` and `actions.toml`                                                        |

### Contributing

Pull requests are welcome. For significant changes, open an issue first to discuss what you'd like to change.

The four main scripts are self-contained and well-commented — most contributions will touch one of them plus the config UI in `gestureControl-config-ui/`. The D-Bus interface (`DBUS_IFACE` in the engine and action daemon) is the stable contract between components; changes there should be discussed before breaking it.

When submitting a PR:

- Keep changes focused — one feature or fix per request
- Test with a real webcam before submitting
- If you're adding a new trigger type or action type, include an example in `config/`

---

## Support

If gestureControl is useful to you, you can support development here:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/mizuna737)

---

## AI Disclosure

This project originated with me, the first implementation and all of the ideation is mine. Documentation, packaging for github, and much of the CUDA specific stuff was co-authored by Qwen 3.6 and Claude. If that turns you off, sorry! I want to code, not write documentation or fiddle with CUDA packages.

---

## License

MIT — see [LICENSE](LICENSE).
