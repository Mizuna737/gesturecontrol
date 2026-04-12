# gestureControl

Hand gesture recognition for Linux. Point a webcam at yourself, define poses, and map them to any action — keystrokes, shell commands, media controls, or anything else you can script.

Built on [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker), communicates over D-Bus, and stays out of your way as a system tray icon.

---

## Install

```bash
paru -S gesturecontrol
```

On first launch, a setup window will appear and handle everything automatically — Python venv, pip dependencies, and the MediaPipe hand landmarker model (~26 MB). Takes about a minute.

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

---

## Configuration

The config UI (accessible from the tray) lets you define everything visually with a live camera preview:

- **Poses** — define finger states (which fingers are extended) and give them names
- **Triggers** — bind poses to events: one-shot, held continuous, chord (two hands), or sequence
- **Actions** — map trigger signals to commands, keypresses, or scaled values

Config files live at `~/.config/gesturecontrol/`:

| File | Purpose |
|---|---|
| `triggers.toml` | Poses, bindings, camera settings |
| `actions.toml` | What to do when a signal fires |

Changes are picked up live — no restart needed.

### Example: media controls

```toml
# triggers.toml
[[poses]]
name = "fist"
thumb = false
index = false
middle = false
ring = false
pinky = false

[[bindings]]
name = "play-pause"
trigger = { type = "pose", pose = "fist" }
```

```toml
# actions.toml
[[bindings]]
signal = "play-pause"
action = { type = "exec", cmd = ["playerctl", "play-pause"] }
```

---

## Architecture

Four components, all optional beyond the engine:

| Component | Role |
|---|---|
| `gestureControl.py` | Engine — reads webcam, runs MediaPipe, emits D-Bus signals |
| `gestureControl-actions.py` | Action daemon — listens on D-Bus, executes configured actions |
| `gestureControl-config.py` | Config UI — Flask server + live camera preview |
| `gestureControl-tray.py` | Tray icon — service controls + config UI launcher |

D-Bus interface: `org.gesturecontrol.Engine` at `/org/gesturecontrol`

Signals: `GestureFired`, `ContinuousUpdate`, `ContinuousEnd`, `SequenceProgress`

Any program can subscribe to these signals independently of the bundled action daemon.

---

## Dependencies

**Required** (installed automatically via pacman):
- `python >= 3.11`, `python-dbus`, `python-gobject`, `gtk3`, `webkit2gtk`, `curl`

**Pip** (installed automatically on first run):
- `mediapipe >= 0.10.30`, `opencv-python >= 4.9`, `flask >= 3.0`, `pillow >= 10.0`

**Optional**:
- `xdotool` — key action support
- `libnotify` — sequence progress notifications
- `playerctl` — media control actions

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

## License

MIT — see [LICENSE](LICENSE).
