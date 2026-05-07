#!/usr/bin/env python3
"""
gestureControl-actions.py — Companion action daemon for gestureControl.py.

Subscribes to D-Bus signals emitted by the gesture engine and executes
configured actions in response. This is the reference implementation of
the signal consumer — any program can subscribe independently.

D-Bus interface listened on: org.gesturecontrol.Engine  at  /org/gesturecontrol

Usage:
  python gestureControl-actions.py [--config PATH]

  --config  path to actions.toml (default: ~/.config/gestureControl/actions.toml)
"""

import sys
import os
_VENV = os.path.expanduser("~/.local/share/gesturecontrol/venv/bin/python3")
if sys.executable != _VENV and os.path.exists(_VENV):
    os.execv(_VENV, [_VENV] + sys.argv)

import argparse
import subprocess
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import dbus
import dbus.mainloop.glib
from gi.repository import GLib

DEFAULT_CONFIG = Path.home() / ".config" / "gesturecontrol" / "actions.toml"

DBUS_IFACE = "org.gesturecontrol.Engine"

# ── Config dataclasses ─────────────────────────────────────────────────────────

@dataclass
class ExecAction:
    """Execute a command directly via subprocess (no shell)."""
    cmd: list

@dataclass
class ExecScaledAction:
    """Render a shell template with {value} placeholder."""
    template: str

@dataclass
class KeyAction:
    """Synthesize a keypress via xdotool."""
    key: str

@dataclass
class ActionBinding:
    """Map a gesture signal to an action with optional context and onEnd handler."""
    signal: str
    action: object
    onEnd: object = None
    context: str | None = None

# ── Config loading ─────────────────────────────────────────────────────────────

def parseAction(d):
    """Build an action dataclass from a raw config dict."""
    kind = d["type"]
    if kind in ("exec", "exec_cmd"):
        return ExecAction(cmd=d["cmd"])
    if kind in ("execScaled", "exec_scaled"):
        return ExecScaledAction(template=d["template"])
    if kind == "key":
        return KeyAction(key=d["key"])
    raise ValueError(f"Unknown action type: {kind!r}")

def loadConfig(path):
    """Load actions.toml. Returns a dict mapping signal name → ActionBinding."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    bindings = {}
    for item in raw.get("bindings", []):
        action = parseAction(item["action"])
        onEndVal = item.get("onEnd", item.get("on_end"))
        onEnd = parseAction(onEndVal) if onEndVal else None
        bindings[item["signal"]] = ActionBinding(
            signal=item["signal"],
            action=action,
            onEnd=onEnd,
            context=item.get("context"),
        )
    return bindings

# ── Focus context ─────────────────────────────────────────────────────────────

_focusCache = {"wmClass": "", "ts": 0.0}

def getFocusedWindowClass():
    now = time.monotonic()
    if now - _focusCache["ts"] < 0.1:
        return _focusCache["wmClass"]
    try:
        result = subprocess.run(
            ["xdotool", "getactivewindow", "getwindowclassname"],
            capture_output=True, text=True, timeout=0.1,
        )
        _focusCache["wmClass"] = result.stdout.strip().lower()
        _focusCache["ts"] = now
    except Exception:
        pass
    return _focusCache["wmClass"]

def contextMatches(context):
    """Return True if context (WM_CLASS substring) matches the focused window."""
    if not context:
        return True
    return context.lower() in getFocusedWindowClass()


# ── Action execution ───────────────────────────────────────────────────────────

def runExec(action):
    """Fire a one-shot subprocess command."""
    subprocess.run(action.cmd, check=False)

def runExecScaled(action, value):
    """Render the template with value and run as a shell command."""
    cmd = action.template.format(value=value)
    subprocess.run(cmd, shell=True, check=False)

def runKey(action):
    """Synthesize a keypress via xdotool."""
    subprocess.run(["xdotool", "key", action.key], check=False)

def dispatchAction(action, value=None):
    """Route to the correct execution function based on action type."""
    if isinstance(action, ExecAction):
        runExec(action)
    elif isinstance(action, ExecScaledAction):
        runExecScaled(action, value)
    elif isinstance(action, KeyAction):
        runKey(action)

# ── D-Bus signal handlers ──────────────────────────────────────────────────────

def onGestureFired(name, hand, bindings):
    binding = bindings.get(str(name))
    if not binding or not contextMatches(binding.context):
        return
    print(f"[action] GestureFired     {name}  ({hand})")
    dispatchAction(binding.action)

def onContinuousUpdate(name, hand, value, bindings):
    binding = bindings.get(str(name))
    if not binding or not contextMatches(binding.context):
        return
    dispatchAction(binding.action, value=float(value))

def onContinuousEnd(name, hand, bindings):
    binding = bindings.get(str(name))
    if not binding or not binding.onEnd or not contextMatches(binding.context):
        return
    print(f"[action] ContinuousEnd    {name}  ({hand})")
    dispatchAction(binding.onEnd)

def onSequenceProgress(name, hand, step, total, bindings):
    stepStr = f"{step}/{total}"
    print(f"[info]   SequenceProgress {name}  {stepStr}  ({hand})")
    subprocess.run(
        ["notify-send", "-t", "800", "-u", "low", f"Gesture: {name}", f"Step {stepStr}"],
        check=False,
    )

# ── Main ───────────────────────────────────────────────────────────────────────

def watchConfig(configPath, bindings):
    """Poll configPath every second on the GLib main loop thread.

    On change, tries to reload. On success, mutates bindings in-place so
    existing signal-handler closures pick up the new config automatically.
    On failure, keeps the old bindings and pops a notification.
    """
    path         = Path(configPath)
    mtimeHolder  = [path.stat().st_mtime if path.exists() else 0.0]

    def check():
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return True
        if mtime == mtimeHolder[0]:
            return True
        mtimeHolder[0] = mtime
        try:
            newBindings = loadConfig(configPath)
            bindings.clear()
            bindings.update(newBindings)
            print(f"[config] Reloaded: {len(bindings)} binding(s)")
        except Exception as e:
            print(f"[config] Reload failed — keeping old config: {e}", file=sys.stderr)
            subprocess.Popen(["notify-send", "-u", "critical", "-t", "0",
                              "gestureControl-actions config error", str(e)])
        return True  # keep the timer running

    GLib.timeout_add_seconds(1, check)


def main():
    parser = argparse.ArgumentParser(description="Gesture action companion daemon")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to actions.toml")
    args = parser.parse_args()

    bindings = loadConfig(args.config)
    print(f"Loaded {len(bindings)} action binding(s) from {args.config}")

    dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
    bus = dbus.SessionBus()

    bus.add_signal_receiver(
        lambda name, hand: onGestureFired(name, hand, bindings),
        dbus_interface=DBUS_IFACE,
        signal_name="GestureFired",
    )
    bus.add_signal_receiver(
        lambda name, hand, value: onContinuousUpdate(name, hand, value, bindings),
        dbus_interface=DBUS_IFACE,
        signal_name="ContinuousUpdate",
    )
    bus.add_signal_receiver(
        lambda name, hand: onContinuousEnd(name, hand, bindings),
        dbus_interface=DBUS_IFACE,
        signal_name="ContinuousEnd",
    )
    bus.add_signal_receiver(
        lambda name, hand, step, total: onSequenceProgress(name, hand, step, total, bindings),
        dbus_interface=DBUS_IFACE,
        signal_name="SequenceProgress",
    )

    watchConfig(args.config, bindings)

    print("gestureControl-actions listening. Ctrl-C to quit.")
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
