#!/usr/bin/env python3
"""
gestureControl-tray.py — System tray icon for the gestureControl engine.

Monitors the engine service, provides start/stop/restart controls,
and launches the configuration UI on demand.

Usage:
  python gestureControl-tray.py
"""

import sys
import os

_VENV = os.path.expanduser("~/.local/share/gesturecontrol/venv/bin/python3")

# ── Re-exec through venv if it's ready ────────────────────────────────────────
if sys.executable != _VENV and os.path.exists(_VENV):
    os.execv(_VENV, [_VENV] + sys.argv)

# ── First-run setup (system Python only — venv packages not yet available) ────
if not os.path.exists(_VENV):
    import subprocess
    import threading

    import gi
    gi.require_version("Gtk", "3.0")
    from gi.repository import Gtk, GLib

    def _firstRunSetup():
        win = Gtk.Window(title="Gesture Control — First Run")
        win.set_default_size(460, 160)
        win.set_resizable(False)
        win.set_position(Gtk.WindowPosition.CENTER)
        win.set_deletable(False)

        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        box.set_margin_top(20)
        box.set_margin_bottom(20)
        box.set_margin_start(20)
        box.set_margin_end(20)
        win.add(box)

        titleLabel = Gtk.Label()
        titleLabel.set_markup("<b>Setting up Gesture Control…</b>")
        titleLabel.set_halign(Gtk.Align.START)
        box.pack_start(titleLabel, False, False, 0)

        statusLabel = Gtk.Label(label="Starting…")
        statusLabel.set_halign(Gtk.Align.START)
        statusLabel.set_ellipsize(3)   # PANGO_ELLIPSIZE_END
        box.pack_start(statusLabel, False, False, 0)

        bar = Gtk.ProgressBar()
        bar.set_pulse_step(0.04)
        box.pack_start(bar, False, False, 0)

        win.show_all()

        pulseId  = [GLib.timeout_add(80, lambda: bar.pulse() or True)]
        succeeded = [False]

        def _worker():
            proc = subprocess.Popen(
                ["/usr/bin/gesturecontrol-setup"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in proc.stdout:
                text = line.strip()
                if text:
                    GLib.idle_add(statusLabel.set_text, text[:72])
            proc.wait()

            GLib.source_remove(pulseId[0])
            if proc.returncode == 0:
                succeeded[0] = True
                GLib.idle_add(bar.set_fraction, 1.0)
                GLib.idle_add(statusLabel.set_text, "Setup complete!")
                GLib.timeout_add(1200, Gtk.main_quit)
            else:
                GLib.idle_add(
                    statusLabel.set_markup,
                    '<span foreground="#e05050">'
                    "Setup failed — run gesturecontrol-setup in a terminal."
                    "</span>",
                )
                GLib.idle_add(win.set_deletable, True)
                GLib.idle_add(win.connect, "destroy", Gtk.main_quit)

        threading.Thread(target=_worker, daemon=True).start()
        Gtk.main()
        return succeeded[0]

    if not _firstRunSetup():
        sys.exit(1)

    # Venv now exists — re-exec under it so venv-only packages are available
    os.execv(_VENV, [_VENV] + sys.argv)

# ── Normal imports (guaranteed to be running under venv from here) ─────────────
import subprocess
import threading
import time
from pathlib import Path

import gi
gi.require_version("Gtk", "3.0")
from gi.repository import Gtk, GLib, GdkPixbuf

from PIL import Image, ImageDraw


# ── Constants ──────────────────────────────────────────────────────────────────

SERVICE_NAME  = "gestureControl"
CONFIG_PORT   = 7070
POLL_INTERVAL = 3   # seconds

SCRIPTS_DIR   = Path(__file__).parent
CONFIG_SCRIPT = SCRIPTS_DIR / "gestureControl-config.py"
VENV_PYTHON   = Path(_VENV)


# ── Icon drawing ───────────────────────────────────────────────────────────────

def _drawHand(draw, size, color):
    """Draw a simplified open-hand silhouette into `draw` (PIL ImageDraw)."""
    palmL = size * 0.18
    palmR = size * 0.82
    palmT = size * 0.52
    palmB = size * 0.92
    pr    = max(2, int(size * 0.08))
    draw.rounded_rectangle([palmL, palmT, palmR, palmB], radius=pr, fill=color)

    p        = max(2, size // 16)
    fingerW  = (palmR - palmL - p * 4) / 5
    heights  = [0.24, 0.10, 0.07, 0.12, 0.28]
    for i, topFrac in enumerate(heights):
        x0 = palmL + i * (fingerW + p)
        x1 = x0 + fingerW
        y0 = size * topFrac
        y1 = palmT + p
        fr = max(2, int(fingerW * 0.45))
        draw.rounded_rectangle([x0, y0, x1, y1], radius=fr, fill=color)


def makePixbuf(active):
    """Return a GdkPixbuf for the tray icon."""
    size  = 64
    img   = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    color = (0, 230, 118, 255) if active else (110, 120, 150, 255)
    _drawHand(ImageDraw.Draw(img), size, color)
    data = GLib.Bytes.new(img.tobytes())
    return GdkPixbuf.Pixbuf.new_from_bytes(
        data, GdkPixbuf.Colorspace.RGB, True, 8, size, size, size * 4
    )


# ── Service helpers ────────────────────────────────────────────────────────────

def serviceActive():
    r = subprocess.run(
        ["systemctl", "--user", "is-active", SERVICE_NAME],
        capture_output=True, text=True,
    )
    return r.stdout.strip() == "active"


def serviceEnabled():
    r = subprocess.run(
        ["systemctl", "--user", "is-enabled", SERVICE_NAME],
        capture_output=True, text=True,
    )
    return r.stdout.strip() in ("enabled", "enabled-runtime")


def serviceCtl(*args):
    subprocess.Popen(["systemctl", "--user", *args, SERVICE_NAME])


# ── Config UI ─────────────────────────────────────────────────────────────────

_configProc = None

def openConfigUI():
    global _configProc
    if _configProc is None or _configProc.poll() is not None:
        _configProc = subprocess.Popen(
            [str(VENV_PYTHON), str(CONFIG_SCRIPT), "--window"],
            start_new_session=True,
        )


# ── Tray app ───────────────────────────────────────────────────────────────────

class TrayApp:
    def __init__(self):
        self._active  = False
        self._enabled = False

        self._statusIcon = Gtk.StatusIcon()
        self._statusIcon.set_from_pixbuf(makePixbuf(False))
        self._statusIcon.set_tooltip_text("gestureControl  ○  Stopped")
        self._statusIcon.set_visible(True)
        self._statusIcon.connect("popup-menu", self._onPopupMenu)
        self._statusIcon.connect("activate",   self._onActivate)

    # ── Menu ───────────────────────────────────────────────────────────────────

    def _buildMenu(self):
        menu = Gtk.Menu()

        label = "gestureControl  ●  Running" if self._active else "gestureControl  ○  Stopped"
        header = Gtk.MenuItem(label=label)
        header.set_sensitive(False)
        menu.append(header)

        menu.append(Gtk.SeparatorMenuItem())

        configItem = Gtk.MenuItem(label="Open Config UI")
        configItem.connect("activate", lambda _: threading.Thread(target=openConfigUI, daemon=True).start())
        menu.append(configItem)

        menu.append(Gtk.SeparatorMenuItem())

        startItem = Gtk.MenuItem(label="Start Engine")
        startItem.set_sensitive(not self._active)
        startItem.connect("activate", self._onStart)
        menu.append(startItem)

        stopItem = Gtk.MenuItem(label="Stop Engine")
        stopItem.set_sensitive(self._active)
        stopItem.connect("activate", self._onStop)
        menu.append(stopItem)

        restartItem = Gtk.MenuItem(label="Restart Engine")
        restartItem.set_sensitive(self._active)
        restartItem.connect("activate", self._onRestart)
        menu.append(restartItem)

        menu.append(Gtk.SeparatorMenuItem())

        loginItem = Gtk.CheckMenuItem(label="Start on Login")
        loginItem.set_active(self._enabled)
        loginItem.connect("toggled", self._onToggleEnabled)
        menu.append(loginItem)

        menu.append(Gtk.SeparatorMenuItem())

        quitItem = Gtk.MenuItem(label="Quit")
        quitItem.connect("activate", self._onQuit)
        menu.append(quitItem)

        menu.show_all()
        return menu

    # ── Signal handlers ────────────────────────────────────────────────────────

    def _onPopupMenu(self, icon, button, activateTime):
        menu = self._buildMenu()
        menu.popup(None, None, Gtk.StatusIcon.position_menu, icon, button, activateTime)

    def _onActivate(self, icon):
        threading.Thread(target=openConfigUI, daemon=True).start()

    def _onStart(self, _):
        serviceCtl("start")
        GLib.timeout_add(1500, self._refreshStatus)

    def _onStop(self, _):
        serviceCtl("stop")
        GLib.timeout_add(1500, self._refreshStatus)

    def _onRestart(self, _):
        serviceCtl("restart")
        GLib.timeout_add(1500, self._refreshStatus)

    def _onToggleEnabled(self, item):
        if item.get_active():
            serviceCtl("enable")
        else:
            serviceCtl("disable")

    def _onQuit(self, _):
        Gtk.main_quit()

    # ── Status polling ─────────────────────────────────────────────────────────

    def _refreshStatus(self):
        """Called on a background thread; schedules GTK updates via GLib.idle_add."""
        active  = serviceActive()
        enabled = serviceEnabled()
        GLib.idle_add(self._applyStatus, active, enabled)
        return False   # don't repeat (GLib.timeout_add callback)

    def _applyStatus(self, active, enabled):
        """Must run on the GTK main thread."""
        self._active  = active
        self._enabled = enabled
        self._statusIcon.set_from_pixbuf(makePixbuf(active))
        self._statusIcon.set_tooltip_text(
            f"gestureControl  {'●' if active else '○'}  {'Running' if active else 'Stopped'}"
        )
        return False   # don't repeat (GLib.idle_add callback)

    def _pollWorker(self):
        """Background thread that periodically refreshes status."""
        while True:
            self._refreshStatus()
            time.sleep(POLL_INTERVAL)

    # ── Run ────────────────────────────────────────────────────────────────────

    def run(self):
        self._refreshStatus()
        threading.Thread(target=self._pollWorker, daemon=True).start()
        Gtk.main()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    TrayApp().run()
