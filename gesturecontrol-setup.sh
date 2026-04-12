#!/usr/bin/env bash
# gesturecontrol-setup — Post-install setup for gesturecontrol.
#
# Creates a per-user Python venv, installs pip dependencies, and downloads
# the MediaPipe hand landmarker model.
#
# Run once after installing the package:
#   gesturecontrol-setup

set -euo pipefail

DATA_DIR="$HOME/.local/share/gesturecontrol"
VENV_DIR="$DATA_DIR/venv"
MODEL_FILE="$DATA_DIR/hand_landmarker.task"
CONFIG_DIR="$HOME/.config/gesturecontrol"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

echo "=== gesturecontrol setup ==="
echo ""

mkdir -p "$DATA_DIR" "$CONFIG_DIR"

# ── Python venv ────────────────────────────────────────────────────────────────
echo "[1/3] Python venv..."

if [[ -x "$VENV_DIR/bin/python3" ]]; then
    echo "      Already exists: $VENV_DIR"
else
    echo "      Creating: $VENV_DIR"
    # --system-site-packages exposes python-dbus and python-gobject from pacman
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

# ── pip dependencies ───────────────────────────────────────────────────────────
echo "[2/3] Installing pip dependencies (mediapipe, opencv, flask, pillow)..."

"$VENV_DIR/bin/pip" install --quiet \
    "mediapipe>=0.10.30" \
    "opencv-python>=4.9" \
    "flask>=3.0" \
    "pillow>=10.0"

# ── MediaPipe model ────────────────────────────────────────────────────────────
echo "[3/3] Hand landmarker model..."

if [[ -f "$MODEL_FILE" ]]; then
    echo "      Already exists: $MODEL_FILE"
else
    echo "      Downloading (~26 MB) from Google..."
    curl -L --progress-bar "$MODEL_URL" -o "$MODEL_FILE"
    echo "      Saved: $MODEL_FILE"
fi

# ── Example configs ────────────────────────────────────────────────────────────
if [[ ! -f "$CONFIG_DIR/triggers.toml" ]]; then
    echo ""
    echo "Installing example configs to $CONFIG_DIR ..."
    cp /usr/share/doc/gesturecontrol/triggers.toml.example "$CONFIG_DIR/triggers.toml"
    cp /usr/share/doc/gesturecontrol/actions.toml.example  "$CONFIG_DIR/actions.toml"
fi

echo ""
echo "=== Setup complete ==="
echo ""
echo "  Venv:    $VENV_DIR"
echo "  Model:   $MODEL_FILE"
echo "  Config:  $CONFIG_DIR/"
echo ""
echo "Enable and start services:"
echo "  systemctl --user enable --now gestureControl.service"
echo "  systemctl --user enable --now gestureControl-actions.service"
echo ""
echo "Or launch the tray icon directly:"
echo "  gesturecontrol-tray"
echo ""
