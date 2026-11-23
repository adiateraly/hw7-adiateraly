"""Shared constants, data types and small helpers."""
from __future__ import annotations

import json
import platform
import sys
from collections import namedtuple
from pathlib import Path

END = "END"          # terminal marker: zero continuation value, never a playable state
STICK, HIT = 0, 1    # Gymnasium action labels
ACTION_NAME = {STICK: "stick", HIT: "hit"}
ACTIONS = (STICK, HIT)

GAMMA = 1.0
ALPHA = 0.1
SAFETY_LIMIT = 100   # max player actions per game

# One step of experience. next_state is END on a terminal transition.
Transition = namedtuple("Transition", "state action reward next_state")


def baseline_action(player_sum: int) -> int:
    """Fixed baseline policy: hit below 20, stick at 20 or 21."""
    return HIT if player_sum < 20 else STICK


def state_key(obs) -> tuple:
    """Gymnasium observation -> hashable (player_sum, dealer_showing, usable_ace)."""
    return (int(obs[0]), int(obs[1]), int(bool(obs[2])))


def fmt(x: float, nd: int = 6) -> str:
    return f"{x:.{nd}f}"


def write_versions(path: Path) -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("gymnasium", "numpy", "matplotlib"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # pragma: no cover
            info[mod] = "not installed"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n")
    return info
