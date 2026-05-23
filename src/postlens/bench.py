"""Latency benchmark stub (P2 fills in the implementation)."""

from __future__ import annotations

from typing import Any


def time_skill(skill: str, backbone: str) -> dict[str, Any]:
    """Time a single skill execution on a given backbone.

    Stub: P2 implements TTFT / tok-s / peak-VRAM / state-size measurements.
    """
    return {
        "skill": skill,
        "backbone": backbone,
        "ttft_s": None,
        "tok_per_s": None,
        "peak_vram_mb": None,
        "state_bytes": None,
        "status": "stub",
    }
