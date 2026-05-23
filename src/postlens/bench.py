"""Latency benchmark: TTFT / tok-s / peak-VRAM / state-size."""

from __future__ import annotations

import dataclasses
import time
from typing import Any

from postlens.backbone import Backbone


@dataclasses.dataclass(frozen=True)
class BenchResult:
    """Single benchmark observation. `None` fields are not-measured."""

    skill: str
    backbone: str
    ttft_s: float | None
    tok_per_s: float | None
    peak_vram_mb: float | None
    state_bytes: int | None
    decoded: int
    status: str

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def time_skill(skill: str, backbone: str | Backbone, **kwargs: Any) -> dict[str, Any]:
    """Functional shim used by SKILL.md `Latency probe` lines.

    If `backbone` is a string, this is the lightweight stub path used in
    documentation snippets (no real backbone instantiated). If `backbone` is
    a concrete `Backbone`, a real `LatencyBench.run` is performed.
    """
    if isinstance(backbone, str):
        return BenchResult(
            skill=skill,
            backbone=backbone,
            ttft_s=None,
            tok_per_s=None,
            peak_vram_mb=None,
            state_bytes=None,
            decoded=0,
            status="stub",
        ).as_dict()
    return LatencyBench(backbone).run(skill=skill, **kwargs).as_dict()


class LatencyBench:
    """Latency benchmark harness driving a concrete `Backbone`."""

    def __init__(self, backbone: Backbone) -> None:
        self.backbone = backbone

    def run(
        self,
        skill: str,
        prompt_tokens: list[int] | None = None,
        decode_tokens: int = 32,
    ) -> BenchResult:
        if decode_tokens < 0:
            raise ValueError("decode_tokens must be >= 0")
        prompt_tokens = prompt_tokens or [0]
        _reset_peak_vram()
        t0 = time.perf_counter()
        state = self.backbone.prefill(prompt_tokens)
        ttft = time.perf_counter() - t0
        last_tok = prompt_tokens[-1]
        decoded = 0
        decode_start = time.perf_counter()
        for _ in range(decode_tokens):
            state, last_tok = self.backbone.step(state, last_tok)
            decoded += 1
        decode_elapsed = time.perf_counter() - decode_start
        tok_per_s = decoded / decode_elapsed if decoded > 0 and decode_elapsed > 0 else None
        peak_vram_mb = _peak_vram_mb()
        return BenchResult(
            skill=skill,
            backbone=self.backbone.arch,
            ttft_s=ttft,
            tok_per_s=tok_per_s,
            peak_vram_mb=peak_vram_mb,
            state_bytes=self.backbone.state_bytes(state),
            decoded=decoded,
            status="ok",
        )


def _peak_vram_mb() -> float | None:
    """Return CUDA peak-allocated VRAM in MiB, or `None` if CUDA is absent."""
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _reset_peak_vram() -> None:
    """Reset CUDA peak-stats so `_peak_vram_mb()` reports per-run, not cumulative."""
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()
