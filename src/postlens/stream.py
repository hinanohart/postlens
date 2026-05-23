"""Chunk-prefill + interleaved scheduler for skill execution.

`InterleavedScheduler` carves a long prefill into chunks so step-decoding of
skill A can overlap with prefill of skill B on a separate CUDA stream when
available. The fallback path executes serially on CPU and still returns the
same trace shape, which keeps unit tests CUDA-free.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable

ChunkFn = Callable[[list[int]], None]


@dataclasses.dataclass(frozen=True)
class ScheduledOp:
    """Trace entry emitted by the scheduler (used by tests + bench)."""

    kind: str  # "prefill" | "step" | "yield"
    skill_id: str
    n_tokens: int
    stream_idx: int


class InterleavedScheduler:
    """Round-robin scheduler that splits prefill into chunks and interleaves
    them with caller-supplied step ops.

    The scheduler is pure logic — it doesn't import torch, doesn't touch
    real CUDA streams. Concrete callers pass `prefill_fn` / `step_fn`
    callbacks that actually drive the backbone.
    """

    def __init__(self, chunk_size: int = 256, n_streams: int = 2) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if n_streams <= 0:
            raise ValueError("n_streams must be > 0")
        self.chunk_size = chunk_size
        self.n_streams = n_streams
        self._cursor = 0

    def plan(self, skill_id: str, prefill_tokens: int, step_tokens: int) -> list[ScheduledOp]:
        """Return the deterministic trace for a single skill run."""
        ops: list[ScheduledOp] = []
        remaining = max(prefill_tokens, 0)
        while remaining > 0:
            n = min(self.chunk_size, remaining)
            ops.append(
                ScheduledOp(
                    kind="prefill",
                    skill_id=skill_id,
                    n_tokens=n,
                    stream_idx=self._next_stream(),
                )
            )
            remaining -= n
        for _ in range(max(step_tokens, 0)):
            ops.append(
                ScheduledOp(
                    kind="step",
                    skill_id=skill_id,
                    n_tokens=1,
                    stream_idx=self._next_stream(),
                )
            )
        ops.append(ScheduledOp(kind="yield", skill_id=skill_id, n_tokens=0, stream_idx=0))
        return ops

    def _next_stream(self) -> int:
        idx = self._cursor % self.n_streams
        self._cursor += 1
        return idx

    @staticmethod
    def merge_plans(plans: Iterable[list[ScheduledOp]]) -> list[ScheduledOp]:
        """Interleave per-skill plans round-robin on op index."""
        plans_list = [list(p) for p in plans]
        out: list[ScheduledOp] = []
        positions = [0] * len(plans_list)
        active = list(range(len(plans_list)))
        while active:
            for i in list(active):
                pos = positions[i]
                if pos >= len(plans_list[i]):
                    active.remove(i)
                    continue
                out.append(plans_list[i][pos])
                positions[i] = pos + 1
        return out
