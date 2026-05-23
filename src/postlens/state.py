"""RecurrentStateStore: skill-boundary snapshot / restore for SSM states.

v0.1.0 ships an in-memory store only. Cross-process persistence is intentionally
deferred to v0.1.1 (planned safetensors-based serialization) — `pickle` is
unsafe for arbitrary code paths and we will not ship it.

Threading model: single-threaded only. The store is process-local and not
guarded by a lock; concurrent `save` / `restore` / `evict` calls from
multiple threads or asyncio tasks are not supported in v0.1.0. v0.1.1 will
add an explicit threading.Lock when cross-process safetensors lands.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import time

from postlens.backbone import BackboneState


@dataclasses.dataclass(frozen=True)
class StateHandle:
    """Opaque handle returned by `RecurrentStateStore.save`."""

    skill_id: str
    digest: str
    created_at: float


class StateStoreError(RuntimeError):
    """Raised on save/restore failures (missing handle, bad payload, etc.)."""


class RecurrentStateStore:
    """In-process cache of recurrent states keyed by `skill_id` + per-save digest.

    Each `save()` produces a unique handle; the digest includes a monotonic
    counter so two saves under the same `skill_id` never collide (event-keyed,
    not content-addressed). Process-local: states do not persist across runs
    in v0.1.0. v0.1.1 will add safetensors-based on-disk persistence (no pickle).

    Not thread-safe — see module docstring.
    """

    def __init__(self) -> None:
        self._mem: dict[str, BackboneState] = {}
        self._counter = itertools.count()

    def _digest(self, skill_id: str, state: BackboneState) -> str:
        h = hashlib.sha256()
        h.update(skill_id.encode("utf-8"))
        h.update(state.arch.encode("utf-8"))
        h.update(state.bytes_size.to_bytes(8, "little", signed=False))
        h.update(next(self._counter).to_bytes(8, "little", signed=False))
        return h.hexdigest()[:16]

    def save(self, skill_id: str, state: BackboneState) -> StateHandle:
        if not isinstance(state, BackboneState):
            raise StateStoreError(f"expected BackboneState, got {type(state).__name__}")
        digest = self._digest(skill_id, state)
        key = f"{skill_id}::{digest}"
        self._mem[key] = state
        return StateHandle(skill_id=skill_id, digest=digest, created_at=time.time())

    def restore(self, handle: StateHandle) -> BackboneState:
        key = f"{handle.skill_id}::{handle.digest}"
        if key not in self._mem:
            raise StateStoreError(f"no state found for handle {handle!r}")
        return self._mem[key]

    def has(self, handle: StateHandle) -> bool:
        key = f"{handle.skill_id}::{handle.digest}"
        return key in self._mem

    def evict(self, handle: StateHandle) -> None:
        key = f"{handle.skill_id}::{handle.digest}"
        self._mem.pop(key, None)

    def __len__(self) -> int:
        return len(self._mem)
