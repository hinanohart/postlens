from __future__ import annotations

import pytest

from postlens.backbone import BackboneState
from postlens.state import RecurrentStateStore, StateStoreError


def _state(arch: str = "dummy", n: int = 0) -> BackboneState:
    return BackboneState(arch=arch, bytes_size=8 + n, payload=n)


def test_save_then_restore_round_trip() -> None:
    store = RecurrentStateStore()
    h = store.save("csv_stat", _state(n=1))
    got = store.restore(h)
    assert got.payload == 1


def test_restore_missing_handle_raises() -> None:
    store = RecurrentStateStore()
    h = store.save("a", _state(n=1))
    store.evict(h)
    with pytest.raises(StateStoreError):
        store.restore(h)


def test_save_rejects_non_backbone_state() -> None:
    store = RecurrentStateStore()
    with pytest.raises(StateStoreError):
        store.save("x", object())  # type: ignore[arg-type]


def test_has_reports_membership() -> None:
    store = RecurrentStateStore()
    h = store.save("a", _state(n=2))
    assert store.has(h)
    store.evict(h)
    assert not store.has(h)


def test_distinct_skills_get_distinct_keys() -> None:
    store = RecurrentStateStore()
    h1 = store.save("a", _state(n=5))
    h2 = store.save("b", _state(n=5))
    assert h1.skill_id != h2.skill_id
    assert len(store) == 2


def test_len_after_evict() -> None:
    store = RecurrentStateStore()
    h1 = store.save("a", _state(n=1))
    h2 = store.save("b", _state(n=2))
    assert len(store) == 2
    store.evict(h1)
    assert len(store) == 1
    assert store.has(h2)
