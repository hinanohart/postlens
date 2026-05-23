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


def test_distinct_saves_under_same_skill_yield_distinct_handles() -> None:
    # Audit finding: same skill + same (arch, bytes_size) used to collide in
    # the digest, silently overwriting the first save. Event-keyed digest fixes this.
    store = RecurrentStateStore()
    s1 = BackboneState(arch="dummy", bytes_size=8, payload="first")
    s2 = BackboneState(arch="dummy", bytes_size=8, payload="second")
    h1 = store.save("csv_stat", s1)
    h2 = store.save("csv_stat", s2)
    assert h1.digest != h2.digest
    assert store.restore(h1).payload == "first"
    assert store.restore(h2).payload == "second"
    assert len(store) == 2


def test_save_under_same_skill_many_times_never_collides() -> None:
    store = RecurrentStateStore()
    handles = [
        store.save("x", BackboneState(arch="dummy", bytes_size=8, payload=i)) for i in range(50)
    ]
    digests = {h.digest for h in handles}
    assert len(digests) == 50
