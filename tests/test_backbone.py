from __future__ import annotations

import pytest

from postlens.backbone import (
    REGISTERED_ARCH,
    RWKV7_GOOSE_REVISION,
    BackboneState,
    DummyBackbone,
    make_backbone,
)


def test_rwkv7_revision_is_40_hex_sha() -> None:
    assert isinstance(RWKV7_GOOSE_REVISION, str)
    assert len(RWKV7_GOOSE_REVISION) == 40
    int(RWKV7_GOOSE_REVISION, 16)  # raises ValueError if not hex


def test_registered_arch_contains_rwkv_and_mamba() -> None:
    assert "rwkv7-goose" in REGISTERED_ARCH
    assert REGISTERED_ARCH["rwkv7-goose"].startswith("RWKV/RWKV7-Goose")
    assert "mamba-2.8b" in REGISTERED_ARCH


def test_dummy_backbone_prefill_then_step_is_deterministic() -> None:
    bb = DummyBackbone(vocab_size=32)
    state = bb.prefill([1, 2, 3, 4])
    assert isinstance(state, BackboneState)
    assert state.arch == "dummy"
    new_state, tok = bb.step(state, 5)
    new_state2, tok2 = bb.step(state, 5)
    assert tok == tok2 == 6
    assert new_state.payload == new_state2.payload


def test_dummy_backbone_state_bytes_positive() -> None:
    bb = DummyBackbone()
    state = bb.prefill([0])
    assert bb.state_bytes(state) > 0


def test_make_backbone_unknown_arch_raises() -> None:
    with pytest.raises(ValueError):
        make_backbone("not-a-real-arch")


def test_make_backbone_dummy_returns_dummy() -> None:
    bb = make_backbone("dummy")
    assert bb.arch == "dummy"
    assert isinstance(bb, DummyBackbone)


def test_dummy_step_wraps_vocab() -> None:
    bb = DummyBackbone(vocab_size=10)
    state = bb.prefill([0])
    _, tok = bb.step(state, 9)
    assert tok == 0
