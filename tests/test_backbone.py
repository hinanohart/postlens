from __future__ import annotations

import pytest

from postlens.backbone import (
    REGISTERED_ARCH,
    RWKV7_GOOSE_REVISION,
    BackboneState,
    DummyBackbone,
    RWKVBackbone,
    make_backbone,
)


def test_rwkv7_revision_is_40_hex_sha() -> None:
    assert isinstance(RWKV7_GOOSE_REVISION, str)
    assert len(RWKV7_GOOSE_REVISION) == 40
    int(RWKV7_GOOSE_REVISION, 16)  # raises ValueError if not hex


def test_registered_arch_contains_rwkv() -> None:
    assert "rwkv7-goose" in REGISTERED_ARCH
    assert REGISTERED_ARCH["rwkv7-goose"].startswith("RWKV/RWKV7-Goose")


def test_registered_arch_excludes_unshipped_v011_entries() -> None:
    # Mamba-3 backbone is deferred to v0.1.1 — must not appear as a live entry
    # in v0.1.0 to avoid misleading make_backbone callers.
    assert "mamba-2.8b" not in REGISTERED_ARCH
    assert "mamba3-2.7b" not in REGISTERED_ARCH


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


def test_rwkv_backbone_refuses_unpinned_revision() -> None:
    # Audit finding: revision pin must be runtime-enforced, not just CI-grepped.
    with pytest.raises(ValueError, match="refusing to load revision"):
        RWKVBackbone.from_pretrained("rwkv7-goose", revision="deadbeef" * 5)


def test_rwkv_backbone_accepts_pinned_revision_path_until_hf_call() -> None:
    # Passing the pinned SHA must pass the gate; the call then fails at the
    # transformers import / network gate (HF_HUB_OFFLINE=1), which proves the
    # gate is opt-in for the exact SHA only.
    with pytest.raises(Exception) as excinfo:
        RWKVBackbone.from_pretrained("rwkv7-goose", revision=RWKV7_GOOSE_REVISION)
    # Must NOT be the pin-violation ValueError
    assert "refusing to load revision" not in str(excinfo.value)


def test_rwkv_backbone_rejects_mutable_branch_revision() -> None:
    # Audit finding (RCE): trust_remote_code must never load a mutable ref;
    # only the pinned immutable SHA is accepted.
    with pytest.raises(ValueError, match="refusing to load revision"):
        RWKVBackbone.from_pretrained("rwkv7-goose", revision="main")


def test_rwkv_backbone_has_no_unsafe_revision_escape_hatch() -> None:
    # Audit finding (RCE): the allow_unsafe_revision override was removed so
    # an arbitrary revision can never reach trust_remote_code at runtime.
    import inspect

    params = inspect.signature(RWKVBackbone.from_pretrained).parameters
    assert "allow_unsafe_revision" not in params


def test_rwkv_backbone_rejects_non_rwkv_arch() -> None:
    with pytest.raises(ValueError, match="only supports rwkv7-goose"):
        RWKVBackbone.from_pretrained("mamba3-2.7b")
