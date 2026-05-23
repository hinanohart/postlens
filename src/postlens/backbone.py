"""Backbone abstraction over RWKV-7 / Mamba-3 (and stub for tests)."""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import Any

# Pinned RWKV-7 Goose 2.9B revision SHA (trust_remote_code freeze).
# CI verifies this literal is unchanged (drift guard).
RWKV7_GOOSE_REVISION = "b742a96904c69424901f2b8cf729b67863168063"

REGISTERED_ARCH: dict[str, str] = {
    "rwkv7-goose": "RWKV/RWKV7-Goose-World3-2.9B-HF",
}


@dataclasses.dataclass(frozen=True)
class BackboneState:
    """Opaque container for a recurrent backbone hidden state.

    payload is whatever the concrete backbone stores (tensor tuple, list of
    tensors, numpy arrays, etc.). The framework treats it as opaque.
    """

    arch: str
    bytes_size: int
    payload: Any = None


class Backbone(ABC):
    """Abstract recurrent backbone (RWKV-7 / Mamba-3 family)."""

    arch: str

    @classmethod
    @abstractmethod
    def from_pretrained(cls, arch: str, revision: str | None = None) -> Backbone: ...

    @abstractmethod
    def prefill(self, token_ids: list[int]) -> BackboneState: ...

    @abstractmethod
    def step(self, state: BackboneState, token_id: int) -> tuple[BackboneState, int]: ...

    @abstractmethod
    def state_bytes(self, state: BackboneState) -> int: ...


class DummyBackbone(Backbone):
    """In-process backbone used by unit tests; no HF download.

    Deterministic next-token = (last + 1) mod vocab_size, prefill carries
    the cumulative XOR as the state payload. Lets us exercise the full
    runtime / state / bench code paths without torch or HF.
    """

    arch = "dummy"

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size

    @classmethod
    def from_pretrained(cls, arch: str, revision: str | None = None) -> DummyBackbone:
        return cls()

    def prefill(self, token_ids: list[int]) -> BackboneState:
        acc = 0
        for t in token_ids:
            acc ^= int(t)
        return BackboneState(arch=self.arch, bytes_size=8, payload=acc)

    def step(self, state: BackboneState, token_id: int) -> tuple[BackboneState, int]:
        new_payload = (int(state.payload or 0) ^ int(token_id)) & 0xFFFFFFFF
        next_tok = (int(token_id) + 1) % self.vocab_size
        return (
            BackboneState(arch=self.arch, bytes_size=8, payload=new_payload),
            next_tok,
        )

    def state_bytes(self, state: BackboneState) -> int:
        return state.bytes_size


class RWKVBackbone(Backbone):
    """RWKV-7 Goose wrapper. Lazy-imports transformers; gated by HF_HUB_OFFLINE.

    Concrete forward path is intentionally minimal in v0.1.0 — the public
    contract (prefill/step/state_bytes) is stable so v0.1.1 can land a
    fla-accelerated path without breaking callers.
    """

    arch = "rwkv7-goose"

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._model = model
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        arch: str = "rwkv7-goose",
        revision: str | None = None,
        allow_unsafe_revision: bool = False,
    ) -> RWKVBackbone:
        if arch != "rwkv7-goose":
            raise ValueError(f"RWKVBackbone only supports rwkv7-goose, got {arch!r}")
        revision = revision or RWKV7_GOOSE_REVISION
        if revision != RWKV7_GOOSE_REVISION and not allow_unsafe_revision:
            raise ValueError(
                f"refusing to load revision {revision!r}: pinned to "
                f"{RWKV7_GOOSE_REVISION!r}. Pass allow_unsafe_revision=True to override "
                f"(only do this if you have audited the new upstream code)."
            )
        from transformers import AutoModelForCausalLM, AutoTokenizer

        repo = REGISTERED_ARCH[arch]
        tok = AutoTokenizer.from_pretrained(repo, revision=revision, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, trust_remote_code=True
        )
        return cls(model=model, tokenizer=tok)

    def prefill(self, token_ids: list[int]) -> BackboneState:
        if not token_ids:
            raise ValueError("prefill requires at least one token id")
        import torch

        ids = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            out = self._model(input_ids=ids, use_cache=True)
        state = _extract_state(out)
        size = _approx_state_bytes(state)
        return BackboneState(arch=self.arch, bytes_size=size, payload=state)

    def step(self, state: BackboneState, token_id: int) -> tuple[BackboneState, int]:
        import torch

        ids = torch.tensor([[token_id]], dtype=torch.long)
        with torch.no_grad():
            out = self._model(input_ids=ids, state=state.payload, use_cache=True)
        next_state = _extract_state(out)
        logits = out.logits[:, -1, :]
        next_tok = int(torch.argmax(logits, dim=-1).item())
        size = _approx_state_bytes(next_state)
        return (
            BackboneState(arch=self.arch, bytes_size=size, payload=next_state),
            next_tok,
        )

    def state_bytes(self, state: BackboneState) -> int:
        return state.bytes_size


def _extract_state(out: Any) -> Any:
    state = getattr(out, "state", None)
    if state is None:
        state = getattr(out, "past_key_values", None)
    return state


def _approx_state_bytes(state: Any) -> int:
    """Best-effort byte-count for an opaque torch state container."""
    if state is None:
        return 0
    try:
        import torch
    except Exception:
        return 0
    total = 0
    if isinstance(state, (list, tuple)):
        for item in state:
            total += _approx_state_bytes(item)
        return total
    if isinstance(state, dict):
        for v in state.values():
            total += _approx_state_bytes(v)
        return total
    if torch.is_tensor(state):
        return state.element_size() * state.nelement()
    return 0


def make_backbone(arch: str, revision: str | None = None) -> Backbone:
    """Factory: dispatch to the concrete backbone for `arch`."""
    if arch == "dummy":
        return DummyBackbone.from_pretrained(arch)
    if arch == "rwkv7-goose":
        return RWKVBackbone.from_pretrained(arch, revision=revision)
    raise ValueError(f"unknown backbone arch: {arch!r}")
