from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _hf_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard-block HF network in fast CI tier."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
