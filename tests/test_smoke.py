from __future__ import annotations

import postlens


def test_version_is_pep440_ish() -> None:
    v = postlens.__version__
    assert isinstance(v, str)
    parts = v.split(".")
    assert len(parts) >= 3
    assert all(p.isdigit() or p[0].isdigit() for p in parts[:3])


def test_public_surface_minimal() -> None:
    assert set(postlens.__all__) >= {"__version__"}
