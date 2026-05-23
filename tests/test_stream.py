from __future__ import annotations

import pytest

from postlens.stream import InterleavedScheduler, ScheduledOp


def test_chunk_size_zero_rejected() -> None:
    with pytest.raises(ValueError):
        InterleavedScheduler(chunk_size=0)


def test_n_streams_zero_rejected() -> None:
    with pytest.raises(ValueError):
        InterleavedScheduler(n_streams=0)


def test_plan_emits_prefill_chunks_then_steps_then_yield() -> None:
    s = InterleavedScheduler(chunk_size=4, n_streams=2)
    ops = s.plan("a", prefill_tokens=10, step_tokens=3)
    kinds = [o.kind for o in ops]
    assert kinds.count("prefill") == 3  # 4 + 4 + 2
    assert kinds.count("step") == 3
    assert kinds[-1] == "yield"


def test_plan_chunk_token_counts_match_prefill_total() -> None:
    s = InterleavedScheduler(chunk_size=8, n_streams=2)
    ops = s.plan("a", prefill_tokens=20, step_tokens=0)
    prefill_total = sum(o.n_tokens for o in ops if o.kind == "prefill")
    assert prefill_total == 20


def test_plan_stream_indices_round_robin() -> None:
    s = InterleavedScheduler(chunk_size=4, n_streams=3)
    ops = s.plan("a", prefill_tokens=12, step_tokens=0)
    prefill_streams = [o.stream_idx for o in ops if o.kind == "prefill"]
    assert prefill_streams == [0, 1, 2]


def test_merge_plans_interleaves_round_robin() -> None:
    s = InterleavedScheduler(chunk_size=4, n_streams=1)
    pa = s.plan("a", prefill_tokens=4, step_tokens=1)
    pb = s.plan("b", prefill_tokens=4, step_tokens=1)
    merged = InterleavedScheduler.merge_plans([pa, pb])
    # First two entries must alternate skills, not group by skill
    assert merged[0].skill_id == "a"
    assert merged[1].skill_id == "b"


def test_plan_zero_decode_only_yields() -> None:
    s = InterleavedScheduler(chunk_size=4)
    ops = s.plan("a", prefill_tokens=0, step_tokens=0)
    assert ops == [ScheduledOp(kind="yield", skill_id="a", n_tokens=0, stream_idx=0)]
