from __future__ import annotations

from postlens.backbone import DummyBackbone
from postlens.bench import BenchResult, LatencyBench, time_skill


def test_time_skill_string_backbone_returns_stub() -> None:
    out = time_skill("csv_stat", "rwkv7")
    assert out["status"] == "stub"
    assert out["ttft_s"] is None
    assert out["skill"] == "csv_stat"


def test_latency_bench_run_against_dummy() -> None:
    bench = LatencyBench(DummyBackbone(vocab_size=64))
    result = bench.run(skill="csv_stat", prompt_tokens=[1, 2, 3], decode_tokens=4)
    assert isinstance(result, BenchResult)
    assert result.status == "ok"
    assert result.decoded == 4
    assert result.ttft_s is not None and result.ttft_s >= 0
    assert result.tok_per_s is None or result.tok_per_s > 0


def test_latency_bench_default_prompt() -> None:
    bench = LatencyBench(DummyBackbone())
    result = bench.run(skill="x", decode_tokens=2)
    assert result.decoded == 2


def test_bench_result_as_dict_has_full_schema() -> None:
    r = BenchResult(
        skill="a",
        backbone="b",
        ttft_s=0.1,
        tok_per_s=10.0,
        peak_vram_mb=None,
        state_bytes=8,
        decoded=2,
        status="ok",
    )
    d = r.as_dict()
    expected_keys = {
        "skill",
        "backbone",
        "ttft_s",
        "tok_per_s",
        "peak_vram_mb",
        "state_bytes",
        "decoded",
        "status",
    }
    assert set(d.keys()) == expected_keys


def test_time_skill_with_real_backbone_runs() -> None:
    bb = DummyBackbone(vocab_size=16)
    out = time_skill("csv_stat", bb, prompt_tokens=[1, 2], decode_tokens=3)
    assert out["status"] == "ok"
    assert out["decoded"] == 3
