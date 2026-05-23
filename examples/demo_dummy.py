"""End-to-end demo: PostAgent + DummyBackbone over the 5 example skills.

Run from the repo root:

    PYTHONPATH=src python3 examples/demo_dummy.py

This downloads nothing, requires no GPU, and finishes in milliseconds. It's
the smoke demo referenced in the README's Quickstart and the v0.1.0 release
acceptance test.
"""

from __future__ import annotations

from pathlib import Path

from postlens.backbone import DummyBackbone
from postlens.bench import LatencyBench
from postlens.runtime import PostAgent
from postlens.skill import load_skills

SKILLS_DIR = Path(__file__).resolve().parent / "skills"


def main() -> int:
    skills = load_skills(SKILLS_DIR)
    print(f"loaded {len(skills)} skills: {[s.name for s in skills]}")

    agent = PostAgent(backbone=DummyBackbone(vocab_size=128), skills=skills)
    for name in agent.list_skills():
        rec = agent.run(name, prompt_tokens=[1, 2, 3, 4, 5], max_new_tokens=4)
        elapsed_us = f"{rec.elapsed_s * 1e6:.0f}us"
        print(f"  ran skill={name!r:<22} tokens={rec.output_tokens} elapsed={elapsed_us}")

    print()
    print("LatencyBench summary:")
    bench = LatencyBench(DummyBackbone(vocab_size=128))
    for name in agent.list_skills():
        result = bench.run(skill=name, prompt_tokens=list(range(128)), decode_tokens=32)
        d = result.as_dict()
        print(
            f"  {d['skill']:<22} ttft={d['ttft_s'] * 1e6:.0f}us tok/s={d['tok_per_s']:.0f}"
            f" state_bytes={d['state_bytes']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
