"""postlens CLI entry point (v0.1.0 minimal driver)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from postlens import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="postlens",
        description="Post-Transformer agent latency framework",
    )
    parser.add_argument("--version", action="version", version=f"postlens {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("info", help="print package info")
    run_p = sub.add_parser(
        "run",
        help="run a task with a chosen backbone and skill directory",
    )
    run_p.add_argument("task", help="path to a task markdown file")
    run_p.add_argument(
        "--backbone",
        default="dummy",
        help="backbone arch (default: dummy; rwkv7-goose for real)",
    )
    run_p.add_argument(
        "--skills",
        default="examples/skills/",
        help="directory containing *.SKILL.md probes",
    )
    run_p.add_argument("--decode-tokens", type=int, default=16, help="tokens to decode per skill")

    args = parser.parse_args(argv)

    if args.cmd == "info":
        print(f"postlens v{__version__}")
        print("Post-Transformer agent latency framework (alpha)")
        return 0
    if args.cmd == "run":
        return _cmd_run(args)

    parser.print_help()
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from postlens.backbone import DummyBackbone, make_backbone
    from postlens.bench import LatencyBench
    from postlens.runtime import PostAgent
    from postlens.skill import load_skills

    task_path = Path(args.task)
    if not task_path.is_file():
        print(f"error: task file not found: {task_path}", file=sys.stderr)
        return 2
    skills_dir = Path(args.skills)
    if not skills_dir.is_dir():
        print(f"error: skills dir not found: {skills_dir}", file=sys.stderr)
        return 2

    backbone = DummyBackbone() if args.backbone == "dummy" else make_backbone(args.backbone)
    skills = load_skills(skills_dir)
    agent = PostAgent(backbone=backbone, skills=skills)

    print(f"task: {task_path}")
    print(f"backbone: {backbone.arch}")
    print(f"skills: {len(skills)} loaded ({', '.join(agent.list_skills())})")
    print()
    print("skill,ttft_s,tok_per_s,state_bytes")
    bench = LatencyBench(backbone)
    for name in agent.list_skills():
        r = bench.run(skill=name, prompt_tokens=list(range(32)), decode_tokens=args.decode_tokens)
        ttft = f"{r.ttft_s:.6f}" if r.ttft_s is not None else ""
        tps = f"{r.tok_per_s:.2f}" if r.tok_per_s is not None else ""
        print(f"{r.skill},{ttft},{tps},{r.state_bytes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
