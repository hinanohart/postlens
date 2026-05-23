# postlens

Post-Transformer agent latency framework. Run smolagents `CodeAgent` with RWKV-7
or Mamba-3 backbones and measure skill-execution latency vs. Transformer baselines.

Status: v0.1.0 (experimental, alpha). No SLA. Not for production.

## Honest weakness map

postlens ships skill probes that include both **retrieval-light** (`csv_stat`, `unit_convert`,
`regex_mask`) and **retrieval-heavy** (`grep_summarize`, `tool_arg_lookup`) tasks. Per
arxiv:2504.18574 (Gather-and-Aggregate sharpness limit in recurrent LMs), retrieval-heavy
skills are expected to degrade on pure-SSM backbones. This is reported, not hidden.

## Install

```bash
pip install -e .
# optional: flash-linear-attention kernels
pip install -e ".[fla]"
```

PyPI publish is deferred to v0.1.1; v0.1.0 is GitHub-source-install only.

## Quickstart

```bash
postlens run examples/tasks/hello.md --backbone rwkv7-goose --skills examples/skills/
```

(`run` subcommand is a v0.1.0 minimal driver; full CLI in v0.2.)

## Benchmark methodology

`postlens.bench.LatencyBench` measures four metrics over a skill bundle:
- TTFT (time-to-first-token)
- tok/s (decode throughput)
- peak VRAM
- recurrent state size (bytes)

Transformer baselines: Qwen3-Omni-7B and Llama-3.3-8B-Instruct.
SSM backbones: RWKV-7 Goose 2.9B (default), Mamba-3 (v0.1.1 opt-in).

## License & attribution

- Apache-2.0
- Interops with [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT) via the optional `fla` extra
- RWKV-7 Goose model: Apache-2.0, paper arxiv:2503.14456
- Mamba-3 paper: arxiv:2603.15569
- SSM retrieval weakness reference: arxiv:2504.18574
- SkillsBench compatibility target: arxiv:2602.12670v3

## Acknowledgements

- RWKV community for the Goose checkpoints
- HuggingFace `smolagents` team for the CodeAgent runtime
- `flash-linear-attention` maintainers for the kernel library
