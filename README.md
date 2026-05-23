# postlens

Post-Transformer agent latency framework. Drive `smolagents.CodeAgent`-style
workflows with **RWKV-7 Goose** (or future Mamba-3) recurrent backbones and
benchmark skill-execution latency against Transformer baselines.

> **Status: v0.1.0 (alpha).** Experimental. No SLA. Not for production.
> v0.1.0 ships the framework, skill loader, scheduler, and the in-process
> latency bench. Cross-process state persistence and PyPI distribution land
> in v0.1.1.

## Why

`vLLM` and `SGLang` do not target RWKV-7 ([vLLM #3583 was closed
"not planned"](https://github.com/vllm-project/vllm/issues/3583)). The
`smolagents` ecosystem is Transformer-shaped by default. postlens fills the
gap: a thin, honest, latency-instrumented harness for skill-driven SSM agents.

## Honest weakness map

postlens ships **five skill probes** — three retrieval-light, two
retrieval-heavy:

| Skill | Class | Why it's here |
|---|---|---|
| `csv_stat` | light | baseline structured compute |
| `unit_convert` | light | baseline NL parsing |
| `regex_mask` | light | baseline regex throughput |
| `grep_summarize` | heavy | exposes SSM Gather-and-Aggregate weakness |
| `tool_arg_lookup` | heavy | exposes SSM long-context retrieval weakness |

Per [arxiv:2504.18574](https://arxiv.org/abs/2504.18574), the
Gather-and-Aggregate head sharpness limit causes recurrent LMs to under-retrieve
in long contexts. We **expect** the two heavy skills to degrade vs. Transformer
baselines. Benchmarks report this, not hide it.

## Install

```bash
git clone https://github.com/hinanohart/postlens
cd postlens
pip install -e .
# optional: flash-linear-attention kernels (MIT)
pip install -e ".[fla]"
# dev tooling
pip install -e ".[dev]"
```

PyPI publication is deferred to v0.1.1. v0.1.0 is GitHub-source-install only
(intentional — gives time for `postlens` name reservation only after API
stability is verified by external users).

## Quickstart

```python
from postlens.backbone import DummyBackbone
from postlens.runtime import PostAgent
from postlens.skill import load_skills

skills = load_skills("examples/skills/")
agent = PostAgent(backbone=DummyBackbone(), skills=skills)
record = agent.run("csv_stat", prompt_tokens=[1, 2, 3, 4], max_new_tokens=8)
print(record.output_tokens, record.elapsed_s)
```

`DummyBackbone` is the test-friendly stand-in. To drive a real RWKV-7 Goose
2.9B model:

```python
from postlens.backbone import RWKVBackbone
bb = RWKVBackbone.from_pretrained("rwkv7-goose")  # pinned to SHA b742a96
```

The pinned revision is enforced at the CI level (drift guard) so
`trust_remote_code=True` does not silently execute new upstream code.

## Benchmark methodology

`postlens.bench.LatencyBench` measures four metrics over a skill execution:

- **TTFT** — time-to-first-token (prefill latency)
- **tok/s** — decode throughput
- **peak VRAM** — `torch.cuda.max_memory_allocated()` snapshot (None on CPU)
- **state bytes** — recurrent state size, exposing SSM's constant-memory edge

> **Honest scope of v0.1.0 numbers.** The bundled bench scaffolding runs
> against `DummyBackbone` by default — its `ttft_s` / `tok_per_s` values
> reflect the harness overhead, not real RWKV-7 inference speed. Real
> RWKV-7 vs Transformer comparison numbers (with `Qwen3-Omni-7B` /
> `Llama-3.3-8B` baselines on `SkillsBench`) land in v0.1.1; this v0.1.0
> ships the **measurement framework** so the v0.1.1 numbers are reproducible
> rather than ad-hoc.

Transformer baselines we plan to compare against in the v0.1.0 paper draft:
- `Qwen/Qwen3-Omni-7B-Instruct`
- `meta-llama/Llama-3.3-8B-Instruct`

SSM backbones:
- v0.1.0 default: **RWKV-7 Goose 2.9B** (`RWKV/RWKV7-Goose-World3-2.9B-HF`)
- v0.1.1 opt-in: Mamba-3 (`state-spaces/mamba3-2.7b`, gated on public release)

## Demo

A self-contained example that runs end-to-end with `DummyBackbone` (no HF
download required) lives at `examples/demo_dummy.py`:

```bash
PYTHONPATH=src python3 examples/demo_dummy.py
```

After `pip install -e .` you can also drive it via the CLI:

```bash
postlens run examples/tasks/hello.md \
  --backbone dummy \
  --skills examples/skills/ \
  --decode-tokens 16
```

The CLI emits a CSV (`skill,ttft_s,tok_per_s,state_bytes`) suitable for piping
into pandas / a notebook.

## v0.1.1 deferred (honest scope)

These items are explicitly out of scope for v0.1.0 and tracked for v0.1.1:

- Mamba-3 concrete backbone (HF repo not public as of 2026-05-23)
- Cross-process state cache (safetensors-based; no pickle)
- Real `torch.cuda.Stream` dispatch (v0.1.0 ships logical scheduler trace)
- `SkillsBench` runner + Transformer baselines (Qwen3-Omni-7B, Llama-3.3-8B)
- Live `smolagents.CodeAgent` composition in `PostAgent.run`
- MCP tool wrapping in `Skill.as_tool`
- PyPI publication

## License & attribution

- **Apache-2.0**
- Optional interop with [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (**MIT**), pulled via the `[fla]` extra
- RWKV-7 Goose paper: arxiv:2503.14456
- Mamba-3 paper: arxiv:2603.15569
- SSM retrieval weakness reference: arxiv:2504.18574
- SkillsBench compatibility target: arxiv:2602.12670v3

## Acknowledgements

- RWKV community for the Goose 2.9B checkpoints
- HuggingFace `smolagents` team for the CodeAgent design
- `flash-linear-attention` maintainers for the kernel library
