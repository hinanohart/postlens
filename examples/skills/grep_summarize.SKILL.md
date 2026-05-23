# SKILL: grep_summarize
**Class**: retrieval-heavy (SSM G&A weakness probe — arxiv:2504.18574)
**Input**: `corpus: str`, `pattern: str`
**Output**: `{matches: list[str], summary: str}`

## Steps
1. Scan `corpus` for regex `pattern`; collect up to 20 matching lines.
2. Concatenate matches; pass to backbone for a one-paragraph summary.
3. Return matches + summary.

## Latency probe
`postlens.bench.time_skill(skill="grep_summarize", backbone={"transformer","rwkv7"})`

## Expected weakness on SSM
Per arxiv:2504.18574, the Gather-and-Aggregate head sharpness limit causes
recurrent LMs to under-retrieve in long contexts. Expect lower match recall
on RWKV-7 / Mamba-3 vs. Transformer baselines. This is reported in the
benchmark, not hidden.

## Safety
- No shell exec. No network. No file write.
- `corpus` capped at 256 KiB.
