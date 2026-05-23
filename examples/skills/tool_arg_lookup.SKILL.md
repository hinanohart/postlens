# SKILL: tool_arg_lookup
**Class**: retrieval-heavy (SSM G&A weakness probe — arxiv:2504.18574)
**Input**: `transcript: list[dict]`, `tool_name: str`, `arg_name: str`
**Output**: `{value: str | None, turn_index: int | None}`

## Steps
1. Walk `transcript` (a list of `{role, content}` turns).
2. Find the most recent turn where `tool_name` was called with `arg_name=...`.
3. Return the value and the turn index; `None` if absent.

## Latency probe
`postlens.bench.time_skill(skill="tool_arg_lookup", backbone={"transformer","rwkv7"})`

## Expected weakness on SSM
Per arxiv:2504.18574, retrieving a specific token (the argument value) from
a long agent transcript is the canonical Gather-and-Aggregate failure mode
for recurrent LMs. Expect degraded accuracy vs. Transformer. Reported, not hidden.

## Safety
- No shell exec. No network. No file write.
- `transcript` length capped at 1000 turns.
