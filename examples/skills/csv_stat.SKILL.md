# SKILL: csv_stat
**Class**: retrieval-light
**Input**: `path: str`, `column: str`
**Output**: `{count: int, mean: float, min: float, max: float}`

## Steps
1. Open `path` (UTF-8) via pandas. Cap rows at 100_000 (fail-closed beyond).
2. Compute `count / mean / min / max` over `column`.
3. Return dict.

## Latency probe
`postlens.bench.time_skill(skill="csv_stat", backbone={"transformer","rwkv7"})`

## Safety
- No shell exec. No network. No file write.
- `path` must be inside `examples/data/` (CI fixture only).
