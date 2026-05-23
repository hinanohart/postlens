# SKILL: unit_convert
**Class**: retrieval-light
**Input**: `expr: str` (natural-language unit conversion, e.g. "12 km in miles")
**Output**: `{value: float, from_unit: str, to_unit: str}`

## Steps
1. Parse `expr` with a regex into `<value> <from> in <to>`.
2. Resolve units against the lookup table (length / mass / time / temperature).
3. Apply conversion factor; return dict.

## Latency probe
`postlens.bench.time_skill(skill="unit_convert", backbone={"transformer","rwkv7"})`

## Safety
- No shell exec. No network. No file write.
- Reject inputs > 200 chars (fail-closed) to bound runtime.
