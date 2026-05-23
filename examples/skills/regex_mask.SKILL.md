# SKILL: regex_mask
**Class**: retrieval-light
**Input**: `text: str`
**Output**: `{masked: str, hits: int}`

## Steps
1. Apply email regex `[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}` → replace with `<EMAIL>`.
2. Apply URL regex `https?://[^\s]+` → replace with `<URL>`.
3. Return `{masked, hits}` where `hits` is the total substitution count.

## Latency probe
`postlens.bench.time_skill(skill="regex_mask", backbone={"transformer","rwkv7"})`

## Safety
- No shell exec. No network. No file write.
- Input capped at 1 MiB; longer strings rejected fail-closed.
