# task: hello

A trivial skill-bundle exercise: run each of the 5 example skills once with a
small prompt and emit the latency summary.

## Inputs
- `skills_dir`: `examples/skills/`
- `backbone`: `rwkv7-goose` (or `dummy` for offline)

## Expected output
A CSV with columns: `skill, ttft_s, tok_per_s, state_bytes`.
