"""`PostAgent`: composition wrapper over a Backbone + StateStore + Skills.

We deliberately *compose* smolagents.CodeAgent (no subclassing) so postlens
stays decoupled from smolagents internals. In v0.1.0 the CodeAgent wiring is
lazy — calling `PostAgent.run` only imports smolagents on demand.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from postlens.backbone import Backbone, BackboneState
from postlens.skill import Skill
from postlens.state import RecurrentStateStore, StateHandle


@dataclasses.dataclass
class RunRecord:
    """Per-skill execution record returned by `PostAgent.run`."""

    skill_id: str
    backbone_arch: str
    state_handle: StateHandle | None
    output_tokens: list[int]
    elapsed_s: float


class PostAgent:
    """Compose a backbone with a skill set and a recurrent state store."""

    def __init__(
        self,
        backbone: Backbone,
        skills: list[Skill],
        store: RecurrentStateStore | None = None,
    ) -> None:
        if not skills:
            raise ValueError("PostAgent requires at least one Skill")
        self.backbone = backbone
        self.skills: dict[str, Skill] = {s.name: s for s in skills}
        self.store = store or RecurrentStateStore()

    def list_skills(self) -> list[str]:
        return sorted(self.skills.keys())

    def run(
        self,
        skill_id: str,
        prompt_tokens: list[int],
        max_new_tokens: int = 16,
    ) -> RunRecord:
        """Drive prefill+step for a single skill, caching the state."""
        if skill_id not in self.skills:
            raise KeyError(f"unknown skill: {skill_id!r}")
        if not prompt_tokens:
            raise ValueError("prompt_tokens must be non-empty")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")
        import time

        t0 = time.perf_counter()
        state = self.backbone.prefill(prompt_tokens)
        handle = self.store.save(skill_id, state)
        out_tokens: list[int] = []
        cur_state: BackboneState = state
        last_tok = prompt_tokens[-1]
        for _ in range(max_new_tokens):
            cur_state, last_tok = self.backbone.step(cur_state, last_tok)
            out_tokens.append(last_tok)
        elapsed = time.perf_counter() - t0
        return RunRecord(
            skill_id=skill_id,
            backbone_arch=self.backbone.arch,
            state_handle=handle,
            output_tokens=out_tokens,
            elapsed_s=elapsed,
        )

    def as_smolagents_tools(self) -> list[dict[str, Any]]:
        """Return tool descriptors for every skill (smolagents-compatible).

        Actual smolagents integration is opt-in: callers wrap these via
        `smolagents.Tool` themselves to keep our CI fast tier free of
        smolagents at install time.
        """
        return [s.as_tool() for s in self.skills.values()]
