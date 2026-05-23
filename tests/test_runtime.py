from __future__ import annotations

from pathlib import Path

import pytest

from postlens.backbone import DummyBackbone
from postlens.runtime import PostAgent
from postlens.skill import load_skills

SKILLS_DIR = Path(__file__).resolve().parents[1] / "examples" / "skills"


def _agent() -> PostAgent:
    return PostAgent(backbone=DummyBackbone(vocab_size=32), skills=load_skills(SKILLS_DIR))


def test_post_agent_requires_skills() -> None:
    with pytest.raises(ValueError):
        PostAgent(backbone=DummyBackbone(), skills=[])


def test_list_skills_returns_sorted_names() -> None:
    a = _agent()
    names = a.list_skills()
    assert names == sorted(names)
    assert "csv_stat" in names


def test_run_unknown_skill_raises() -> None:
    a = _agent()
    with pytest.raises(KeyError):
        a.run("not-a-skill", prompt_tokens=[1, 2, 3])


def test_run_produces_max_new_tokens() -> None:
    a = _agent()
    rec = a.run("csv_stat", prompt_tokens=[1, 2, 3, 4], max_new_tokens=5)
    assert len(rec.output_tokens) == 5
    assert rec.backbone_arch == "dummy"
    assert rec.state_handle is not None


def test_run_stores_state_handle_in_store() -> None:
    a = _agent()
    rec = a.run("regex_mask", prompt_tokens=[7, 8], max_new_tokens=1)
    assert a.store.has(rec.state_handle)


def test_as_smolagents_tools_returns_one_per_skill() -> None:
    a = _agent()
    descs = a.as_smolagents_tools()
    assert len(descs) == 5
    assert all("klass" in d for d in descs)


def test_run_empty_prompt_raises() -> None:
    # Audit finding: empty prompt used to silently fall back to last_tok=0
    # — fail-closed instead.
    a = _agent()
    with pytest.raises(ValueError, match="prompt_tokens must be non-empty"):
        a.run("csv_stat", prompt_tokens=[], max_new_tokens=1)


def test_run_negative_max_new_tokens_raises() -> None:
    a = _agent()
    with pytest.raises(ValueError, match="max_new_tokens must be >= 0"):
        a.run("csv_stat", prompt_tokens=[1], max_new_tokens=-1)
