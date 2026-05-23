from __future__ import annotations

from pathlib import Path

import pytest

from postlens.skill import Skill, SkillParseError, load_skills

SKILLS_DIR = Path(__file__).resolve().parents[1] / "examples" / "skills"


def test_skill_dir_contains_five_skills() -> None:
    paths = sorted(SKILLS_DIR.glob("*.SKILL.md"))
    assert len(paths) == 5


def test_load_all_five_parses() -> None:
    skills = load_skills(SKILLS_DIR)
    names = {s.name for s in skills}
    assert names == {
        "csv_stat",
        "unit_convert",
        "regex_mask",
        "grep_summarize",
        "tool_arg_lookup",
    }


def test_honest_weakness_map_three_light_two_heavy() -> None:
    skills = load_skills(SKILLS_DIR)
    light = [s for s in skills if "light" in s.klass]
    heavy = [s for s in skills if "heavy" in s.klass]
    assert len(light) == 3
    assert len(heavy) == 2


def test_skill_parses_steps_nonempty() -> None:
    s = Skill.load(SKILLS_DIR / "csv_stat.SKILL.md")
    assert s.name == "csv_stat"
    assert s.input_spec.startswith("`path:")
    assert "count" in s.output_spec
    assert len(s.steps) >= 2


def test_as_tool_descriptor_contains_klass() -> None:
    s = Skill.load(SKILLS_DIR / "grep_summarize.SKILL.md")
    desc = s.as_tool()
    assert desc["name"] == "grep_summarize"
    assert "heavy" in desc["klass"]


def test_from_text_rejects_missing_header() -> None:
    with pytest.raises(SkillParseError):
        Skill.from_text("not a skill file")


def test_load_skills_non_directory_raises(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("nope")
    with pytest.raises(SkillParseError):
        load_skills(f)
