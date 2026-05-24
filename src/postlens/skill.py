"""Skill loader: parses Anthropic Skills `.SKILL.md` markdown files."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

# DoS guards: a `.SKILL.md` is a small markdown descriptor (the bundled ones
# are < 1 KiB). Cap per-file size and per-directory file count so a hostile or
# accidental huge file/dir cannot exhaust memory.
MAX_SKILL_BYTES = 1 << 20  # 1 MiB
MAX_SKILL_FILES = 256

_HEADER_RE = re.compile(r"^# SKILL:\s*(\S+)\s*$", re.MULTILINE)
_FIELD_RE = re.compile(r"^\*\*(\w+)\*\*:\s*(.+?)\s*$", re.MULTILINE)
_STEPS_HEADER_RE = re.compile(r"^##\s+Steps\s*$", re.MULTILINE)
_NEXT_HEADER_RE = re.compile(r"^##\s+", re.MULTILINE)


class SkillParseError(ValueError):
    """Raised when a `.SKILL.md` cannot be parsed."""


@dataclasses.dataclass(frozen=True)
class Skill:
    """Parsed representation of an Anthropic Skills markdown file."""

    name: str
    klass: str  # "retrieval-light" | "retrieval-heavy" | other
    input_spec: str
    output_spec: str
    steps: tuple[str, ...]
    raw: str
    source: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> Skill:
        p = Path(path)
        size = p.stat().st_size
        if size > MAX_SKILL_BYTES:
            raise SkillParseError(
                f"skill file {p} is {size} bytes, exceeds limit of {MAX_SKILL_BYTES}"
            )
        text = p.read_text(encoding="utf-8")
        return cls._parse(text, source=p)

    @classmethod
    def from_text(cls, text: str) -> Skill:
        return cls._parse(text, source=None)

    @classmethod
    def _parse(cls, text: str, source: Path | None) -> Skill:
        m = _HEADER_RE.search(text)
        if not m:
            raise SkillParseError("missing `# SKILL: <name>` header")
        name = m.group(1).strip()
        fields = {k.lower(): v for k, v in _FIELD_RE.findall(text)}
        klass = fields.get("class", "unspecified")
        input_spec = fields.get("input", "")
        output_spec = fields.get("output", "")
        steps_block = _extract_section(text, _STEPS_HEADER_RE, _NEXT_HEADER_RE)
        steps = tuple(_iter_steps(steps_block))
        return cls(
            name=name,
            klass=klass,
            input_spec=input_spec,
            output_spec=output_spec,
            steps=steps,
            raw=text,
            source=source,
        )

    def as_tool(self, fn: Callable[..., Any] | None = None) -> dict[str, Any]:
        """Return a smolagents-compatible tool descriptor.

        Caller supplies the actual implementation `fn`; v0.1.0 ships
        descriptors only (no smolagents import — keeps the fast tier light).
        """
        return {
            "name": self.name,
            "description": f"{self.klass} skill: {self.input_spec} -> {self.output_spec}",
            "inputs": self.input_spec,
            "outputs": self.output_spec,
            "fn": fn,
            "klass": self.klass,
        }


def load_skills(directory: str | Path) -> list[Skill]:
    """Load every `*.SKILL.md` under `directory` (non-recursive)."""
    d = Path(directory)
    if not d.is_dir():
        raise SkillParseError(f"not a directory: {d}")
    paths = sorted(d.glob("*.SKILL.md"))
    if len(paths) > MAX_SKILL_FILES:
        raise SkillParseError(
            f"{d} has {len(paths)} skill files, exceeds limit of {MAX_SKILL_FILES}"
        )
    return [Skill.load(p) for p in paths]


def _extract_section(text: str, header_re: re.Pattern[str], next_header_re: re.Pattern[str]) -> str:
    m = header_re.search(text)
    if not m:
        return ""
    start = m.end()
    after = text[start:]
    n = next_header_re.search(after)
    end = n.start() if n else len(after)
    return after[:end]


def _iter_steps(block: str) -> Iterable[str]:
    """Yield numbered steps, merging indented continuation lines into the
    preceding step.

    A line starting with `N.` opens a new step. Subsequent lines that begin
    with whitespace (continuation) or `- ` (sub-bullet) are appended to the
    current step. Blank lines flush the current step.
    """
    current: list[str] = []

    def _flush() -> Iterable[str]:
        if current:
            yield " ".join(current).strip()
            current.clear()

    step_open_re = re.compile(r"^(\d+)\.\s+(.+)$")
    for raw in block.splitlines():
        if not raw.strip():
            yield from _flush()
            continue
        m = step_open_re.match(raw.lstrip())
        if m and not raw.startswith((" ", "\t")):
            yield from _flush()
            current.append(m.group(2).strip())
        elif current and (raw.startswith((" ", "\t")) or raw.lstrip().startswith("- ")):
            current.append(raw.strip().lstrip("- ").strip())
        # else: drop top-level non-numbered prose silently (matches v0.1.0 contract)
    yield from _flush()
