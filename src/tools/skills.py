"""Resolve skill strings to canonical names so scoring compares concepts, not spelling.

`React`, `ReactJS` and `React.js` are one skill; a criterion asking for React
must not miss a CV that spells it differently. The alias table lives in
`data/skills/aliases.yaml` so a recruiter can extend it without touching code,
the same arrangement as `data/rubrics/`.

Two tables come out of that one file, and the split matters. The *lookup* table
keys on the canonical form, which strips spaces -- correct for comparison, since
it makes `React.js` and `reactjs` one skill. The *surface-form* table keeps the
strings a human actually wrote, because those become `search_evidence` queries:
searching a CV for `"reactnative"` finds nothing in a CV that says
`"React Native"`. Collapsing the two would silently lose every multi-word skill.

`expand_skill` is the bridge to `search_evidence`: the scoring node expands a
criterion's skill into every surface form, then searches the CV for each one, so
the evidence quote is whatever the candidate actually wrote.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

from src.contracts.tools import SkillMatch

DEFAULT_ALIAS_PATH = Path("data/skills/aliases.yaml")

# Characters that distinguish one language from another and must survive
# canonicalization: C# is not C, C++ is not C.
_KEPT_PUNCTUATION = "#+"
_STRIP_RE = re.compile(rf"[^0-9a-z{re.escape(_KEPT_PUNCTUATION)}]+")


def canonical_form(raw: str) -> str:
    """Comparison key for a skill string: lowercase, no spaces, no punctuation.

    `#` and `+` are kept so `c#` and `c++` stay distinct from `c`.
    """
    return _STRIP_RE.sub("", raw.strip().lower())


def load_skill_aliases(path: str | Path = DEFAULT_ALIAS_PATH) -> dict[str, str]:
    """The lookup table: canonical-form-of-any-surface-form -> skill name.

    Cached per path, so the returned mapping is shared. Treat it as read-only.
    """
    return _load_tables(Path(path))[0]


def load_skill_surface_forms(path: str | Path = DEFAULT_ALIAS_PATH) -> dict[str, list[str]]:
    """The search table: skill name -> its surface forms, verbatim from the YAML.

    Verbatim because these strings become `search_evidence` queries. The lookup
    table's canonical key for "react native" is "reactnative", which matches no
    CV; the string a human wrote does.
    """
    return _load_tables(Path(path))[1]


@lru_cache(maxsize=8)
def _load_tables(path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Parse the YAML once into the lookup table and the surface-form table."""
    if not path.exists():
        raise FileNotFoundError(f"skill alias table not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"alias table must be a mapping, got {type(raw).__name__}")

    lookup: dict[str, str] = {}
    surface: dict[str, list[str]] = {}
    for skill, aliases in raw.items():
        name = str(skill)
        forms: list[str] = []
        for form in [name, *(aliases or [])]:
            text = str(form).strip()
            key = canonical_form(text)
            if not key:
                raise ValueError(f"empty surface form listed under {name!r}")
            claimed_by = lookup.get(key)
            if claimed_by is not None and claimed_by != name:
                raise ValueError(
                    f"surface form {text!r} is claimed by both {claimed_by!r} and {name!r}"
                )
            lookup[key] = name
            if text not in forms:
                forms.append(text)
        surface[name] = forms
    return lookup, surface


def normalize_skill(raw: str, aliases: dict[str, str] | None = None) -> SkillMatch:
    """Resolve `raw` to a canonical skill name.

    An unknown skill is not an error: its canonical form becomes its name and
    `known` is False, so a CV listing something the table has never heard of is
    still comparable against another CV that lists the same thing.
    """
    key = canonical_form(raw)
    if not key:
        raise ValueError(f"skill must contain at least one alphanumeric character: {raw!r}")

    table = load_skill_aliases() if aliases is None else aliases
    skill = table.get(key)
    if skill is None:
        return SkillMatch(raw=raw, canonical=key, matched_alias=None, known=False)
    return SkillMatch(raw=raw, canonical=skill, matched_alias=key, known=True)


def expand_skill(skill: str, path: str | Path = DEFAULT_ALIAS_PATH) -> list[str]:
    """Every verbatim surface form of `skill`, longest first, for `search_evidence`.

    Longest first so a search tries "react native" before the "react" it
    contains. An unknown skill expands to itself, so a criterion naming
    something the table has never heard of is still searchable.
    """
    match = normalize_skill(skill, load_skill_aliases(path))
    if not match.known:
        return [skill.strip()]
    forms = load_skill_surface_forms(path)[match.canonical]
    return sorted(forms, key=lambda form: (-len(form), form))


def skills_match(left: str, right: str, aliases: dict[str, str] | None = None) -> bool:
    """True when two skill strings name the same skill."""
    table = load_skill_aliases() if aliases is None else aliases
    return normalize_skill(left, table).canonical == normalize_skill(right, table).canonical
