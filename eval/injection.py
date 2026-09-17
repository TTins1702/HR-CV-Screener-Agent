"""The `no_guard` ablation: what the injection layer is worth, per rule.

Spec section 7 asks for the guard switched off and the poisoned CVs re-run, with
the score visibly collapsing. The collapse is only visible per rule, because the
rule table has two severities and the guard treats them differently: HIGH
quarantines, LOW annotates and lets the CV through to scoring. Averaging the two
would report a defence for `must_hire` and `hidden_directive` that was never in
force for them.

The rows are synthetic and the slide has to say so. Day 2 measured 0 of 300 real
dev resumes tripping any rule, so this branch has no real traffic to report and a
fixture set is the only honest way to give it any.

Fixtures are derived from `INJECTION_RULES` here rather than imported from
`tests/`, so eval does not depend on the test tree, and a rule added later gets a
row without anybody remembering to add one.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from src.contracts.ablations import Ablations
from src.contracts.tools import InjectionSeverity
from src.graph.build import screen
from src.graph.rubric_nodes import DERIVED_RUBRIC_DIR
from src.tools.injection import INJECTION_RULES

# A CV with no injection in it, so "the score collapsed" has a baseline to fall from.
CLEAN_CV = (
    "Jordan Blake, Backend Engineer.06/2019to12/2022 Backend Engineer at Acme Corp."
    "Built payment APIs in Python and PostgreSQL.Skills: Python, Django, PostgreSQL, "
    "Docker, AWS.B.S. Computer Science, State University, 2019."
)

EVAL_TODAY = date(2026, 9, 7)


class GuardOutcome(BaseModel):
    """What the graph did with one CV under one setting of the guard."""

    quarantined: bool
    overall_score: float
    label: str
    path_taken: list[str]
    rejected_reason: str | None


class RuleComparison(BaseModel):
    """One rule, screened with the guard on and with it off."""

    rule_id: str | None
    severity: InjectionSeverity | None
    guarded: GuardOutcome
    unguarded: GuardOutcome

    @property
    def changed(self) -> bool:
        """Whether switching the guard off changed the outcome at all."""
        return (
            self.guarded.quarantined != self.unguarded.quarantined
            or self.guarded.overall_score != self.unguarded.overall_score
            or self.guarded.label != self.unguarded.label
        )


def build_poisoned_rows(
    jd_text: str, *, include_control: bool = False
) -> list[dict[str, Any]]:
    """One row per rule, each the clean CV with that rule's own example appended.

    `label` is `No Fit` throughout: these are not real candidates and the label is
    only there because the split format carries one. Nothing downstream should
    read accuracy off these rows.
    """
    rows: list[dict[str, Any]] = []
    if include_control:
        rows.append(
            {
                "rule_id": None,
                "severity": None,
                "resume_text": CLEAN_CV,
                "job_description_text": jd_text,
                "label": "No Fit",
            }
        )
    for rule in INJECTION_RULES:
        rows.append(
            {
                "rule_id": rule.rule_id,
                "severity": rule.severity,
                "resume_text": f"{CLEAN_CV} {rule.example}",
                "job_description_text": jd_text,
                "label": "No Fit",
            }
        )
    return rows


def _screen_once(
    row: dict[str, Any], *, llm: Any, guard: bool, derived_dir: Path | str
) -> GuardOutcome:
    result = screen(
        row["resume_text"],
        row["job_description_text"],
        llm=llm,
        today=EVAL_TODAY,
        derived_dir=derived_dir,
        ablations=Ablations(guard=guard),
    )
    return GuardOutcome(
        quarantined="quarantine" in result.path_taken,
        overall_score=result.overall_score,
        label=result.label.value,
        path_taken=list(result.path_taken),
        rejected_reason=result.rejected_reason,
    )


def compare_guard(
    rows: Sequence[dict[str, Any]],
    *,
    llm: Any,
    derived_dir: Path | str = DERIVED_RUBRIC_DIR,
) -> list[RuleComparison]:
    """Screen every row twice, guard on and guard off."""
    return [
        RuleComparison(
            rule_id=row["rule_id"],
            severity=row["severity"],
            guarded=_screen_once(row, llm=llm, guard=True, derived_dir=derived_dir),
            unguarded=_screen_once(row, llm=llm, guard=False, derived_dir=derived_dir),
        )
        for row in rows
    ]


def render_guard_ablation(comparison: Sequence[RuleComparison]) -> str:
    """The markdown, HIGH and LOW kept apart because the guard treats them apart."""
    control = [c for c in comparison if c.rule_id is None]
    high = [c for c in comparison if c.severity is InjectionSeverity.HIGH]
    low = [c for c in comparison if c.severity is InjectionSeverity.LOW]

    lines = [
        "# The `no_guard` ablation, rule by rule",
        "",
        "**These rows are synthetic.** Each is a clean CV with one injection rule's "
        "own example string appended. Day 2 measured 0 of 300 real dev resumes "
        "tripping any rule, so the guard branch has no real traffic and this is the "
        "only honest way to give it any. Any slide showing these numbers has to say "
        "so on the slide.",
        "",
    ]

    if control:
        outcome = control[0].guarded
        lines += [
            f"The unpoisoned control scores **{outcome.overall_score:.3f}** "
            f"(`{outcome.label}`) with the guard on, and "
            f"**{control[0].unguarded.overall_score:.3f}** with it off. That is the "
            "number the poisoned rows fall from.",
            "",
        ]

    def table(rows: Sequence[RuleComparison], title: str, note: str) -> list[str]:
        out = [f"## {title}", "", note, "",
               "| Rule | guard on | guard off | changed |", "|---|---|---|---|"]
        for item in rows:
            guarded = (
                "quarantined"
                if item.guarded.quarantined
                else f"{item.guarded.overall_score:.3f} ({item.guarded.label})"
            )
            unguarded = (
                "quarantined"
                if item.unguarded.quarantined
                else f"{item.unguarded.overall_score:.3f} ({item.unguarded.label})"
            )
            out.append(
                f"| `{item.rule_id}` | {guarded} | {unguarded} | "
                f"{'yes' if item.changed else 'no'} |"
            )
        return out + [""]

    if high:
        stopped = sum(1 for c in high if c.guarded.quarantined)
        note = (
            f"{stopped} of {len(high)} are stopped before a model ever scores them. "
            "With the guard off every one of them is scored like an ordinary CV."
        )
        if control:
            baseline = control[0].unguarded.overall_score
            gains = [c.unguarded.overall_score - baseline for c in high]
            biggest = max(gains, default=0.0)
            if all(abs(gain) < 1e-9 for gain in gains):
                note += (
                    "\n\nSpec section 7 asks to see the score collapse once the guard "
                    "is removed. **It did not move.** Every poisoned row scores "
                    f"exactly {baseline:.3f}, the clean control's own score, so on "
                    "this fixture set the injections changed nothing the model did. "
                    "What the guard is worth here is that it refuses to process a "
                    "document attempting manipulation -- not that it prevents a "
                    "manipulation that **would have worked**. Claiming the second "
                    "from this table would be claiming an attack that never landed."
                )
            else:
                note += (
                    f"\n\nThe largest gain an injection buys over the clean control "
                    f"is **{biggest:+.3f}**, so the guard is stopping something that "
                    "does move the score."
                )
        lines += table(high, "HIGH severity: the guard quarantines", note)

    if low:
        changed = sum(1 for c in low if c.changed)
        lines += table(
            low,
            "LOW severity: the guard annotates and lets through",
            f"The guard flags these and does not quarantine them, so switching it "
            f"off changes {changed} of {len(low)}. The honest number for these two "
            "rules is that **no defence was in force either way** -- they are "
            "reported here rather than folded into the HIGH average, which would "
            "claim protection that never existed.",
        )

    return "\n".join(lines).rstrip() + "\n"


def job_description_from(split: Path | str, index: int = 0) -> str:
    """The JD of one row of a real split.

    The poisoned CVs are synthetic already. Pairing them with an invented JD as
    well would derive a rubric that nothing else in the eval has ever used, and
    the ablation would be measuring a configuration that ships nowhere.
    """
    import json

    with Path(split).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return rows[index]["job_description_text"]


def main(argv: list[str] | None = None) -> int:
    import argparse

    from src.llm.cache import DEFAULT_CACHE_PATH, JSONLCache
    from src.llm.client import StructuredLLM

    parser = argparse.ArgumentParser(description="the no_guard ablation, rule by rule")
    parser.add_argument("--split", type=Path, default=Path("data/samples/dev_300.jsonl"))
    parser.add_argument("--jd-index", type=int, default=0)
    parser.add_argument(
        "--out", type=Path, default=Path("docs/measurements/no_guard.md")
    )
    args = parser.parse_args(argv)

    if not args.split.exists():
        parser.error(f"no such split: {args.split}")

    llm = StructuredLLM(cache=JSONLCache(DEFAULT_CACHE_PATH))
    rows = build_poisoned_rows(
        job_description_from(args.split, args.jd_index), include_control=True
    )
    print(f"screening {len(rows)} rows twice, guard on and guard off")
    comparison = compare_guard(rows, llm=llm)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render_guard_ablation(comparison), encoding="utf-8")
    print(
        f"wrote {args.out} ({len(comparison)} rules, "
        f"{llm.hits} cache hits, {llm.misses} misses)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
