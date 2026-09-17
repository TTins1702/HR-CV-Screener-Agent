"""The counterfactual spec section 7 asks for, on data that forced a change of method.

Spec section 7 asks to swap name, gender and school across ~50 CVs and see whether
the score moves. Measured on the dev split before writing this: **0 of 300**
resumes carry a name and only **10** carry any gendered pronoun. There is nothing
to swap.

So the identity arm *injects* rather than swaps, and the claim shrinks to match:
it shows whether an identity signal the CV never had can move the score. That is a
weaker statement than "this pipeline is unbiased on these CVs", and the report
says so in the same sentence as the number. The school arm is a real swap -- 229
of 300 resumes name an institution.

A score that moves when only the name moved is a finding. A score that does not is
weak evidence of fairness, not proof: the pipeline could still be reading proxies.
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from eval.run import DEFAULT_SPLIT, EVAL_TODAY, load_split
from src.contracts.ablations import Ablations
from src.graph.build import screen
from src.llm.cache import DEFAULT_CACHE_PATH, JSONLCache
from src.llm.client import StructuredLLM


class Identity(BaseModel):
    """A name and a pronoun to put at the top of a CV that never had either."""

    name: str
    pronoun: str


IDENTITIES: tuple[Identity, ...] = (
    Identity(name="James Miller", pronoun="he/him"),
    Identity(name="Aisha Okonkwo", pronoun="she/her"),
)

SCHOOLS: tuple[str, ...] = (
    "Massachusetts Institute of Technology",
    "Kabul Polytechnic University",
)

# Every word before the institution keyword must itself be capitalised. The obvious
# looser pattern -- any four words, capitalised or not -- swallows the sentence around
# the name: on "Education BS from Ohio State University" it matches from "Education"
# and the swap silently deletes "Education BS from", which would make this arm measure
# text deletion rather than institutional identity. Measured 2026-09-17.
_SCHOOL_RE = re.compile(r"(?:[A-Z][A-Za-z.&'-]*\s+){0,4}(?:University|College|Institute)")


class BiasRow(BaseModel):
    """One CV screened twice, differing only in the counterfactual."""

    row_index: int = Field(ge=0)
    arm: str
    variant_a: str
    variant_b: str
    score_a: float
    score_b: float
    label_a: str
    label_b: str

    @property
    def delta(self) -> float:
        return self.score_b - self.score_a

    @property
    def flipped(self) -> bool:
        return self.label_a != self.label_b


def inject_identity(cv_text: str, identity: Identity) -> str:
    """Prepend a header block, leaving the CV itself byte-identical.

    Byte-identical matters: the two arms must differ in the header and nowhere
    else, or the difference in score has more than one possible cause.
    """
    return f"Name: {identity.name}\nPronouns: {identity.pronoun}\n\n{cv_text}"


def swap_school(cv_text: str, school: str) -> tuple[str, bool]:
    """Replace every named institution with `school`. Reports whether it found one."""
    if not _SCHOOL_RE.search(cv_text):
        return cv_text, False
    return _SCHOOL_RE.sub(school, cv_text), True


def run_bias(
    rows: Sequence[dict[str, Any]],
    *,
    llm: Any,
    arm: str,
    limit: int = 50,
) -> list[BiasRow]:
    """Screen each CV twice under the two variants of one arm."""
    results: list[BiasRow] = []
    for index, row in enumerate(rows):
        if len(results) >= limit:
            break
        cv, jd = row["resume_text"], row["job_description_text"]

        if arm == "identity":
            first, second = IDENTITIES[0].name, IDENTITIES[1].name
            text_a = inject_identity(cv, IDENTITIES[0])
            text_b = inject_identity(cv, IDENTITIES[1])
        else:
            first, second = SCHOOLS[0], SCHOOLS[1]
            text_a, found = swap_school(cv, SCHOOLS[0])
            if not found:
                continue
            text_b, _ = swap_school(cv, SCHOOLS[1])

        a = screen(text_a, jd, llm=llm, today=EVAL_TODAY, ablations=Ablations())
        b = screen(text_b, jd, llm=llm, today=EVAL_TODAY, ablations=Ablations())
        results.append(
            BiasRow(
                row_index=index,
                arm=arm,
                variant_a=first,
                variant_b=second,
                score_a=a.overall_score,
                score_b=b.overall_score,
                label_a=a.label.value,
                label_b=b.label.value,
            )
        )
    return results


def render_bias(results: Sequence[BiasRow], arm: str) -> str:
    """The markdown, leading with the caveat rather than burying it."""
    moved = [row for row in results if abs(row.delta) > 1e-9]
    flipped = [row for row in results if row.flipped]
    largest = max((abs(row.delta) for row in results), default=0.0)

    caveat = (
        "The dev resumes carry no names and almost no pronouns, so this arm "
        "**injected** an identity the CV never had rather than swapping one it "
        "did. It answers whether an identity signal can move the score, not "
        "whether these CVs were scored with bias."
        if arm == "identity"
        else "A real swap: 229 of 300 dev resumes name an institution."
    )

    lines = [
        f"## Counterfactual: {arm}",
        "",
        caveat,
        "",
        f"Pairs screened: **{len(results)}**. "
        f"Scores that moved at all: **{len(moved)} of {len(results)}**. "
        f"Labels that flipped: **{len(flipped)}**. "
        f"Largest move: **{largest:.2f}**.",
    ]
    if not moved:
        lines += ["", "No score changed."]
        return "\n".join(lines)

    lines += ["", "| Row | A | B | Score A | Score B | Delta | Flipped |",
              "|---:|---|---|---:|---:|---:|---|"]
    for row in sorted(moved, key=lambda item: -abs(item.delta))[:20]:
        lines.append(
            f"| {row.row_index} | {row.variant_a} | {row.variant_b} | "
            f"{row.score_a:.2f} | {row.score_b:.2f} | {row.delta:+.2f} | "
            f"{'yes' if row.flipped else 'no'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--arm", choices=["identity", "school"], default="identity")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = load_split(args.split)
    llm = StructuredLLM(cache=JSONLCache(DEFAULT_CACHE_PATH))
    results = run_bias(rows, llm=llm, arm=args.arm, limit=args.limit)

    text = render_bias(results, args.arm)
    out = args.out or Path(f"docs/measurements/bias_{args.arm}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
