"""Synthetic CVs carrying prompt injections.

The `guard` branch carries zero traffic on real data -- Day 2 measured 0 of 300 dev
resumes tripping any rule -- so its traffic has to come from a fixture set, and the
slide must label it as synthetic. Building the fixtures from the rule table's own
`example` strings means a rule added later automatically gets a fixture.
"""

from __future__ import annotations

from src.contracts.tools import InjectionSeverity
from src.tools.injection import INJECTION_RULES

CLEAN_CV = (
    "Jordan Blake, Backend Engineer.06/2019to12/2022 Backend Engineer at Acme Corp."
    "Built payment APIs in Python and PostgreSQL.Skills: Python, Django, PostgreSQL, "
    "Docker, AWS.B.S. Computer Science, State University, 2019."
)


def poisoned_cvs(severity: InjectionSeverity | None = None) -> list[tuple[str, str]]:
    """`(rule_id, cv_text)` for every rule, or only those of `severity`."""
    return [
        (rule.rule_id, f"{CLEAN_CV} {rule.example}")
        for rule in INJECTION_RULES
        if severity is None or rule.severity is severity
    ]
