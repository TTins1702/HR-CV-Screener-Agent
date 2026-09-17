"""Which of the graph's defences are switched on for one run.

Spec section 7 asks for ablations: turn a branch off, re-run, and show what the
number does. Putting the three switches in one frozen object -- rather than
scattering them across rubric fields, environment variables and code edits --
means an eval run names its own configuration (`Ablations.label`), and means the
difference between two runs is attributable to exactly one branch.

This rides on `ScreeningState`, not on `build_graph`, because spec section 4 says
one pydantic model flows through the graph and "which defences are on" is a fact
about the run rather than about the wiring. It also keeps `src/graph/routes.py`
four pure functions of state, which is the property that makes the control flow
reviewable in one screen.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

_SWITCHES = ("must_have_gate", "guard", "gray_zone")


class Ablations(BaseModel):
    """Defences on or off. The default -- everything on -- is what ships."""

    model_config = ConfigDict(frozen=True)

    must_have_gate: bool = True
    guard: bool = True
    gray_zone: bool = True

    @property
    def label(self) -> str:
        """A short name for this configuration, used in filenames and headings."""
        off = [name for name in _SWITCHES if not getattr(self, name)]
        return "shipped" if not off else "+".join(f"no_{name}" for name in off)


SHIPPED = Ablations()
