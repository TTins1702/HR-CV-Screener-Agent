"""Write the Mermaid source for slide 1, straight out of the compiled graph.

Spec section 9 asks that the diagram be generated from the graph rather than drawn
by hand, so it cannot drift from the code. Conditional edges come out dotted
(`-.->`), which is the distinction the slide colours.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.graph.build import graph_mermaid

DEFAULT_OUT = Path("docs/diagrams/graph.mmd")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    diagram = graph_mermaid()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(diagram, encoding="utf-8")
    print(f"wrote {args.out} ({len(diagram.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
