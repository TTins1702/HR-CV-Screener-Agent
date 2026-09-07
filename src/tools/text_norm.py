"""Normalize CV text for matching while keeping a way back to the original offsets.

The scored dataset has whitespace stripped between sentences: 286 of the 300 dev
resumes contain at least one glued boundary like `"projects.Proven"` or
`"12/2011toPresentData Analyst"`. Matching a quote against the raw text therefore
fails on any span crossing such a boundary. Every matching tool works on the
normalized copy instead, then maps the hit back so the evidence it returns is a
verbatim slice of the CV the recruiter is looking at.

Three boundaries get a space inserted:
  * punctuation followed by an alphanumeric   -- "projects.Proven"
  * a lowercase letter followed by a capital  -- "SkillsPython"
  * a digit followed by a capital             -- "2011Present"
"""

from __future__ import annotations

#: A space is inserted after one of these when an alphanumeric follows it.
#: `|` is deliberately absent: it would split the chat-template control tokens
#: `scan_injection` looks for (`<|im_start|>` -> `<| im_start|>`) and whitespace
#: collapsing already handles `Python | Java` bullet separators.
_BOUNDARY_PUNCTUATION = frozenset(".,;:!?)]}•")


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Return a normalized copy of `text` and a map from its indices back into `text`.

    The normalized copy is lowercased, has every whitespace run collapsed to one
    space, has leading whitespace dropped, and has a space inserted at each glued
    boundary. `index_map[i]` is the offset in `text` of the character that
    produced `normalized[i]`; `index_map[len(normalized)]` is `len(text)`, so
    `text[index_map[i]:index_map[j]]` is always a valid slice.
    """
    characters: list[str] = []
    index_map: list[int] = []
    previous_kept = ""
    position = 0
    length = len(text)

    while position < length:
        character = text[position]

        if character.isspace():
            run_end = position
            while run_end < length and text[run_end].isspace():
                run_end += 1
            if characters and characters[-1] != " ":
                characters.append(" ")
                index_map.append(position)
            previous_kept = " "
            position = run_end
            continue

        if characters and characters[-1] != " " and _is_glued_boundary(previous_kept, character):
            characters.append(" ")
            index_map.append(position)

        characters.append(character.lower())
        index_map.append(position)
        previous_kept = character
        position += 1

    index_map.append(length)
    return "".join(characters), index_map


def normalize(text: str) -> str:
    """The normalized text alone, for callers that do not need the offsets."""
    return normalize_with_map(text)[0]


def _is_glued_boundary(previous: str, current: str) -> bool:
    """True when the dataset has dropped a space between `previous` and `current`."""
    if previous in _BOUNDARY_PUNCTUATION and current.isalnum():
        return True
    if previous.islower() and current.isupper():
        return True
    return previous.isdigit() and current.isupper()
