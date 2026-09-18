"""Builds the per-node walkthrough slides behind the trace carousel.

Every mark is located by a tool the graph itself uses -- `scan_injection`,
`search_evidence`, `calculate_experience` -- never by a string search written
here. On screen a highlight the interface invented looks exactly like evidence
the system actually found, so the difference has to be structural: each mark
carries the `locator` that produced it, and the UI shows it.
"""

from __future__ import annotations

from typing import Any, Callable

from pydantic import BaseModel, Field

from src.tools.injection import scan_injection

#: Shown on an output row whose value the locators could not place in the text.
#: Dropping the row instead would make extraction look cleaner than it is.
UNLOCATED_NOTE = "không định vị được trên CV"

NODE_SUMMARIES: dict[str, str] = {
    "ingest": "Nhận cặp CV/JD nguyên văn vào graph. Không chuẩn hóa, không cắt gọt: mọi vị trí ký tự của evidence về sau đều tính trên đúng chuỗi này.",
    "guard": "Quét CV tìm chỉ thị ẩn nhắm vào model chấm điểm, bằng các luật tất định. Chỉ mức HIGH mới cách ly tài liệu.",
    "quarantine": "Từ chối xử lý tài liệu đang cố can thiệp vào model. Nhãn trả về là No Fit, lý do thật nằm ở rejected_reason.",
    "extract": "Đọc CV thô, trả về hồ sơ có cấu trúc: kỹ năng, các giai đoạn làm việc, bằng cấp, chứng chỉ.",
    "repair": "Vá lại các trường hồ sơ vi phạm ràng buộc schema, rồi cho chạy lại thay vì bỏ cả hồ sơ.",
    "load_rubric": "Nạp bộ tiêu chí chấm điểm, hoặc suy luận bộ tiêu chí từ JD nếu không có preset.",
    "must_have_check": "Cổng tiên quyết trước khi chấm: chỉ chặn những must-have có thể phán được mà không cần điểm.",
    "reject_fast": "Loại sớm hồ sơ thiếu yêu cầu bắt buộc, không gọi model chấm điểm.",
    "score_criteria": "Chấm từng tiêu chí, mỗi điểm phải kèm trích dẫn nguyên văn cắt ra từ CV.",
    "aggregate": "Cộng điểm theo trọng số thành một điểm tổng, và kiểm tra xem nó có rơi vào vùng xám không.",
    "deep_review": "Chấm lại các tiêu chí của hồ sơ nằm sát ngưỡng, trước khi chốt nhãn.",
    "decide": "Đối chiếu điểm tổng với hai ngưỡng của rubric để ra nhãn cuối cùng.",
    "rank": "Chốt kết quả và tổng kết đường đi mà lần chạy này đã qua.",
}

_DEFAULT_SUMMARY = "Node của graph trong lần chạy này."
_END_CAPTION = "kết thúc"


class Mark(BaseModel):
    """One span of a document, and the tool that found it."""

    id: int
    doc: str
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    label: str
    locator: str
    score: float = Field(default=1.0, ge=0.0, le=1.0)


class OutputRow(BaseModel):
    """One thing the node produced.

    `note` is a problem the reader should see; `detail` is the node's own
    explanation. Keeping them apart stops a failure to locate a value from
    reading like commentary.
    """

    field: str
    value: str
    mark_ids: list[int] = Field(default_factory=list)
    note: str | None = None
    detail: str | None = None


class Slide(BaseModel):
    """One node's walkthrough."""

    node: str
    documents: list[str] = Field(default_factory=list)
    caption: str
    summary: str
    marks: list[Mark] = Field(default_factory=list)
    outputs: list[OutputRow] = Field(default_factory=list)


class _Builder:
    """Per-slide state: the documents, and a mark-id counter scoped to the slide."""

    def __init__(self, cv_text: str, jd_text: str, snapshot: dict[str, Any]):
        self.cv_text = cv_text
        self.jd_text = jd_text
        self.snapshot = snapshot
        self.marks: list[Mark] = []
        self._next_id = 1

    def add_mark(
        self, doc: str, start: int, end: int, label: str, locator: str, score: float = 1.0
    ) -> int:
        """Record a mark and return its id, or 0 if the span is not usable."""
        source = self.cv_text if doc == "cv" else self.jd_text
        if not (0 <= start < end <= len(source)):
            return 0
        mark_id = self._next_id
        self._next_id += 1
        self.marks.append(
            Mark(
                id=mark_id,
                doc=doc,
                start=start,
                end=end,
                label=label,
                locator=locator,
                score=score,
            )
        )
        return mark_id


def _ingest(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    """`ingest` marks nothing: not touching the text is the whole point of it."""
    empty = not builder.cv_text.strip() or not builder.jd_text.strip()
    return ["cv", "jd"], [
        OutputRow(field="cv_text", value=f"{len(builder.cv_text)} ký tự"),
        OutputRow(field="jd_text", value=f"{len(builder.jd_text)} ký tự"),
        OutputRow(
            field="tài liệu rỗng",
            value="có" if empty else "không",
            detail=(
                "CV hoặc JD rỗng sẽ bị cách ly ngay, không ném lỗi." if empty else None
            ),
        ),
    ]


def _guard(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    """Re-run the guard's own scanner.

    The streamed payload only carries findings for a run that was quarantined, so
    reading it would show nothing for the LOW-severity rules that fired without
    blocking -- which are exactly the ones worth showing a recruiter.
    """
    report = scan_injection(builder.cv_text)
    outputs = [OutputRow(field="severity", value=report.severity.value)]

    for finding in report.findings:
        mark_id = builder.add_mark(
            "cv",
            finding.evidence.start,
            finding.evidence.end,
            finding.rule_id,
            "scan_injection",
            finding.evidence.score,
        )
        outputs.append(
            OutputRow(
                field=finding.rule_id,
                value=finding.severity.value,
                mark_ids=[mark_id] if mark_id else [],
                note=None if mark_id else UNLOCATED_NOTE,
            )
        )

    if not report.findings:
        outputs.append(
            OutputRow(
                field="phát hiện",
                value="0",
                detail="Không luật nào khớp. CV đi thẳng sang extract.",
            )
        )
    return ["cv"], outputs


_BUILDERS: dict[str, Callable[[_Builder], tuple[list[str], list[OutputRow]]]] = {
    "ingest": _ingest,
    "guard": _guard,
}


def build_slides(
    cv_text: str, jd_text: str, snapshots: list[dict[str, Any]]
) -> list[Slide]:
    """One slide per executed node, in the order the run executed them.

    A repeated node gets a slide per pass: `repair` can run twice, and folding the
    two together would hide a whole pass of the loop.
    """
    slides: list[Slide] = []
    for index, snapshot in enumerate(snapshots):
        node = snapshot.get("node") or snapshot.get("current_node") or ""
        if not node:
            continue

        following = snapshots[index + 1] if index + 1 < len(snapshots) else None
        next_node = (
            (following.get("node") or following.get("current_node")) if following else None
        )
        caption = f"{node} → {next_node or _END_CAPTION}"

        builder = _Builder(cv_text, jd_text, snapshot)
        make = _BUILDERS.get(node)
        documents, outputs = make(builder) if make else ([], [])

        slides.append(
            Slide(
                node=node,
                documents=documents,
                caption=caption,
                summary=NODE_SUMMARIES.get(node, _DEFAULT_SUMMARY),
                marks=builder.marks,
                outputs=outputs,
            )
        )
    return slides
