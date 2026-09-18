"""Builds the per-node walkthrough slides behind the trace carousel.

Every mark is located by a tool the graph itself uses -- `scan_injection`,
`search_evidence`, `calculate_experience` -- never by a string search written
here. On screen a highlight the interface invented looks exactly like evidence
the system actually found, so the difference has to be structural: each mark
carries the `locator` that produced it, and the UI shows it.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable

from pydantic import BaseModel, Field

from src.contracts.rubric import Criterion, JDRubric
from src.contracts.screening import CandidateProfile, CriterionScore
from src.contracts.state import ScreeningState
from src.graph.rubric_nodes import blocking_must_haves, required_years
from src.tools.evidence import search_evidence
from src.tools.experience import calculate_experience
from src.contracts.tools import InjectionSeverity, Scorecard
from src.tools.injection import scan_injection
from src.tools.skills import expand_skill

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

    def __init__(
        self,
        cv_text: str,
        jd_text: str,
        snapshot: dict[str, Any],
        previous: dict[str, Any] | None = None,
    ):
        self.cv_text = cv_text
        self.jd_text = jd_text
        self.snapshot = snapshot
        #: The step before this one. `repair` and `deep_review` are diffs, and a
        #: diff against the final state would always report no change.
        self.previous = previous
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


def _locate(builder: _Builder, doc: str, query: str, label: str) -> int:
    """Mark the best span matching `query`, or return 0 if there is none."""
    source = builder.cv_text if doc == "cv" else builder.jd_text
    found = search_evidence(source, query, max_results=1)
    if not found:
        return 0
    best = found[0]
    return builder.add_mark(doc, best.start, best.end, label, "search_evidence", best.score)


def _row(field: str, value: str, mark_id: int) -> OutputRow:
    """An output row that says so when its value could not be placed in the text."""
    return OutputRow(
        field=field,
        value=value,
        mark_ids=[mark_id] if mark_id else [],
        note=None if mark_id else UNLOCATED_NOTE,
    )


def _extract(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    raw = builder.snapshot.get("profile")
    if not raw:
        return ["cv"], [
            OutputRow(
                field="profile",
                value="chưa có",
                detail="Node chưa trả về hồ sơ có cấu trúc trong bước này.",
            )
        ]

    profile = CandidateProfile.model_validate(raw)
    outputs: list[OutputRow] = []

    for skill in profile.skills:
        mark_id = 0
        # Longest surface form first, so "react native" is tried before "react".
        for form in expand_skill(skill):
            mark_id = _locate(builder, "cv", form, "skills")
            if mark_id:
                break
        outputs.append(_row("skills", skill, mark_id))

    if profile.work_periods:
        report = calculate_experience(
            builder.cv_text, today=date.today(), work_periods=profile.work_periods
        )
        for item in report.ranges:
            if not item.is_employment:
                continue
            mark_id = builder.add_mark(
                "cv",
                item.source.start,
                item.source.end,
                "work_periods",
                "calculate_experience",
                item.source.score,
            )
            ending = "nay" if item.is_current else item.end.isoformat()
            outputs.append(
                _row("work_periods", f"{item.start.isoformat()} → {ending}", mark_id)
            )

    for degree in profile.degrees:
        outputs.append(_row("degrees", degree, _locate(builder, "cv", degree, "degrees")))

    for certification in profile.certifications:
        outputs.append(
            _row(
                "certifications",
                certification,
                _locate(builder, "cv", certification, "certifications"),
            )
        )

    # Two different claims about the same quantity: what the tool measured from
    # dated ranges, and what the model asserted. They are reported side by side
    # because the gap between them is the tool's measurable contribution.
    outputs.append(
        OutputRow(
            field="total_experience_years",
            value=(
                f"{profile.total_experience_years:.2f}"
                if profile.total_experience_years is not None
                else "—"
            ),
            detail="do calculate_experience đo từ các khoảng ngày trên CV",
        )
    )
    outputs.append(
        OutputRow(
            field="llm_declared_years",
            value=(
                f"{profile.llm_declared_years:.2f}"
                if profile.llm_declared_years is not None
                else "—"
            ),
            detail="do model tự khai, không dùng để chấm",
        )
    )
    outputs.append(
        OutputRow(field="extraction_confidence", value=f"{profile.extraction_confidence:.2f}")
    )
    if profile.missing_fields:
        outputs.append(
            OutputRow(field="missing_fields", value=", ".join(profile.missing_fields))
        )

    return ["cv"], outputs


def _score_criteria(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    """Marks come straight from the scores: this evidence already carries offsets.

    `locator="criterion_evidence"` keeps it apart from a span the walkthrough had
    to go looking for. The two are different kinds of claim.
    """
    raw_scores = builder.snapshot.get("criterion_scores") or []
    if not raw_scores:
        return ["cv"], [
            OutputRow(
                field="criterion_scores",
                value="chưa có",
                detail="Node chưa chấm tiêu chí nào trong bước này.",
            )
        ]

    outputs: list[OutputRow] = []
    for raw in raw_scores:
        score = CriterionScore.model_validate(raw)
        mark_ids = [
            mark_id
            for evidence in score.evidence
            if (
                mark_id := builder.add_mark(
                    "cv",
                    evidence.start,
                    evidence.end,
                    score.criterion_id,
                    "criterion_evidence",
                    evidence.score,
                )
            )
        ]
        outputs.append(
            OutputRow(
                field=score.criterion_id,
                value=f"{score.score:.2f} · {score.tool_used or 'llm'}",
                mark_ids=mark_ids,
                note=None if mark_ids else UNLOCATED_NOTE,
                detail=score.reasoning or None,
            )
        )
    return ["cv"], outputs


def _rubric_of(builder: _Builder) -> JDRubric | None:
    raw = builder.snapshot.get("rubric")
    return JDRubric.model_validate(raw) if raw else None


def _mark_terms_in_jd(builder: _Builder, criterion: Criterion) -> list[int]:
    """Mark where the JD asks for this criterion.

    Skill terms first, because those are the exact strings the gate searches for.
    The bundled preset rubric names none at all, though -- its criteria carry only
    a description -- so the description is the fallback: it is still the wording
    this criterion was derived from, and `search_evidence` places it in the JD.
    """
    mark_ids = []
    for term in criterion.skill_terms:
        for surface in expand_skill(term):
            mark_id = _locate(builder, "jd", surface, criterion.id)
            if mark_id:
                mark_ids.append(mark_id)
                break
    if mark_ids:
        return mark_ids

    described = _locate(builder, "jd", criterion.description, criterion.id)
    return [described] if described else []


def _approximate_note(builder: _Builder, mark_ids: list[int]) -> str | None:
    """Say so when a mark was placed by similarity rather than an exact match.

    A paraphrase located at 0.86 looks identical on screen to a verbatim quote.
    The difference has to be written down, or the slide overstates its evidence.
    """
    scores = [m.score for m in builder.marks if m.id in mark_ids and m.score < 1.0]
    if not scores:
        return None
    return f"khớp gần đúng với JD ở mức {min(scores):.2f}, không phải trùng khít"


def _load_rubric(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    """The one node whose input is the JD, so the JD is what this slide shows."""
    rubric = _rubric_of(builder)
    if rubric is None:
        return ["jd"], [
            OutputRow(
                field="rubric",
                value="chưa có",
                detail="Node chưa nạp được bộ tiêu chí trong bước này.",
            )
        ]

    outputs = [
        OutputRow(
            field="job_title",
            value=rubric.job_title,
            detail="tên vị trí mà bộ tiêu chí này chấm",
        )
    ]
    for criterion in rubric.criteria:
        badge = " · MUST-HAVE" if criterion.must_have else ""
        mark_ids = _mark_terms_in_jd(builder, criterion)
        approximate = _approximate_note(builder, mark_ids)
        outputs.append(
            OutputRow(
                field=criterion.id,
                value=f"{round(criterion.weight * 100)}%{badge}",
                mark_ids=mark_ids,
                detail=(
                    f"{criterion.description} — {approximate}"
                    if approximate
                    else criterion.description
                ),
            )
        )
    outputs.append(
        OutputRow(
            field="good_fit_threshold",
            value=f"{rubric.good_fit_threshold:.2f}",
            detail="điểm tổng từ mức này trở lên là Good Fit",
        )
    )
    outputs.append(
        OutputRow(
            field="potential_fit_threshold",
            value=f"{rubric.potential_fit_threshold:.2f}",
            detail="dưới mức này là No Fit",
        )
    )
    return ["jd"], outputs


def _gate_state(builder: _Builder, rubric: JDRubric, profile: CandidateProfile) -> ScreeningState:
    """Rebuild just enough state to ask the gate its own question."""
    return ScreeningState(
        cv_text=builder.cv_text, jd_text=builder.jd_text, rubric=rubric, profile=profile
    )


def _must_have_check(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    """Both documents: the requirement comes from the JD, the answer from the CV.

    The verdict is `blocking_must_haves` itself, not a rule restated here. A gate
    explained by a second copy of its logic is an explanation of the copy.
    """
    rubric = _rubric_of(builder)
    raw_profile = builder.snapshot.get("profile")
    if rubric is None or not raw_profile:
        return ["cv", "jd"], [
            OutputRow(
                field="cổng must-have",
                value="không chạy được",
                detail="Thiếu bộ tiêu chí hoặc hồ sơ trích xuất, nên cổng bỏ qua.",
            )
        ]

    profile = CandidateProfile.model_validate(raw_profile)
    blocking = set(blocking_must_haves(_gate_state(builder, rubric, profile)))

    outputs: list[OutputRow] = []
    for criterion in rubric.must_haves():
        mark_ids = _mark_terms_in_jd(builder, criterion)
        failed = criterion.id in blocking

        # The gate judges exactly two shapes of criterion: a skill that names its
        # terms, and a year count that names a number. Everything else it leaves
        # to `score_criteria`. Treating "not blocked" as "passed" would credit the
        # gate with a verdict it never reached.
        if criterion.kind == "skill" and criterion.skill_terms:
            judged = True
            for term in criterion.skill_terms:
                for surface in expand_skill(term):
                    found = _locate(builder, "cv", surface, criterion.id)
                    if found:
                        mark_ids.append(found)
                        break
            detail = "cần một trong: " + ", ".join(criterion.skill_terms)
        elif criterion.kind == "experience_years":
            required = required_years(criterion.description)
            judged = required is not None
            have = max(
                profile.total_experience_years or 0.0, profile.llm_declared_years or 0.0
            )
            if judged:
                detail = f"cần {required:.1f} năm, hồ sơ có {have:.2f} năm"
                # Show the dated ranges the number was measured from, so the
                # comparison is not just two figures with nothing behind them.
                report = calculate_experience(
                    builder.cv_text,
                    today=date.today(),
                    work_periods=profile.work_periods,
                )
                for item in report.ranges:
                    if not item.is_employment:
                        continue
                    found = builder.add_mark(
                        "cv",
                        item.source.start,
                        item.source.end,
                        criterion.id,
                        "calculate_experience",
                        item.source.score,
                    )
                    if found:
                        mark_ids.append(found)
            else:
                detail = "mô tả không nêu số năm cụ thể, nên cổng để score_criteria chấm"
        else:
            judged = False
            detail = "cổng chỉ phán được kỹ năng có nêu từ khóa và số năm, nên tiêu chí này để score_criteria chấm"

        if failed:
            value, note = "thiếu", "chặn ở cổng must-have"
        elif judged:
            value, note = "đạt", None
        else:
            value, note = "không phán được", None

        approximate = _approximate_note(builder, mark_ids)
        outputs.append(
            OutputRow(
                field=criterion.id,
                value=value,
                mark_ids=mark_ids,
                note=note,
                detail=f"{detail} — {approximate}" if approximate else detail,
            )
        )

    if not outputs:
        outputs.append(
            OutputRow(
                field="must-have",
                value="0",
                detail="Bộ tiêu chí không đánh dấu tiêu chí nào là bắt buộc.",
            )
        )
    return ["cv", "jd"], outputs


def _reject_fast(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    """Terminal shortcut: no scoring call is made, which is where the saving is."""
    blocking = builder.snapshot.get("blocking_must_haves") or []
    rubric = _rubric_of(builder)
    by_id = {c.id: c for c in rubric.criteria} if rubric else {}

    outputs: list[OutputRow] = []
    for criterion_id in blocking:
        criterion = by_id.get(criterion_id)
        outputs.append(
            OutputRow(
                field=criterion_id,
                value="thiếu",
                mark_ids=_mark_terms_in_jd(builder, criterion) if criterion else [],
                note="đây là lý do hồ sơ bị loại sớm",
                detail=criterion.description if criterion else None,
            )
        )

    result = builder.snapshot.get("result") or {}
    outputs.append(
        OutputRow(
            field="rejected_reason",
            value=result.get("rejected_reason") or "—",
            detail="nhãn trả về là No Fit vì contract không có nhãn thứ tư",
        )
    )
    outputs.append(
        OutputRow(
            field="lượt gọi chấm điểm",
            value="0",
            detail="Cổng cắt trước score_criteria, nên không tốn token chấm điểm.",
        )
    )
    return ["cv", "jd"], outputs


def _quarantine(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    """Terminal node for a document the graph refuses to score."""
    report = scan_injection(builder.cv_text)
    outputs: list[OutputRow] = []

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
                note=(
                    "luật mức HIGH — đây là thứ kích hoạt cách ly"
                    if finding.severity is InjectionSeverity.HIGH
                    else None
                ),
            )
        )

    flags = builder.snapshot.get("injection_flags") or []
    if flags:
        outputs.append(OutputRow(field="injection_flags", value=", ".join(flags)))

    result = builder.snapshot.get("result") or {}
    outputs.append(
        OutputRow(
            field="rejected_reason",
            value=result.get("rejected_reason") or "—",
            detail="lý do thật của quyết định nằm ở đây",
        )
    )
    outputs.append(
        OutputRow(
            field="label",
            value=result.get("label") or "No Fit",
            detail=(
                "Contract chỉ có ba nhãn, không có nhãn thứ tư cho 'từ chối xử lý'. "
                "Đọc nhãn này mà không đọc rejected_reason là hiểu sai kết quả."
            ),
        )
    )
    return ["cv"], outputs


def _profile_fields(raw: dict[str, Any] | None) -> dict[str, str]:
    """The profile's scalar view, for diffing one pass against the next."""
    if not raw:
        return {}
    profile = CandidateProfile.model_validate(raw)
    return {
        "skills": ", ".join(profile.skills),
        "degrees": ", ".join(profile.degrees),
        "certifications": ", ".join(profile.certifications),
        "work_periods": str(len(profile.work_periods)),
        "total_experience_years": (
            f"{profile.total_experience_years:.2f}"
            if profile.total_experience_years is not None
            else "—"
        ),
        "extraction_confidence": f"{profile.extraction_confidence:.2f}",
        "missing_fields": ", ".join(profile.missing_fields) or "—",
    }


def _repair(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    """What this pass rewrote, against the pass before it.

    The comparison is why each node keeps its own snapshot: read off the final
    state, every repair slide would show the repaired profile on both sides and
    claim nothing changed.
    """
    after = _profile_fields(builder.snapshot.get("profile"))
    before = _profile_fields((builder.previous or {}).get("profile"))

    outputs: list[OutputRow] = []
    for field, new_value in after.items():
        old_value = before.get(field)
        if old_value is None or old_value == new_value:
            continue
        outputs.append(
            OutputRow(
                field=field,
                value=f"{old_value or '—'} → {new_value or '—'}",
                detail="trường này bị viết lại trong lượt vá",
            )
        )

    if not outputs:
        outputs.append(
            OutputRow(
                field="thay đổi",
                value="không có",
                detail=(
                    "Không so được với lượt trước."
                    if not before
                    else "Lượt vá chạy nhưng không trường nào đổi giá trị."
                ),
            )
        )

    outputs.append(
        OutputRow(
            field="repair_attempts",
            value=str(builder.snapshot.get("repair_attempts", 0)),
            detail="số lượt vá đã dùng; chạm trần thì hồ sơ đi tiếp với dữ liệu hiện có",
        )
    )
    return ["cv"], outputs


def _deep_review(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    """Which criteria the second opinion moved, and the quotes behind them."""
    after = [CriterionScore.model_validate(raw) for raw in builder.snapshot.get("criterion_scores") or []]
    before = {
        raw.get("criterion_id"): raw.get("score")
        for raw in (builder.previous or {}).get("criterion_scores") or []
    }

    if not after:
        return ["cv"], [
            OutputRow(
                field="criterion_scores",
                value="chưa có",
                detail="Node chưa chấm lại tiêu chí nào trong bước này.",
            )
        ]

    outputs: list[OutputRow] = []
    for score in after:
        old = before.get(score.criterion_id)
        changed = old is not None and abs(old - score.score) > 1e-9
        mark_ids = []
        # Only a criterion whose score moved earns marks: marking the untouched
        # ones would bury the handful this node actually revisited.
        if changed:
            mark_ids = [
                mark_id
                for evidence in score.evidence
                if (
                    mark_id := builder.add_mark(
                        "cv",
                        evidence.start,
                        evidence.end,
                        score.criterion_id,
                        "criterion_evidence",
                        evidence.score,
                    )
                )
            ]
        outputs.append(
            OutputRow(
                field=score.criterion_id,
                value=(
                    f"{old:.2f} → {score.score:.2f}" if changed else f"{score.score:.2f}"
                ),
                mark_ids=mark_ids,
                detail=score.reasoning or ("giữ nguyên sau khi chấm lại" if not changed else None),
            )
        )
    return ["cv"], outputs


def _no_document_row(field: str, detail: str) -> list[OutputRow]:
    return [OutputRow(field=field, value="chưa có", detail=detail)]


def _aggregate(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    """Weighted sum only -- this node never looks at the CV or the JD."""
    raw = builder.snapshot.get("scorecard")
    if not raw:
        return [], _no_document_row(
            "scorecard", "Node chưa tổng hợp được điểm trong bước này."
        )

    scorecard = Scorecard.model_validate(raw)
    rubric = _rubric_of(builder)
    weights = {c.id: c.weight for c in rubric.criteria} if rubric else {}

    outputs: list[OutputRow] = []
    for criterion_id, contribution in scorecard.weighted_contributions.items():
        weight = weights.get(criterion_id)
        outputs.append(
            OutputRow(
                field=criterion_id,
                value=f"{contribution:.4f}",
                detail=(
                    f"điểm tiêu chí × trọng số {round(weight * 100)}%"
                    if weight is not None
                    else "phần đóng góp vào điểm tổng"
                ),
            )
        )

    outputs.append(
        OutputRow(
            field="overall_score",
            value=f"{scorecard.overall_score:.4f}",
            detail="tổng các phần đóng góp ở trên",
        )
    )
    outputs.append(
        OutputRow(
            field="in_gray_zone",
            value="có" if scorecard.in_gray_zone else "không",
            detail=(
                "Điểm nằm sát một trong hai ngưỡng, nên hồ sơ được chấm lại ở deep_review."
                if scorecard.in_gray_zone
                else "Điểm nằm cách cả hai ngưỡng, đi thẳng sang decide."
            ),
        )
    )
    if scorecard.missing_must_haves:
        outputs.append(
            OutputRow(
                field="missing_must_haves",
                value=", ".join(scorecard.missing_must_haves),
                note="tính sau khi chấm, khác với cổng must_have_check trước đó",
            )
        )
    if scorecard.unscored_criteria:
        outputs.append(
            OutputRow(
                field="unscored_criteria",
                value=", ".join(scorecard.unscored_criteria),
                note="không có điểm, nên không đóng góp vào điểm tổng",
            )
        )
    return [], outputs


def _decide(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    """Two thresholds and one number: the whole of the labelling rule."""
    result = builder.snapshot.get("result") or {}
    raw_scorecard = builder.snapshot.get("scorecard")
    rubric = _rubric_of(builder)

    score = None
    if raw_scorecard:
        score = Scorecard.model_validate(raw_scorecard).overall_score
    elif "overall_score" in result:
        score = result["overall_score"]

    if score is None or rubric is None:
        return [], _no_document_row(
            "quyết định", "Thiếu điểm tổng hoặc bộ tiêu chí, chưa dựng được đối chiếu."
        )

    outputs = [
        OutputRow(
            field="overall_score",
            value=f"{score:.4f}",
            detail="điểm đem ra đối chiếu",
        ),
        OutputRow(
            field="good_fit_threshold",
            value=f"{rubric.good_fit_threshold:.2f}",
            detail=("đạt" if score >= rubric.good_fit_threshold else "không đạt"),
        ),
        OutputRow(
            field="potential_fit_threshold",
            value=f"{rubric.potential_fit_threshold:.2f}",
            detail=("đạt" if score >= rubric.potential_fit_threshold else "không đạt"),
        ),
        OutputRow(
            field="label",
            value=result.get("label") or "—",
            detail="nhãn cuối cùng, suy ra từ hai dòng ngay trên",
        ),
    ]
    return [], outputs


def _rank(builder: _Builder) -> tuple[list[str], list[OutputRow]]:
    """Closing node: the outcome, and the route that produced it."""
    result = builder.snapshot.get("result") or {}
    if not result:
        return [], _no_document_row("kết quả", "Lần chạy chưa chốt được kết quả.")

    outputs = [
        OutputRow(field="label", value=result.get("label") or "—"),
        OutputRow(
            field="overall_score",
            value=(
                f"{result['overall_score']:.4f}" if "overall_score" in result else "—"
            ),
        ),
    ]
    if result.get("rejected_reason"):
        outputs.append(
            OutputRow(field="rejected_reason", value=result["rejected_reason"])
        )
    outputs.append(
        OutputRow(
            field="path_taken",
            value=" → ".join(result.get("path_taken") or []) or "—",
            detail="đúng các node lần chạy này đã đi qua, kể cả nhánh rẽ",
        )
    )
    return [], outputs


_BUILDERS: dict[str, Callable[[_Builder], tuple[list[str], list[OutputRow]]]] = {
    "ingest": _ingest,
    "guard": _guard,
    "extract": _extract,
    "score_criteria": _score_criteria,
    "load_rubric": _load_rubric,
    "must_have_check": _must_have_check,
    "reject_fast": _reject_fast,
    "quarantine": _quarantine,
    "repair": _repair,
    "deep_review": _deep_review,
    "aggregate": _aggregate,
    "decide": _decide,
    "rank": _rank,
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

        previous = snapshots[index - 1] if index > 0 else None
        builder = _Builder(cv_text, jd_text, snapshot, previous)
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
