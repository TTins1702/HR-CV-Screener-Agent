import json

import pytest

from scripts.dump_criterion_scores import CacheOnlyLLM, dump
from tests.eval.test_run import ROWS, SchemaStub


def write_split(tmp_path, rows=ROWS):
    path = tmp_path / "split.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def read_dump(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_every_row_keeps_its_criterion_scores_and_the_weights_behind_them(tmp_path):
    """The values `RowRecord` throws away.

    Records store how many criteria were scored, not what they scored, so the
    0.482 ordering could not be traced to the numbers that produced it.
    """
    out = tmp_path / "dump.jsonl"

    written = dump(write_split(tmp_path), out, llm=SchemaStub(), derived_dir=tmp_path)

    assert written == 2
    rows = read_dump(out)
    first = rows[0]
    assert first["true_label"] == "Good Fit"
    assert {c["id"] for c in first["criteria"]} == {"lang", "seniority"}
    assert all(c["weight"] == 0.5 for c in first["criteria"])
    assert all(c["kind"] in {"skill", "experience_years"} for c in first["criteria"])


def test_the_weighted_sum_of_the_dumped_criteria_reproduces_the_overall_score(tmp_path):
    """The dump has to be able to answer for the number it sits beside.

    If the criterion scores and weights here did not re-aggregate to the same
    overall score, every ceiling computed from them would be measuring a
    different pipeline than the one that shipped.
    """
    out = tmp_path / "dump.jsonl"
    dump(write_split(tmp_path), out, llm=SchemaStub(), derived_dir=tmp_path)

    for row in read_dump(out):
        if "reject_fast" in row["path_taken"]:
            continue
        weighted = sum(c["weight"] * c["score"] for c in row["criteria"])
        total = sum(c["weight"] for c in row["criteria"])
        assert weighted / total == pytest.approx(row["overall_score"], abs=1e-4)


def test_rows_that_share_a_job_description_share_a_fingerprint(tmp_path):
    """What the cross-validation folds are grouped by.

    Both fixture rows answer the same JD. Ungrouped, a classifier can learn that
    JD's label and read back 0.701 where the honest figure is 0.577.
    """
    out = tmp_path / "dump.jsonl"
    dump(write_split(tmp_path), out, llm=SchemaStub(), derived_dir=tmp_path)

    rows = read_dump(out)
    assert rows[0]["jd_fingerprint"] == rows[1]["jd_fingerprint"]


def test_a_row_that_raises_is_recorded_rather_than_losing_the_whole_dump(tmp_path):
    class Broken(SchemaStub):
        def parse(self, *, system, user, schema):
            if self.calls > 1:
                raise RuntimeError("extractor gave up")
            return super().parse(system=system, user=user, schema=schema)

    out = tmp_path / "dump.jsonl"
    dump(write_split(tmp_path), out, llm=Broken(), derived_dir=tmp_path)

    rows = read_dump(out)
    assert any("error" in row for row in rows)


def test_the_cache_only_client_refuses_to_pay_for_a_drifted_prompt():
    """A replay that quietly turns into a 41-minute paid run is the failure here.

    `--allow-live` exists so that spending money is a thing somebody typed.
    """
    with pytest.raises(RuntimeError, match="cache miss"):
        CacheOnlyLLM(cache=None)._request([], object)
