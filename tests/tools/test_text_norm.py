import pytest

from src.tools.text_norm import normalize, normalize_with_map


def test_lowercases_and_collapses_whitespace():
    assert normalize("Hello   WORLD\n\tagain") == "hello world again"


def test_inserts_a_space_at_a_glued_sentence_boundary():
    assert normalize("consulting projects.Proven ability") == (
        "consulting projects. proven ability"
    )


def test_inserts_a_space_at_a_glued_camel_boundary():
    assert normalize("SkillsPython") == "skills python"


def test_inserts_a_space_between_a_digit_and_a_following_capital():
    assert normalize("12/2011toPresentData Analyst") == "12/2011to present data analyst"


def test_leaves_an_already_clean_string_alone_apart_from_case():
    assert normalize("python and sql") == "python and sql"


def test_does_not_split_a_chat_template_control_token():
    # scan_injection matches `<|im_start|>` against the normalized text, so a
    # space inserted after the pipe would make that rule unmatchable forever.
    assert normalize("<|im_start|>system") == "<|im_start|>system"


def test_normalizing_twice_changes_nothing():
    text = "Rate: 3.5 years.Python|Java  •Docker"
    once = normalize(text)
    assert normalize(once) == once


def test_map_recovers_the_original_span_verbatim():
    text = "consulting projects.Proven ability to lead"
    normalized, index_map = normalize_with_map(text)
    start = normalized.index("proven ability")
    end = start + len("proven ability")
    assert text[index_map[start] : index_map[end]] == "Proven ability"


def test_map_recovers_a_span_across_a_glued_boundary():
    text = "used laptops.Entered admissions data"
    normalized, index_map = normalize_with_map(text)
    query = "laptops. entered admissions"
    start = normalized.index(query)
    end = start + len(query)
    # The original has no space after the period, so the recovered span is glued.
    assert text[index_map[start] : index_map[end]] == "laptops.Entered admissions"


def test_map_has_one_more_entry_than_the_normalized_text():
    text = "  Padded   text.Here  "
    normalized, index_map = normalize_with_map(text)
    assert len(index_map) == len(normalized) + 1
    assert index_map[len(normalized)] == len(text)


def test_map_is_monotonically_non_decreasing():
    text = "A.B  c.D\nE.f   ghI"
    _, index_map = normalize_with_map(text)
    assert all(a <= b for a, b in zip(index_map, index_map[1:]))


def test_leading_whitespace_is_dropped_and_offsets_stay_correct():
    text = "\n\n   Python"
    normalized, index_map = normalize_with_map(text)
    assert normalized == "python"
    assert text[index_map[0] : index_map[6]] == "Python"


def test_empty_input_is_handled():
    assert normalize_with_map("") == ("", [0])


@pytest.mark.parametrize("text", ["", "   ", "a", "a.B", "1990toPresent", "x" * 5000])
def test_every_map_index_is_a_valid_slice_bound(text):
    normalized, index_map = normalize_with_map(text)
    for index in index_map:
        assert 0 <= index <= len(text)
