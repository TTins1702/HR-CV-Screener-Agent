import pytest

from src.tools.evidence import search_evidence
from src.tools.skills import (
    DEFAULT_ALIAS_PATH,
    canonical_form,
    expand_skill,
    load_skill_aliases,
    load_skill_surface_forms,
    normalize_skill,
    skills_match,
)


def test_canonical_form_strips_case_punctuation_and_spaces():
    assert canonical_form("React.js") == "reactjs"
    assert canonical_form("  NODE  JS ") == "nodejs"
    assert canonical_form("CI/CD") == "cicd"
    assert canonical_form("C#") == "c#"


def test_canonical_form_keeps_the_sharp_and_plus_that_name_a_language():
    assert canonical_form("C++") == "c++"
    assert canonical_form("c sharp") == "csharp"


def test_the_shipped_alias_table_loads():
    aliases = load_skill_aliases()
    assert DEFAULT_ALIAS_PATH.exists()
    assert aliases[canonical_form("reactjs")] == "react"
    assert aliases[canonical_form("k8s")] == "kubernetes"


def test_the_canonical_name_maps_to_itself():
    aliases = load_skill_aliases()
    assert aliases[canonical_form("react")] == "react"
    assert aliases[canonical_form("node.js")] == "node.js"


def test_normalize_skill_resolves_a_known_alias():
    match = normalize_skill("ReactJS")
    assert match.raw == "ReactJS"
    assert match.canonical == "react"
    assert match.known is True
    assert match.matched_alias == "reactjs"


def test_normalize_skill_resolves_a_punctuated_variant_not_listed_verbatim():
    assert normalize_skill("REACT . JS").canonical == "react"


def test_normalize_skill_falls_back_to_the_canonical_form_of_an_unknown_skill():
    match = normalize_skill("Rust")
    assert match.canonical == "rust"
    assert match.known is False
    assert match.matched_alias is None


def test_normalize_skill_rejects_an_empty_string():
    with pytest.raises(ValueError):
        normalize_skill("   ")


def test_the_surface_form_table_keeps_multi_word_forms_verbatim():
    forms = load_skill_surface_forms()
    assert "react native" in forms["react"]
    assert "react.js" in forms["react"]
    assert "react" in forms["react"]


def test_expand_skill_returns_every_surface_form_longest_first():
    forms = expand_skill("react")
    assert forms == ["react native", "react.js", "reactjs", "react"]


def test_expand_skill_accepts_an_alias_as_its_argument():
    assert expand_skill("k8s") == expand_skill("kubernetes")


def test_expand_skill_returns_an_unknown_skill_alone():
    assert expand_skill("Rust") == ["Rust"]


def test_an_expanded_form_actually_finds_the_skill_in_a_cv():
    """The reason surface forms stay verbatim: a canonical key would miss this."""
    cv = "Built mobile apps with React Native and shipped a React.js dashboard."
    found = [form for form in expand_skill("react") if search_evidence(cv, form)]
    assert "react native" in found
    assert "reactnative" not in expand_skill("react")


def test_skills_match_across_surface_forms():
    assert skills_match("React", "react.js") is True
    assert skills_match("k8s", "Kubernetes") is True
    assert skills_match("Postgres", "PostgreSQL") is True


def test_skills_match_rejects_different_skills():
    assert skills_match("java", "javascript") is False
    assert skills_match("mysql", "postgresql") is False


def test_skills_match_on_two_unknown_skills_compares_canonical_forms():
    assert skills_match("Rust", "rust") is True
    assert skills_match("Rust", "Zig") is False


def test_a_custom_alias_table_overrides_the_default(tmp_path):
    path = tmp_path / "aliases.yaml"
    path.write_text("cobol:\n  - cobol85\n", encoding="utf-8")
    aliases = load_skill_aliases(path)
    assert normalize_skill("COBOL85", aliases).canonical == "cobol"
    assert normalize_skill("reactjs", aliases).known is False


def test_a_missing_alias_file_is_reported_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_skill_aliases(tmp_path / "nope.yaml")


def test_a_duplicate_alias_across_two_skills_is_rejected(tmp_path):
    path = tmp_path / "aliases.yaml"
    path.write_text("java:\n  - jdk\nkotlin:\n  - jdk\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jdk"):
        load_skill_aliases(path)


def test_loading_is_cached_so_repeated_calls_return_the_same_object():
    assert load_skill_aliases() is load_skill_aliases()
