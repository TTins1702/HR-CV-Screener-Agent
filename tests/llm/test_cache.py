import json

from src.llm.cache import JSONLCache, cache_key

MESSAGES = [{"role": "system", "content": "extract"}, {"role": "user", "content": "cv"}]
SCHEMA = {"type": "object", "properties": {"skills": {"type": "array"}}}


def key(**overrides) -> str:
    base = dict(
        model="gpt-4o-mini",
        messages=MESSAGES,
        schema_name="Extraction",
        schema=SCHEMA,
        temperature=0.0,
        seed=42,
    )
    base.update(overrides)
    return cache_key(**base)


def test_the_key_is_stable_across_calls():
    assert key() == key()


def test_the_key_is_a_sha256_hex_digest():
    assert len(key()) == 64
    assert set(key()) <= set("0123456789abcdef")


def test_every_input_that_can_change_the_answer_changes_the_key():
    baseline = key()

    assert key(model="gpt-4o") != baseline
    assert key(messages=[{"role": "user", "content": "other"}]) != baseline
    assert key(schema_name="Other") != baseline
    assert key(schema={"type": "string"}) != baseline
    assert key(temperature=0.7) != baseline
    assert key(seed=7) != baseline


def test_a_miss_returns_none(tmp_path):
    cache = JSONLCache(tmp_path / "c.jsonl")

    assert cache.get("nope") is None
    assert len(cache) == 0


def test_put_then_get_round_trips(tmp_path):
    cache = JSONLCache(tmp_path / "c.jsonl")

    cache.put("k", {"content": '{"skills": []}', "prompt_tokens": 10, "completion_tokens": 2})

    assert cache.get("k")["prompt_tokens"] == 10
    assert len(cache) == 1


def test_entries_survive_a_new_process(tmp_path):
    path = tmp_path / "c.jsonl"
    JSONLCache(path).put("k", {"content": "{}", "prompt_tokens": 1, "completion_tokens": 1})

    assert JSONLCache(path).get("k") == {"content": "{}", "prompt_tokens": 1, "completion_tokens": 1}


def test_the_last_write_for_a_key_wins(tmp_path):
    path = tmp_path / "c.jsonl"
    cache = JSONLCache(path)
    cache.put("k", {"content": "1", "prompt_tokens": 1, "completion_tokens": 1})
    cache.put("k", {"content": "2", "prompt_tokens": 1, "completion_tokens": 1})

    assert JSONLCache(path).get("k")["content"] == "2"


def test_a_truncated_tail_from_an_interrupted_run_does_not_poison_the_cache(tmp_path):
    path = tmp_path / "c.jsonl"
    JSONLCache(path).put("k", {"content": "{}", "prompt_tokens": 1, "completion_tokens": 1})
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "half", "val')

    cache = JSONLCache(path)

    assert cache.get("k") is not None
    assert len(cache) == 1


def test_the_file_is_one_json_object_per_line(tmp_path):
    path = tmp_path / "c.jsonl"
    cache = JSONLCache(path)
    cache.put("a", {"content": "1", "prompt_tokens": 1, "completion_tokens": 1})
    cache.put("b", {"content": "2", "prompt_tokens": 1, "completion_tokens": 1})

    lines = path.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 2
    assert {json.loads(line)["key"] for line in lines} == {"a", "b"}
