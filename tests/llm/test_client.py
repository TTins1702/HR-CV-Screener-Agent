from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from src.llm.cache import JSONLCache
from src.llm.client import LLMRefusal, StructuredLLM


class Answer(BaseModel):
    value: int


def fake_response(content: str, refusal: str | None = None, prompt=100, completion=20):
    message = SimpleNamespace(content=content, refusal=refusal, parsed=None)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
    )


class FakeOpenAI:
    """Records every request and replays scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(parse=self._parse))

    def _parse(self, **kwargs):
        self.requests.append(kwargs)
        if not self.responses:
            raise AssertionError("FakeOpenAI ran out of scripted responses")
        return self.responses.pop(0)


def llm(tmp_path, responses, **kwargs) -> tuple[StructuredLLM, FakeOpenAI]:
    fake = FakeOpenAI(responses)
    return (
        StructuredLLM(
            model="gpt-4o-mini",
            seed=42,
            cache=JSONLCache(tmp_path / "c.jsonl"),
            client=fake,
            **kwargs,
        ),
        fake,
    )


def test_parse_returns_the_validated_schema_and_the_usage(tmp_path):
    client, _ = llm(tmp_path, [fake_response('{"value": 7}')])

    answer, usage = client.parse(system="s", user="u", schema=Answer)

    assert answer.value == 7
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 20
    assert usage.cached is False


def test_every_request_pins_temperature_zero_and_the_seed(tmp_path):
    client, fake = llm(tmp_path, [fake_response('{"value": 1}')])

    client.parse(system="s", user="u", schema=Answer)

    request = fake.requests[0]
    assert request["temperature"] == 0.0
    assert request["seed"] == 42
    assert request["model"] == "gpt-4o-mini"
    assert request["response_format"] is Answer
    assert request["messages"] == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]


def test_the_second_identical_call_is_served_from_cache(tmp_path):
    client, fake = llm(tmp_path, [fake_response('{"value": 7}')])

    client.parse(system="s", user="u", schema=Answer)
    answer, usage = client.parse(system="s", user="u", schema=Answer)

    assert answer.value == 7
    assert usage.cached is True
    assert usage.prompt_tokens == 100  # the cost it *would* have been, replayed
    assert len(fake.requests) == 1
    assert (client.hits, client.misses) == (1, 1)


def test_a_different_prompt_is_a_miss_not_a_stale_hit(tmp_path):
    client, fake = llm(tmp_path, [fake_response('{"value": 1}'), fake_response('{"value": 2}')])

    first, _ = client.parse(system="s", user="u", schema=Answer)
    second, _ = client.parse(system="s", user="different", schema=Answer)

    assert (first.value, second.value) == (1, 2)
    assert len(fake.requests) == 2


def test_the_cache_survives_a_new_client_on_the_same_file(tmp_path):
    first, _ = llm(tmp_path, [fake_response('{"value": 7}')])
    first.parse(system="s", user="u", schema=Answer)

    second, fake = llm(tmp_path, [])
    answer, usage = second.parse(system="s", user="u", schema=Answer)

    assert answer.value == 7
    assert usage.cached is True
    assert fake.requests == []


def test_a_refusal_raises_rather_than_returning_an_empty_answer(tmp_path):
    client, _ = llm(tmp_path, [fake_response("", refusal="I cannot help with that")])

    with pytest.raises(LLMRefusal, match="cannot help"):
        client.parse(system="s", user="u", schema=Answer)


def test_a_refusal_is_not_cached(tmp_path):
    client, fake = llm(
        tmp_path, [fake_response("", refusal="no"), fake_response('{"value": 3}')]
    )
    with pytest.raises(LLMRefusal):
        client.parse(system="s", user="u", schema=Answer)

    answer, _ = client.parse(system="s", user="u", schema=Answer)

    assert answer.value == 3
    assert len(fake.requests) == 2


def test_model_and_seed_fall_back_to_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("SCREENER_SEED", "7")

    client = StructuredLLM(cache=JSONLCache(tmp_path / "c.jsonl"), client=FakeOpenAI([]))

    assert client.model == "gpt-4o"
    assert client.seed == 7


def test_no_api_key_is_needed_until_a_real_call_is_made(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = StructuredLLM(cache=JSONLCache(tmp_path / "c.jsonl"))

    assert client.model  # constructing it did not touch openai


@pytest.mark.network
def test_a_live_call_returns_a_parsed_answer_and_real_usage(tmp_path):
    client = StructuredLLM(cache=JSONLCache(tmp_path / "c.jsonl"))

    answer, usage = client.parse(
        system="Answer with the number the user names.",
        user="The number is seven.",
        schema=Answer,
    )

    assert answer.value == 7
    assert usage.prompt_tokens > 0
    assert usage.cached is False
