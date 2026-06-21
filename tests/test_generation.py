from rag.generate import build_prompt, generate_answer
from rag.providers.llm import REFUSAL, StubLLM
from rag.types import Chunk, Scored


def _ctx(text, source="s.md", score=0.9):
    return Scored(chunk=Chunk(id=source + text[:5], text=text, source=source), score=score)


def test_answer_includes_citation():
    contexts = [_ctx("InsightRAG uses a cross-encoder reranker to reorder candidates.")]
    ans = generate_answer("What reranker does InsightRAG use?", contexts, llm=StubLLM())
    assert "[1]" in ans.text
    assert ans.citations and ans.citations[0].source == "s.md"
    assert not ans.refused


def test_refuses_when_no_context():
    ans = generate_answer("Who won the 2050 world cup?", [], llm=StubLLM())
    assert ans.refused
    assert ans.text == REFUSAL
    assert ans.citations == []


def test_refuses_when_context_irrelevant():
    contexts = [_ctx("The capital of France is Paris.")]
    ans = generate_answer("Explain photosynthesis in plants.", contexts, llm=StubLLM())
    assert ans.refused


def test_build_prompt_numbers_contexts_and_truncates():
    contexts = [_ctx("alpha " * 100, source="a"), _ctx("beta " * 100, source="b")]
    prompt = build_prompt("q?", contexts, max_chars=120)
    assert "[1]" in prompt and "QUESTION: q?" in prompt
    # respects the character budget (plus small structural overhead)
    assert len(prompt) < 400


def test_citation_only_for_used_markers():
    contexts = [
        _ctx("InsightRAG reranks with a cross-encoder.", source="a"),
        _ctx("Unrelated passage about cooking pasta.", source="b"),
    ]
    ans = generate_answer("What does InsightRAG rerank with?", contexts, llm=StubLLM())
    assert {c.source for c in ans.citations} == {"a"}
