from langgraph.pregel import Pregel

from deep_agent.graph import SUBAGENTS, SYSTEM_PROMPT, graph


def test_graph_compiles() -> None:
    assert isinstance(graph, Pregel)


def test_subagents_configured() -> None:
    names = {item["name"] for item in SUBAGENTS}
    assert names == {
        "crm_assistant",
        "invoicing_assistant",
        "accounting_assistant",
        "critic",
    }


def test_system_prompt_is_nonempty() -> None:
    assert len(SYSTEM_PROMPT.strip()) > 0


def test_system_prompt_requires_plan_and_approval() -> None:
    """La política Plan & Approval debe estar presente antes de cualquier mutación."""
    prompt = SYSTEM_PROMPT.lower()
    assert "plan & approval policy" in prompt
    assert "wait for the user's explicit approval" in prompt
    assert "never execute mutations in the same turn" in prompt


def test_subagents_require_approval_before_mutation() -> None:
    """Los subagentes que ejecutan mutaciones deben exigir aprobación del plan."""
    mutation_subagents = {
        "crm_assistant",
        "invoicing_assistant",
        "accounting_assistant",
    }
    for subagent in SUBAGENTS:
        if subagent["name"] in mutation_subagents:
            sp = subagent["system_prompt"].lower()
            assert "explicit user approval" in sp
            assert "propose the changes" in sp
