import asyncio
import time

import pytest
from langchain_core.messages import AIMessage, ChatMessage, FunctionMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import create_retriever_tool

import jennah
from jennah.agent.v1 import memory_pb2 as m

from fake import API_KEY
from langchain_jennah import FORM_TIMEOUT, JennahMemory, JennahRetriever

MATCHES = m.QueryMemoryResponse(semantic=m.SemanticResult(matches=[
    m.SemanticMatch(chunk_id="c1", raw_content="User lives in Tokyo", distance=0.12,
                    metadata={"jennah.source_step": "step_f1", "jennah.source_turns": "0-1"}),
    m.SemanticMatch(chunk_id="c2", raw_content="User likes hiking", distance=0.31, metadata={"topic": "hobby"}),
    m.SemanticMatch(chunk_id="c3", raw_content="User has a cat", distance=0.47),
]))


def run(coro):
    return asyncio.run(coro)


def aclient(fake):
    return jennah.AsyncClient(endpoint=fake.endpoint, insecure=True, api_key=API_KEY)


def check_docs(docs):
    assert [d.page_content for d in docs] == ["User lives in Tokyo", "User likes hiking", "User has a cat"]
    assert [d.metadata["chunk_id"] for d in docs] == ["c1", "c2", "c3"]
    assert docs[0].metadata["distance"] == pytest.approx(0.12)
    assert docs[0].metadata["jennah.source_step"] == "step_f1"
    assert docs[0].metadata["jennah.source_turns"] == "0-1"
    assert docs[1].metadata["topic"] == "hobby"


# ---- 5.1 client ownership ----


def test_supplied_client_is_left_open(fake, client):
    fake.query = MATCHES
    mem = JennahMemory("demo.a", client=client)
    JennahRetriever(memory=mem).invoke("q")
    mem.close()
    # still usable: the adapter did not create it, so it did not close it
    assert len(client.memory.QueryMemory(m.QueryMemoryRequest(agent_instance_id="demo.a"), timeout=5)
               .semantic.matches) == 3


def test_supplied_async_client_is_left_open(fake):
    async def go():
        ac = aclient(fake)
        mem = JennahMemory("demo.a", async_client=ac)
        await JennahRetriever(memory=mem).ainvoke("q")
        await mem.aclose()
        await ac.memory.QueryMemory(m.QueryMemoryRequest(agent_instance_id="demo.a"), timeout=5)
        await ac.close()

    run(go())


def test_sync_invoke_with_no_clients_supplied(fake, machine, monkeypatch):
    monkeypatch.setenv("JENNAH_API_KEY", API_KEY)
    fake.query = MATCHES
    mem = JennahMemory("demo.a", endpoint=fake.endpoint, insecure=True)
    check_docs(JennahRetriever(memory=mem).invoke("where do I live?"))
    created = mem._client
    mem.close()
    with pytest.raises(ValueError):  # the adapter created it, so it closed it
        created.memory.QueryMemory(m.QueryMemoryRequest(agent_instance_id="demo.a"), timeout=5)


def test_async_invoke_with_no_clients_supplied(fake, machine, monkeypatch):
    monkeypatch.setenv("JENNAH_API_KEY", API_KEY)
    fake.query = MATCHES

    async def go():
        mem = JennahMemory("demo.a", endpoint=fake.endpoint, insecure=True)
        check_docs(await JennahRetriever(memory=mem).ainvoke("q"))
        await mem.aclose()
        assert mem._async_client is None

    run(go())


# ---- 5.2 retriever ----


def test_retriever_sync_path(fake, client):
    fake.query = MATCHES
    mem = JennahMemory("demo.a", client=client, limit=3, metadata_filter={"topic": "hobby"})
    check_docs(JennahRetriever(memory=mem).invoke("q"))
    req = fake.of("QueryMemory")[0].request
    assert req.agent_instance_id == "demo.a"
    assert req.semantic.query_text == "q" and req.semantic.limit == 3
    assert [(f.key, f.value, f.operator) for f in req.semantic.filters] == [
        ("topic", "hobby", m.MetadataFilter.OPERATOR_EQUALS)]
    assert 0 < fake.of("QueryMemory")[0].deadline <= 62


def test_retriever_async_path(fake):
    fake.query = MATCHES

    async def go():
        ac = aclient(fake)
        f = m.MetadataFilter(key="n", value="3", operator=m.MetadataFilter.OPERATOR_GREATER_THAN)
        mem = JennahMemory("demo.a", async_client=ac, limit=2, metadata_filter=[f])
        check_docs(await JennahRetriever(memory=mem).ainvoke("q"))
        await ac.close()

    run(go())
    req = fake.of("QueryMemory")[0].request
    assert req.semantic.limit == 2 and req.semantic.filters[0].operator == m.MetadataFilter.OPERATOR_GREATER_THAN


def test_limit_and_filter_are_fixed(client):
    mem = JennahMemory("demo.a", client=client)
    r = JennahRetriever(memory=mem)
    with pytest.raises(AttributeError):
        mem.limit = 50
    with pytest.raises(Exception):
        r.memory = JennahMemory("demo.other", client=client)


# ---- 5.3 formation ----


def test_roles_map(fake, client):
    mem = JennahMemory("demo.a", client=client)
    mem.form_memory([
        SystemMessage("be brief"), HumanMessage("I moved to Tokyo"), AIMessage("Noted!"),
        ToolMessage("42", tool_call_id="t1"),
    ], formation_key="frm_s_1")
    req = fake.of("FormMemory")[0].request
    assert req.scope_id == "demo.a" and req.formation_key == "frm_s_1"
    assert [(t.role, t.content) for t in req.turns] == [
        (m.TURN_ROLE_SYSTEM, "be brief"), (m.TURN_ROLE_USER, "I moved to Tokyo"),
        (m.TURN_ROLE_ASSISTANT, "Noted!"), (m.TURN_ROLE_TOOL, "42")]


def test_block_content_is_flattened_to_text(fake, client):
    JennahMemory("demo.a", client=client).form_memory(
        [HumanMessage(content=[{"type": "text", "text": "hello"}, {"type": "text", "text": " world"}])],
        formation_key="k")
    assert fake.of("FormMemory")[0].request.turns[0].content == "hello world"


@pytest.mark.parametrize("bad", [ChatMessage("x", role="user"), FunctionMessage("x", name="f")])
def test_unsupported_type_is_refused_by_name(fake, client, bad):
    mem = JennahMemory("demo.a", client=client)
    with pytest.raises(TypeError, match=type(bad).__name__):
        mem.form_memory([HumanMessage("hi"), bad], formation_key="k")
    assert fake.calls == []


def test_missing_key_fails_before_any_call(fake, client):
    mem = JennahMemory("demo.a", client=client)
    with pytest.raises(TypeError):
        mem.form_memory([HumanMessage("hi")])
    for key in ("", "   "):
        with pytest.raises(ValueError, match="formation_key"):
            mem.form_memory([HumanMessage("hi")], formation_key=key)

    async def go():
        with pytest.raises(ValueError):
            await mem.aform_memory([HumanMessage("hi")], formation_key="")

    run(go())
    assert fake.calls == []


def test_same_key_replays_and_receipt_is_unaltered(fake, client):
    receipt = m.FormMemoryResponse(
        vector_rows=1, edge_supersessions=1, candidate_cap=20, candidates_dropped=2,
        candidates=[m.FormedCandidate(decision=m.MEMORY_DECISION_REVISED, matched_id="e_old")],
        redactions=[m.RedactionRecord(turn_index=0, masked_count=1)])
    fake.form = lambda req: receipt
    mem = JennahMemory("demo.a", client=client)
    msgs = [HumanMessage("I live in Tokyo now"), AIMessage("Got it")]
    first = mem.form_memory(msgs, formation_key="frm_s_1")
    second = mem.form_memory(msgs, formation_key="frm_s_1")
    assert first == receipt and second == receipt
    assert fake.extractions == 1 and len(fake.of("FormMemory")) == 2


def test_formation_deadline(fake, client):
    JennahMemory("demo.a", client=client).form_memory([HumanMessage("hi")], formation_key="k")
    assert FORM_TIMEOUT - 10 < fake.of("FormMemory")[0].deadline <= FORM_TIMEOUT + 2


def test_async_formation(fake):
    fake.form = lambda req: m.FormMemoryResponse(vector_rows=len(req.turns))

    async def go():
        ac = aclient(fake)
        r = await JennahMemory("demo.a", async_client=ac).aform_memory(
            [HumanMessage("a"), AIMessage("b")], formation_key="k")
        await ac.close()
        return r

    assert run(go()).vector_rows == 2
    assert FORM_TIMEOUT - 10 < fake.of("FormMemory")[0].deadline <= FORM_TIMEOUT + 2


# ---- 5.4 tool binding ----


def test_retriever_tool_offers_only_the_query(fake, client):
    fake.query = MATCHES
    tool = create_retriever_tool(JennahRetriever(memory=JennahMemory("demo.a", client=client)),
                                 "recall", "Search long-term memory.")
    schema = tool.tool_call_schema.model_json_schema()
    assert list(schema["properties"]) == ["query"]
    assert "User lives in Tokyo" in tool.invoke({"query": "where?"})
    assert fake.of("QueryMemory")[0].request.agent_instance_id == "demo.a"


# ---- 5.5 the async path never stalls the loop ----

STALL_BOUND = 0.1
SERVER_LATENCY = 0.5


async def longest_gap(work):
    """Run work() while a 10 ms ticker runs on the same loop; return the longest
    gap between ticks. A tick count would miss a single long block."""
    gaps = []
    done = asyncio.Event()

    async def ticker():
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.01)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    t = asyncio.create_task(ticker())
    await asyncio.sleep(0.05)
    try:
        await work()
    finally:
        done.set()
        await t
    return max(gaps)


def test_async_retrieval_and_formation_do_not_stall(fake):
    fake.query = MATCHES
    fake.delay = {"QueryMemory": SERVER_LATENCY, "FormMemory": SERVER_LATENCY}

    async def go():
        ac = aclient(fake)
        mem = JennahMemory("demo.a", async_client=ac)
        r = JennahRetriever(memory=mem)
        await r.ainvoke("warm up the channel")
        gap_recall = await longest_gap(lambda: r.ainvoke("q"))
        gap_form = await longest_gap(lambda: mem.aform_memory([HumanMessage("x")], formation_key="k"))
        await ac.close()
        return gap_recall, gap_form

    gap_recall, gap_form = run(go())
    assert gap_recall < STALL_BOUND, gap_recall
    assert gap_form < STALL_BOUND, gap_form


def test_stall_measure_catches_a_blocking_substitute(fake, client, monkeypatch):
    """The control: with _aget_relevant_documents replaced by a sync call, the
    same measurement must fail, or the test above proves nothing."""
    fake.query = MATCHES
    fake.delay = {"QueryMemory": SERVER_LATENCY}

    async def blocking(self, query, *, run_manager):
        return self._get_relevant_documents(query, run_manager=None)

    monkeypatch.setattr(JennahRetriever, "_aget_relevant_documents", blocking)
    r = JennahRetriever(memory=JennahMemory("demo.a", client=client))
    gap = run(longest_gap(lambda: r.ainvoke("q")))
    assert gap >= STALL_BOUND, gap
