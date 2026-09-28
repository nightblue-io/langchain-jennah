## langchain-jennah

Jennah memory for LangChain: a retriever for recall, and a formation call that
turns a LangChain conversation into long-term memory.

- `JennahRetriever` is a LangChain `BaseRetriever`. It runs a semantic query
  against one Jennah workspace and returns one `Document` per match.
- `JennahMemory.form_memory` / `aform_memory` submit LangChain messages to
  Jennah's `memory:form`. Jennah extracts what is worth remembering, reconciles
  it with what the workspace already holds, and returns a receipt with every
  decision it made.

Both work on LangChain's sync path (`invoke`) and async path (`ainvoke`). The
async path uses the SDK's asynchronous client directly, so it never blocks the
event loop and never runs the sync client on a thread.

## Install

```sh
pip install git+https://github.com/nightblue-io/langchain-jennah
```

Requires Python 3.10 or later, `jennah-sdk-py` 0.61.0 or later, and
`langchain-core` 1.x. The example agent also needs a chat model integration:

```sh
pip install "langchain-jennah[examples] @ git+https://github.com/nightblue-io/langchain-jennah"
```

## Credentials

With no client supplied, the adapter creates one from the SDK's default
resolution: `$JENNAH_API_KEY`, then the session stored by `jnh login`. You can
also pass your own `jennah.Client` and `jennah.AsyncClient`.

## Usage

```python
from langchain_core.messages import AIMessage, HumanMessage
from langchain_jennah import JennahMemory, JennahRetriever

memory = JennahMemory("my-agent", limit=6)
retriever = JennahRetriever(memory=memory)

docs = await retriever.ainvoke("where does the user live?")
for d in docs:
    print(d.page_content, d.metadata["distance"])

receipt = await memory.aform_memory(
    [HumanMessage("I moved to Tokyo last month"), AIMessage("Congratulations!")],
    formation_key="frm_session42_7",
)
print(receipt.edge_supersessions, [c.decision for c in receipt.candidates])
```

### `JennahMemory(agent_id, *, client=None, async_client=None, limit=6, metadata_filter=None, endpoint=None, insecure=False)`

Binds one workspace. The workspace, the result `limit`, and the
`metadata_filter` are fixed here and cannot be changed per call.
`metadata_filter` is either a dict (each key must equal its value) or a list of
`MetadataFilter` messages for ordered comparisons. `endpoint` and `insecure`
apply only to clients the adapter creates itself. `close()` and `aclose()` close
only those clients, never ones you passed in.

### `JennahRetriever(memory=...)`

Each document's `page_content` is the chunk's raw content. Its `metadata` holds
the chunk's own metadata map, plus `chunk_id` and `distance`. Memory written by
a formation also carries `jennah.source_step` and `jennah.source_turns`, which
trace it back to the conversation it came from.

Wrapped as a tool, the retriever exposes only a query:

```python
from langchain_core.tools import create_retriever_tool

tool = create_retriever_tool(retriever, "recall", "Search the user's long-term memory.")
# tool's input schema: {"query": string}. The model cannot pick another workspace.
```

### `form_memory(messages, *, formation_key)` / `aform_memory(...)`

- `HumanMessage`, `AIMessage`, `SystemMessage` and `ToolMessage` map to user,
  assistant, system and tool turns. Any other message type raises `TypeError`
  naming it, and nothing is sent.
- `formation_key` is required. Use a stable name for this formation, such as
  session id plus turn number. Repeating a call with the same key returns the
  first receipt and forms no additional memory, which makes retries safe. The
  adapter never generates a key for you, because a key generated per call would
  make every retry look like a new formation.
- The receipt is Jennah's `FormMemoryResponse`, unaltered: every candidate's
  decision, supersessions, dropped candidates, summarized structures and
  redactions.
- Formation runs model inference before it writes, so it takes seconds. The
  adapter allows it 300 seconds. Recall gets 60.

## Example agent

`examples/chat.py` is an async chat loop. Each turn it recalls, answers, prints
the reply, and hands that turn's formation to a single background worker, so
you can keep chatting while memory forms:

```sh
export JENNAH_API_KEY=jennah_sk_...      # or: jnh login
export ANTHROPIC_API_KEY=sk-ant-...      # or GEMINI_API_KEY, or GOOGLE_CLOUD_PROJECT for Vertex AI
python examples/chat.py                  # creates a demo.langchain_* workspace
python examples/chat.py --agent demo.x   # reuses one
```

```text
you> I just adopted a cat named Mochi.
assistant> Congratulations! Mochi is such a sweet name. ...
  [turn 1: forming ...]
you> What food would you suggest for a young cat?
  [turn 1: formed: 2 new]
assistant> For a young cat like Mochi, ...
  [turn 2: forming ...]
```

Formations run one at a time: each turn's window overlaps the previous one's,
and letting them reconcile concurrently could record the same fact twice. A
failed formation is printed. On exit the agent waits for any formation still
running and prints its result.

Because formation runs in the background, a turn's recall may not include what
the previous turn formed yet. That is fine: the chat history held in the
session already contains that turn.

## Tests

```sh
pip install ".[examples,test]"
pytest
```

The tests run against an in-process fake Jennah reached through the SDK's own
clients. They include a check that async retrieval and formation never stall
the event loop, with a control that proves the check catches a blocking call.

## License

Apache-2.0
