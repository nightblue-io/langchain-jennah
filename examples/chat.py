"""An async chat agent that remembers across sessions, with formation in the background.

Each turn recalls from Jennah, answers, prints the reply, and queues that turn's
formation. One worker task drains the queue, so formations run one at a time
while the conversation carries on:

    you> I moved to Tokyo last month
    assistant> Congratulations on the move! ...
      [turn 1: forming ...]
    you> any good hiking near there?
      [turn 1: formed: 1 new]
    assistant> Plenty! ...

Why one at a time: formation N+1's window overlaps formation N's. If N+1
reconciled before N committed, both could decide the same fact is new. A turn's
recall may not yet include the previous turn's formation, which is fine, because
the model's in-session history already holds that turn.

    pip install "langchain-jennah[examples] @ git+https://github.com/nightblue-io/langchain-jennah"
    python examples/chat.py                    # mints a demo workspace
    python examples/chat.py --agent demo.x     # reuses one
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import sys
from typing import Awaitable, Callable, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

import jennah
from jennah.agent.v1 import agent_pb2, memory_pb2

from langchain_jennah import JennahMemory, JennahRetriever

ANTHROPIC_MODEL = "claude-sonnet-5"
GEMINI_MODEL = "gemini-3.8-flash"

# How many recent messages each formation submits: this exchange and the two
# before it, so pronouns in this turn have their antecedents.
FORM_WINDOW = 6

ReadLine = Callable[[], Awaitable[Optional[str]]]
Write = Callable[[str], None]


def select_model(provider: str, env=os.environ) -> BaseChatModel:
    """The chat model. "auto" prefers Anthropic when ANTHROPIC_API_KEY is set, else
    Gemini on an AI Studio key, else Gemini on Vertex AI when a GCP project is set."""
    vertex = env.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ("1", "true") or (
        not env.get("GEMINI_API_KEY") and not env.get("GOOGLE_API_KEY") and bool(env.get("GOOGLE_CLOUD_PROJECT"))
    )
    if provider == "auto":
        if env.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or vertex:
            provider = "gemini"
        else:
            raise SystemExit("no chat credentials: set ANTHROPIC_API_KEY, GEMINI_API_KEY, or GOOGLE_CLOUD_PROJECT")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=ANTHROPIC_MODEL, max_tokens=2048)
    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # langchain-google-genai calls generate_content with automatic function
        # calling on, and google-genai warns about that on every request.
        logging.getLogger("google_genai.models").setLevel(logging.ERROR)
        if vertex:
            return ChatGoogleGenerativeAI(
                model=GEMINI_MODEL, vertexai=True, project=env.get("GOOGLE_CLOUD_PROJECT"),
                location=env.get("GOOGLE_CLOUD_LOCATION") or "global",
            )
        return ChatGoogleGenerativeAI(model=GEMINI_MODEL)
    raise SystemExit(f"unknown --provider {provider!r} (want auto|anthropic|gemini)")


def system_prompt(docs) -> str:
    lines = [
        "You are a warm, concise assistant with long-term memory that persists across sessions. "
        "Use the remembered context below when it is relevant.\n\n# Remembered\n"
    ]
    lines += [f"- {' '.join(d.page_content.split())}\n" for d in docs] or ["(nothing yet)\n"]
    return "".join(lines)


def summary(r: memory_pb2.FormMemoryResponse) -> str:
    if not r.candidates:
        return "nothing worth remembering"
    counts: dict[str, int] = {}
    for c in r.candidates:
        word = memory_pb2.MemoryDecision.Name(c.decision).removeprefix("MEMORY_DECISION_").lower()
        counts[word] = counts.get(word, 0) + 1
    out = ", ".join(f"{n} {w}" for w, n in counts.items())
    if r.edge_supersessions + r.chunk_supersessions:
        out += f"; retired {r.edge_supersessions + r.chunk_supersessions} earlier assertion(s)"
    return out


class Formations:
    """The formation queue and its one worker. Awaiting each formation before
    taking the next is what serializes them. A failure is reported, never dropped."""

    def __init__(self, memory: JennahMemory, write: Write) -> None:
        self._memory = memory
        self._write = write
        self._queue: asyncio.Queue = asyncio.Queue()
        self.pending = 0
        self._worker = asyncio.create_task(self._run())

    def submit(self, turn: int, messages: list[BaseMessage], key: str) -> None:
        self.pending += 1
        self._queue.put_nowait((turn, messages, key))

    async def _run(self) -> None:
        while True:
            turn, messages, key = await self._queue.get()
            try:
                receipt = await self._memory.aform_memory(messages, formation_key=key)
                self._write(f"  [turn {turn}: formed: {summary(receipt)}]")
            except Exception as e:
                self._write(f"  [turn {turn}: formation failed: {e}]")
            finally:
                self.pending -= 1
                self._queue.task_done()

    async def drain(self) -> None:
        if self.pending:
            self._write(f"  [waiting for {self.pending} formation(s) to finish ...]")
        await self._queue.join()
        self._worker.cancel()


async def chat(memory: JennahMemory, model: BaseChatModel, read_line: ReadLine, write: Write,
               session_id: Optional[str] = None) -> None:
    retriever = JennahRetriever(memory=memory)
    session_id = session_id or secrets.token_hex(8)
    history: list[BaseMessage] = []
    formations = Formations(memory, write)
    turn = 0
    try:
        while True:
            line = await read_line()
            if line is None:
                break
            line = line.strip()
            if not line:
                continue
            if line in ("/exit", "/quit"):
                break
            try:
                docs = await retriever.ainvoke(line)
                reply = await model.ainvoke([SystemMessage(system_prompt(docs)), *history, HumanMessage(line)])
            except Exception as e:
                write(f"error: {e}")
                continue
            text = str(reply.text).strip()
            write(f"assistant> {text}")

            turn += 1
            history += [HumanMessage(line), AIMessage(text)]
            # The key is (session, turn ordinal): a retry of this formation replays
            # it, and a user repeating themselves on a later turn is a new formation.
            write(f"  [turn {turn}: forming ...]")
            formations.submit(turn, history[-FORM_WINDOW:], f"frm_{session_id}_{turn}")
    finally:
        await formations.drain()


async def stdin_line() -> Optional[str]:
    """One line from stdin, read on a thread so the formation worker keeps running
    while the user types."""
    def read():
        try:
            return input("you> ")
        except EOFError:
            return None

    return await asyncio.to_thread(read)


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--agent", default="", help="existing workspace id; without it a demo.* workspace is created")
    p.add_argument("--provider", default="auto", help="auto|anthropic|gemini")
    p.add_argument("--region", default="", help="home region for a NEW workspace; empty uses the platform default")
    p.add_argument("--endpoint", default=None, help=f"gRPC host:port (default {jennah.DEFAULT_ENDPOINT})")
    p.add_argument("--insecure", action="store_true", help="plaintext, for a local server")
    args = p.parse_args()

    model = select_model(args.provider)
    client = jennah.AsyncClient(endpoint=args.endpoint, insecure=args.insecure)
    agent_id = args.agent.strip()
    if not agent_id:
        agent_id = "demo.langchain_" + secrets.token_hex(8)
        await client.agents.CreateAgent(
            agent_pb2.CreateAgentRequest(agent_instance_id=agent_id, agent_name="langchain-demo", region=args.region),
            timeout=60,
        )
        print(f"created workspace {agent_id} (reuse it with --agent {agent_id})")
    else:
        print(f"using workspace {agent_id}")
    memory = JennahMemory(agent_id, async_client=client)
    print("Ctrl-D or /exit to quit.\n")
    try:
        await chat(memory, model, stdin_line, lambda s: print(s, flush=True))
    finally:
        await client.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
