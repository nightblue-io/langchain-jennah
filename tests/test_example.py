"""examples/chat.py against the fake: background formation, one at a time, drained on exit."""

import asyncio
import importlib.util
import os
import threading

import grpc
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import jennah
from jennah.agent.v1 import memory_pb2 as m

from fake import API_KEY
from langchain_jennah import JennahMemory

_spec = importlib.util.spec_from_file_location(
    "example_chat", os.path.join(os.path.dirname(__file__), "..", "examples", "chat.py"))
example = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example)


def model(*replies):
    return GenericFakeChatModel(messages=iter([AIMessage(r) for r in replies]))


class Script:
    """Scripted input. Each step is a line, None for EOF, or a callable run (on the
    loop) before the next line is handed over."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.out = []

    async def read(self):
        while self.steps:
            step = self.steps.pop(0)
            if callable(step):
                await step(self)
                continue
            return step
        return None

    def write(self, s):
        self.out.append(s)


def run_chat(fake, script, replies):
    async def go():
        ac = jennah.AsyncClient(endpoint=fake.endpoint, insecure=True, api_key=API_KEY)
        try:
            await example.chat(JennahMemory("demo.a", async_client=ac), model(*replies), script.read, script.write,
                               session_id="s1")
        finally:
            await ac.close()

    asyncio.run(asyncio.wait_for(go(), 20))


def test_second_turn_answers_while_first_formation_is_held(fake):
    fake.hold["FormMemory"] = threading.Event()
    fake.form = lambda req: m.FormMemoryResponse(candidates=[m.FormedCandidate(decision=m.MEMORY_DECISION_NEW)])
    seen = {}

    async def check_then_release(script):
        # By the time the third line is asked for, turn 2 has been answered while
        # turn 1's formation is still held open inside the fake.
        seen["out"] = list(script.out)
        seen["form_calls"] = [(c.request.formation_key, c.ended) for c in fake.of("FormMemory")]
        fake.hold["FormMemory"].set()

    script = Script("I moved to Tokyo", "hiking near there?", check_then_release, None)
    run_chat(fake, script, ["Congrats!", "Plenty."])

    assert "assistant> Plenty." in seen["out"]
    assert not any("turn 1: formed" in s for s in seen["out"])
    assert seen["form_calls"] == [("frm_s1_1", None)]
    assert "  [turn 1: formed: 1 new]" in script.out and "  [turn 2: formed: 1 new]" in script.out
    assert fake.of("QueryMemory")[1].request.semantic.query_text == "hiking near there?"


def test_formations_do_not_overlap(fake):
    fake.delay["FormMemory"] = 0.2
    script = Script("one", "two", "three", None)
    run_chat(fake, script, ["a", "b", "c"])
    calls = fake.of("FormMemory")
    assert [c.request.formation_key for c in calls] == ["frm_s1_1", "frm_s1_2", "frm_s1_3"]
    for earlier, later in zip(calls, calls[1:]):
        assert later.started >= earlier.ended
    # each window is the last six messages, user and assistant alike
    assert [t.content for t in calls[2].request.turns] == ["one", "a", "two", "b", "three", "c"]


def test_exit_drains_the_in_flight_formation(fake):
    fake.hold["FormMemory"] = threading.Event()
    threading.Timer(0.3, fake.hold["FormMemory"].set).start()
    script = Script("remember I like tea", None)
    run_chat(fake, script, ["Tea it is."])
    i = script.out.index("  [waiting for 1 formation(s) to finish ...]")
    assert script.out[i + 1] == "  [turn 1: formed: nothing worth remembering]"
    assert fake.of("FormMemory")[0].ended is not None


def test_failed_formation_is_reported(fake):
    fake.fail["FormMemory"] = grpc.StatusCode.INVALID_ARGUMENT
    script = Script("hi", "/exit")
    run_chat(fake, script, ["hello"])
    assert any(s.startswith("  [turn 1: formation failed:") for s in script.out)
