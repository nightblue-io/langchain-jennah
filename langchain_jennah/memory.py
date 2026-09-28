"""JennahMemory: one Jennah workspace, bound once, for recall and formation."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

import jennah
from jennah.agent.v1 import memory_pb2

# Recall is a millisecond-class call. A formation is not: it runs two bounded
# generative calls plus a retrieval before it writes, and the platform holds the
# whole thing to the 300 s its load balancer allows a request. A shorter client
# deadline would abandon formations the server goes on to commit.
RECALL_TIMEOUT = 60.0
FORM_TIMEOUT = 300.0

MetadataFilter = Union[Mapping[str, str], Sequence[memory_pb2.MetadataFilter], None]

# The four message kinds formation has a turn role for. Order matters only in
# that each check is an isinstance, so chunk subclasses map with their parents.
_ROLES = (
    (HumanMessage, memory_pb2.TURN_ROLE_USER),
    (AIMessage, memory_pb2.TURN_ROLE_ASSISTANT),
    (SystemMessage, memory_pb2.TURN_ROLE_SYSTEM),
    (ToolMessage, memory_pb2.TURN_ROLE_TOOL),
)


def _filters(f: MetadataFilter) -> tuple[memory_pb2.MetadataFilter, ...]:
    if not f:
        return ()
    if isinstance(f, Mapping):
        return tuple(
            memory_pb2.MetadataFilter(key=k, value=v, operator=memory_pb2.MetadataFilter.OPERATOR_EQUALS) for k, v in f.items()
        )
    return tuple(f)


def to_turns(messages: Sequence[BaseMessage]) -> list[memory_pb2.ConversationTurn]:
    """LangChain messages as formation turns.

    A message of any other kind is refused by name rather than dropped: a
    formation silently missing a turn would form memory from a conversation that
    never happened in that shape.
    """
    turns = []
    for msg in messages:
        for cls, role in _ROLES:
            if isinstance(msg, cls):
                turns.append(memory_pb2.ConversationTurn(role=role, content=str(msg.text)))
                break
        else:
            raise TypeError(
                f"cannot form memory from a {type(msg).__name__}: only HumanMessage, AIMessage, "
                "SystemMessage and ToolMessage map to a conversation turn"
            )
    return turns


def _check_key(formation_key: str) -> None:
    # The adapter never mints a key. The key exists so that a caller's RETRY
    # replays the original formation, and a key minted per call cannot do that.
    if not isinstance(formation_key, str) or not formation_key.strip():
        raise ValueError("formation_key is required: a stable name for this formation, reused on retry")


class JennahMemory:
    """One Jennah workspace, for recall and for formation.

    The workspace is fixed here and nowhere else: no retrieval, formation, or tool
    built from this object takes an argument that names a workspace, so a model
    calling a tool cannot redirect it.

    The synchronous methods use a ``jennah.Client`` and the asynchronous ones a
    ``jennah.AsyncClient``. Either may be supplied; one that is not is created on
    first use from the SDK's default credential resolution (``$JENNAH_API_KEY``,
    then the ``jnh login`` session), with ``endpoint`` and ``insecure`` passed
    through. :meth:`close` and :meth:`aclose` close only the clients this object
    created.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        client: Optional[jennah.Client] = None,
        async_client: Optional[jennah.AsyncClient] = None,
        limit: int = 6,
        metadata_filter: MetadataFilter = None,
        endpoint: Optional[str] = None,
        insecure: bool = False,
    ) -> None:
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id is required")
        self._agent_id = agent_id.strip()
        self._limit = int(limit)
        self._filters = _filters(metadata_filter)
        self._client = client
        self._async_client = async_client
        self._owns_client = False
        self._owns_async_client = False
        self._endpoint = endpoint
        self._insecure = insecure

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def metadata_filter(self) -> tuple[memory_pb2.MetadataFilter, ...]:
        return self._filters

    # ---- clients ----

    def _sync(self) -> jennah.Client:
        if self._client is None:
            self._client = jennah.Client(endpoint=self._endpoint, insecure=self._insecure)
            self._owns_client = True
        return self._client

    def _async(self) -> jennah.AsyncClient:
        if self._async_client is None:
            self._async_client = jennah.AsyncClient(endpoint=self._endpoint, insecure=self._insecure)
            self._owns_async_client = True
        return self._async_client

    def close(self) -> None:
        """Close the synchronous client, if this object created it."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
            self._owns_client = False

    async def aclose(self) -> None:
        """Close the asynchronous client, if this object created it."""
        if self._owns_async_client and self._async_client is not None:
            await self._async_client.close()
            self._async_client = None
            self._owns_async_client = False

    # ---- recall ----

    def _query_request(self, query: str) -> memory_pb2.QueryMemoryRequest:
        return memory_pb2.QueryMemoryRequest(
            agent_instance_id=self._agent_id,
            semantic=memory_pb2.SemanticQuery(query_text=query, limit=self._limit, filters=self._filters),
        )

    def search(self, query: str) -> list[memory_pb2.SemanticMatch]:
        resp = self._sync().memory.QueryMemory(self._query_request(query), timeout=RECALL_TIMEOUT)
        return list(resp.semantic.matches)

    async def asearch(self, query: str) -> list[memory_pb2.SemanticMatch]:
        resp = await self._async().memory.QueryMemory(self._query_request(query), timeout=RECALL_TIMEOUT)
        return list(resp.semantic.matches)

    def as_retriever(self):
        from .retriever import JennahRetriever

        return JennahRetriever(memory=self)

    # ---- formation ----

    def _form_request(self, messages: Sequence[BaseMessage], formation_key: str) -> memory_pb2.FormMemoryRequest:
        _check_key(formation_key)
        return memory_pb2.FormMemoryRequest(
            scope_id=self._agent_id, turns=to_turns(messages), formation_key=formation_key
        )

    def form_memory(self, messages: Sequence[BaseMessage], *, formation_key: str) -> memory_pb2.FormMemoryResponse:
        """Submit a conversation to memory:form and return the platform's receipt
        unaltered: every candidate's decision, the supersessions, and the
        disclosures (dropped candidates, summarized structures, redactions).

        Repeating a call with the same ``formation_key`` replays the first receipt
        and forms nothing further. Everything is validated before any call.
        """
        req = self._form_request(messages, formation_key)
        return self._sync().memory.FormMemory(req, timeout=FORM_TIMEOUT)

    async def aform_memory(
        self, messages: Sequence[BaseMessage], *, formation_key: str
    ) -> memory_pb2.FormMemoryResponse:
        """:meth:`form_memory`, awaited on the SDK's asynchronous client."""
        req = self._form_request(messages, formation_key)
        return await self._async().memory.FormMemory(req, timeout=FORM_TIMEOUT)
