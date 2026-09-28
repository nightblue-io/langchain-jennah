"""JennahRetriever: semantic recall from one Jennah workspace as a LangChain retriever."""

from __future__ import annotations

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun, CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from jennah.agent.v1 import memory_pb2

from .memory import JennahMemory


def to_document(m: memory_pb2.SemanticMatch) -> Document:
    """One match as a Document. The content is the chunk's raw content; the
    metadata is the chunk's own map (including reserved ``jennah.*`` provenance
    keys when a formation wrote it) plus ``chunk_id`` and ``distance``, which
    take precedence over chunk keys of the same name."""
    metadata = dict(m.metadata)
    metadata["chunk_id"] = m.chunk_id
    metadata["distance"] = m.distance
    return Document(page_content=m.raw_content, metadata=metadata, id=m.chunk_id or None)


class JennahRetriever(BaseRetriever):
    """Semantic recall from the workspace ``memory`` is bound to, one document per
    match in the platform's order.

    The workspace, the result limit and the metadata filter all come from
    ``memory`` and none can be set per call, so wrapping this retriever as a tool
    gives the model a query field and nothing else.

    Both call paths are native: ``invoke`` uses the SDK's synchronous client and
    ``ainvoke`` its asynchronous one, so an async caller's event loop is never
    blocked on the network and never handed to a thread.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    memory: JennahMemory

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        return [to_document(m) for m in self.memory.search(query)]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [to_document(m) for m in await self.memory.asearch(query)]
