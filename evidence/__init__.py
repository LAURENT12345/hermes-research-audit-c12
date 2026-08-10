"""C1 — Hermes Evidence Pipeline.

Converts C0 chunks into structured, provenance-preserving Evidence objects
that can be validated and persisted.

C1 is infrastructure (Layer C). It is intentionally LLM-free:

* no OpenCode Go integration, no LLM calls of any kind,
* no claims / claim extraction / claim-evidence relationships,
* no answer or citation generation / verification,
* no web sources (only ``source_type = "local_pdf"``),
* no knowledge of the Dynamic Research Protocol or the research agent.

C1 builds on C0: a Chunk (canonical document data in ``docpipe``) becomes
an Evidence object (a structured, validated reference to that chunk).
The original chunk is never modified, replaced, or duplicated into the
evidence table — C0 remains the canonical document/chunk storage.
"""

__version__ = "0.1.0"
