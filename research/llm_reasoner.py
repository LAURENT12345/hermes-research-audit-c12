"""LLM Reasoner adapter for :class:`ReasonerPort` (Commit 12).

The adapter produces PROPOSAL-ONLY output. The canonical flow is::

    ResearchLoop -> ReasonerPort -> LLMReasoner -> proposal objects
        -> deterministic verifier -> domain-layer apply -> persisted state

The LLM NEVER becomes an authoritative state writer. This module:

* never imports ``research.store`` (no DB access of any kind),
* never imports ``research.claims`` / ``research.verifier`` (no domain
  apply functions, no status transitions),
* never writes files and never mutates claim status,
* never creates citations or evidence — evidence IDs are authoritative
  identifiers that the model may only REFERENCE from the supplied input,

so a malicious or hallucinating model cannot corrupt claim status,
evidence links, citations, contradictions, context or coverage. Every
proposal it emits must still pass the deterministic verifier before it
can be applied through the domain layer.

The C2 methods (``plan_queries``, ``construct_claims``, ``adjudicate``,
``synthesize``) are delegated to the embedded :class:`DeterministicReasoner`
so the adapter is a drop-in :class:`ReasonerPort` for the research loop.
The C3 methods (``propose_*``) go through the LLM transport.

Structured output contract (parsing is fail-closed; see
:func:`parse_proposals`):

* the model returns a single JSON object
  ``{"proposals": [{"type": ..., ...fields}, ...]}``,
* malformed JSON, an unknown ``type``, missing required fields, wrong
  field types, or a proposal type that does not belong to the requested
  method reject the WHOLE batch,
* empty ``{"proposals": []}`` is a valid "no proposals" outcome,
* fabricated IDs reach the verifier as proposals and are rejected there
  by deterministic validation — the parser never checks ID existence,
* enum/label/range violations (invalid role, unknown context key,
  out-of-range strength, wrong coverage label) reach the verifier and
  are rejected there — the parser only enforces JSON shape.

Failure semantics: every transport/parse failure raises
:class:`LLMReasonerError` subclasses. A model failure never converts into
"no evidence" or "research gap" and never touches an existing research
session — the adapter is pure, so there is nothing to corrupt.
"""

from __future__ import annotations

import json
from typing import Optional, Sequence

from research import config
from research.llm_transport import (
    LLMTimeoutError,
    LLMTransport,
    LLMTransportError,
    TransportConfig,
    build_transport,
    resolve_transport_config,
)
from research.reasoner import (
    ClaimContextProposal,
    ClaimDecompositionProposal,
    CoverageProposal,
    DeterministicReasoner,
    EvidenceRelationProposal,
    ReasonerPort,
    SemanticContradictionProposal,
)

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class LLMReasonerError(Exception):
    """Base class for LLM reasoner adapter failures."""


class LLMResponseError(LLMReasonerError):
    """The model's structured output failed parsing (fail-closed)."""


class LLMNotConfiguredError(LLMReasonerError):
    """No model was configured and no transport was injected."""


# ---------------------------------------------------------------------------
# Versioned system prompt (Phase 3 contract)
# ---------------------------------------------------------------------------

#: Versioned system prompt for the research reasoner. Bump the suffix when
#: the contract changes; keep the rules list stable across calls so model
#: input stays deterministic per (question, claims, evidence, context).
REASONER_SYSTEM_PROMPT_V1 = (
    "You are the research reasoner of a deterministic research engine. "
    "Your output is ADVISORY PROPOSAL DATA ONLY. You never change "
    "authoritative state.\n"
    "\n"
    "Hard rules:\n"
    "1. Evidence IDs are authoritative identifiers of persisted evidence. "
    "NEVER invent IDs. You may only reference IDs that appear in the "
    "supplied input.\n"
    "2. Every factual assertion must be grounded in the supplied evidence. "
    "Do not assert facts that the supplied evidence does not support.\n"
    "3. Distinguish observation from inference. When you infer, say so in "
    "the rationale; never present inference as observed fact.\n"
    "4. A contradiction is a PROPOSAL/OBSERVATION, never an authoritative "
    "fact. It never changes claim status.\n"
    "5. Claim context must use ONLY the allowed context keys listed in the "
    "input; a missing context value is explicit null, never universal "
    "applicability.\n"
    "6. Coverage is an assessment proposal, not authoritative state. The "
    "engine derives the authoritative coverage deterministically.\n"
    "7. Return JSON only. No markdown fences, no commentary, no trailing "
    "text.\n"
    "8. Never return citations that are not backed by the supplied "
    "evidence.\n"
    "9. Never change claim status directly; never propose status "
    "transitions.\n"
    "10. Do not create research gaps merely because evidence is missing. "
    "'No evidence found' is different from 'scientific research gap'.\n"
    "11. If you cannot produce a well-formed proposal for an item, omit it "
    "rather than guessing.\n"
)

#: Per-method task instructions (injected into the deterministic user
#: message; constant strings, so the input remains byte-stable).
_TASK_TEMPLATES: dict[str, str] = {
    "claim_decomposition": (
        "Propose atomic child propositions for the given claim. "
        "Return {\"proposals\": [{\"type\": \"claim_decomposition\", "
        "\"source_claim_id\": <source claim id from input>, "
        "\"child_statements\": [<atomic statement strings>], "
        "\"child_evidence_ids\": [<evidence ids from input only>], "
        "\"rationale\": <optional string>}]}. "
        "You may only reference evidence ids listed in the input."
    ),
    "evidence_relation": (
        "Propose claim<->evidence relationships for the given claim and "
        "evidence list. Return {\"proposals\": [{\"type\": "
        "\"evidence_relation\", \"claim_id\": <claim id from input>, "
        "\"evidence_id\": <evidence id from input>, \"role\": "
        "\"supporting\" or \"contradicting\"}]}. "
        "You may only reference evidence ids listed in the input."
    ),
    "semantic_contradiction": (
        "OBSERVE whether the two given claims are semantically "
        "contradictory, grounded in their supplied evidence. This is a "
        "proposal, not an authoritative fact. Return {\"proposals\": "
        "[{\"type\": \"semantic_contradiction\", \"claim_a_id\": <id>, "
        "\"claim_b_id\": <id>, \"evidence_a_ids\": [<evidence ids of "
        "claim_a from input>], \"evidence_b_ids\": [<evidence ids of "
        "claim_b from input>], \"rationale\": <non-empty string>, "
        "\"context\": <optional object with allowed keys>, \"strength\": "
        "<optional float 0..1>}]}. If no evidence is supplied for either "
        "claim, return {\"proposals\": []}."
    ),
    "claim_context": (
        "Propose scope/context qualifiers for the given claim using ONLY "
        "the allowed context keys listed in the input. Return "
        "{\"proposals\": [{\"type\": \"claim_context\", \"claim_id\": "
        "<id>, \"context\": {<allowed key>: <string or null>}}]} or "
        "{\"proposals\": []} if you have no grounded context."
    ),
    "coverage": (
        "Assess evidence coverage for the given claim. This is an "
        "assessment proposal; the engine derives authoritative coverage "
        "deterministically. Return {\"proposals\": [{\"type\": "
        "\"coverage\", \"claim_id\": <id>, \"coverage\": <one of "
        "directly_supported, partially_supported, unsupported, "
        "contradictory, insufficient, irrelevant>, \"notes\": <optional "
        "string>}]} or {\"proposals\": []}. 'No evidence found' is "
        "'unsupported', NOT a research gap."
    ),
}

#: Proposal type name -> proposal dataclass constructor.
_PROPOSAL_TYPES = frozenset(_TASK_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# Deterministic input serialization (Phase 4)
# ---------------------------------------------------------------------------


def _stable_json(value) -> str:
    """Byte-stable JSON: sorted keys, compact separators, UTF-8.

    ``sort_keys`` applies recursively in ``json.dumps``; lists keep their
    order, so callers MUST sort collections by stable ID first.
    """
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def serialize_claim(claim) -> str:
    """Deterministic serialization of one claim for model input."""
    context = None
    if claim.context:
        try:
            context = json.loads(claim.context)
        except (TypeError, json.JSONDecodeError):
            context = None
    return _stable_json(
        {
            "claim_id": claim.claim_id,
            "session_id": claim.session_id,
            "statement": claim.statement,
            "status": claim.status,
            "parent_claim_id": claim.parent_claim_id,
            "context": context,
        }
    )


def serialize_evidence(evidence) -> str:
    """Deterministic serialization of one evidence for model input."""
    return _stable_json(
        {
            "evidence_id": evidence.evidence_id,
            "document_id": evidence.document_id,
            "chunk_id": evidence.chunk_id,
            "source_title": evidence.source_title,
            "page_number": evidence.page_number,
            "section_path": evidence.section_path,
            "content": evidence.content,
            "locator": evidence.locator,
        }
    )


def _sorted_by_id(objects) -> list:
    """Stable ordering by object id attribute (never dict iteration order)."""
    return sorted(objects, key=lambda obj: obj.claim_id if hasattr(obj, "claim_id") else obj.evidence_id)


def build_model_input(method: str, **blocks) -> str:
    """Byte-stable user message for one reasoner method.

    ``blocks`` are pre-serialized JSON strings (see ``serialize_*``);
    they are embedded as values, so the exact same question/claims/
    evidence/context always yields the exact same model input.
    """
    payload: dict = {"task": _TASK_TEMPLATES[method]}
    for key in sorted(blocks):
        payload[key] = json.loads(blocks[key])
    return _stable_json(payload)


# ---------------------------------------------------------------------------
# Structured output parsing (Phase 2) — fail-closed
# ---------------------------------------------------------------------------


def _require_str(item: dict, field: str) -> str:
    value = item.get(field)
    if not isinstance(value, str):
        raise LLMResponseError(
            f"proposal field {field!r} must be a string, got "
            f"{type(value).__name__}"
        )
    return value


def _require_str_list(item: dict, field: str) -> tuple[str, ...]:
    """A REQUIRED list field: missing -> adapter error (fail-closed).

    An empty list passes through: semantic emptiness is decided by the
    deterministic verifier (e.g. ``child_statements`` must not be empty,
    ``evidence_a_ids`` must not be empty).
    """
    value = item.get(field)
    if value is None:
        raise LLMResponseError(f"proposal field {field!r} is required")
    if not isinstance(value, list) or not all(
        isinstance(entry, str) for entry in value
    ):
        raise LLMResponseError(
            f"proposal field {field!r} must be a list of strings, got "
            f"{type(value).__name__}"
        )
    return tuple(value)


def _optional_str_list(item: dict, field: str) -> tuple[str, ...]:
    """An OPTIONAL list field: missing/absent -> empty tuple."""
    value = item.get(field)
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(entry, str) for entry in value
    ):
        raise LLMResponseError(
            f"proposal field {field!r} must be a list of strings, got "
            f"{type(value).__name__}"
        )
    return tuple(value)


def _optional_str(item: dict, field: str) -> Optional[str]:
    value = item.get(field)
    if value is None:
        return None
    if not isinstance(value, str):
        raise LLMResponseError(
            f"proposal field {field!r} must be a string or null, got "
            f"{type(value).__name__}"
        )
    return value


def _optional_float(item: dict, field: str) -> Optional[float]:
    value = item.get(field)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LLMResponseError(
            f"proposal field {field!r} must be a number or null, got "
            f"{type(value).__name__}"
        )
    return float(value)


def _optional_dict(item: dict, field: str) -> Optional[dict]:
    value = item.get(field)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise LLMResponseError(
            f"proposal field {field!r} must be an object or null, got "
            f"{type(value).__name__}"
        )
    return dict(value)


def _parse_claim_decomposition(item: dict) -> ClaimDecompositionProposal:
    return ClaimDecompositionProposal(
        source_claim_id=_require_str(item, "source_claim_id"),
        child_statements=_require_str_list(item, "child_statements"),
        child_evidence_ids=_optional_str_list(item, "child_evidence_ids"),
        rationale=_optional_str(item, "rationale"),
    )


def _parse_evidence_relation(item: dict) -> EvidenceRelationProposal:
    return EvidenceRelationProposal(
        claim_id=_require_str(item, "claim_id"),
        evidence_id=_require_str(item, "evidence_id"),
        role=_require_str(item, "role"),
    )


def _parse_semantic_contradiction(
    item: dict,
) -> SemanticContradictionProposal:
    return SemanticContradictionProposal(
        claim_a_id=_require_str(item, "claim_a_id"),
        claim_b_id=_require_str(item, "claim_b_id"),
        evidence_a_ids=_require_str_list(item, "evidence_a_ids"),
        evidence_b_ids=_require_str_list(item, "evidence_b_ids"),
        rationale=_require_str(item, "rationale"),
        context=_optional_dict(item, "context"),
        strength=_optional_float(item, "strength"),
    )


def _parse_claim_context(item: dict) -> ClaimContextProposal:
    context = _optional_dict(item, "context")
    if context is None:
        raise LLMResponseError("proposal field 'context' is required")
    for key, value in context.items():
        if value is not None and not isinstance(value, str):
            raise LLMResponseError(
                f"context value for {key!r} must be a string or null, got "
                f"{type(value).__name__}"
            )
    return ClaimContextProposal(
        claim_id=_require_str(item, "claim_id"), context=context
    )


def _parse_coverage(item: dict) -> CoverageProposal:
    return CoverageProposal(
        claim_id=_require_str(item, "claim_id"),
        coverage=_require_str(item, "coverage"),
        notes=_optional_str(item, "notes"),
    )


_PARSERS = {
    "claim_decomposition": _parse_claim_decomposition,
    "evidence_relation": _parse_evidence_relation,
    "semantic_contradiction": _parse_semantic_contradiction,
    "claim_context": _parse_claim_context,
    "coverage": _parse_coverage,
}


def parse_proposals(raw: str, *, method: str) -> list:
    """Parse the model's structured output into typed proposal objects.

    Fail-closed rules:

    * empty/whitespace response -> :class:`LLMResponseError`,
    * malformed JSON -> :class:`LLMResponseError`,
    * missing ``proposals`` key / non-list -> :class:`LLMResponseError`,
    * unknown ``type`` -> :class:`LLMResponseError`,
    * a type not allowed for the requested method -> error,
    * missing required fields / wrong field types -> error,
    * any single bad item rejects the WHOLE batch.

    ``{"proposals": []}`` is a valid empty result and returns ``[]``.
    Fabricated IDs, invalid enums and out-of-range values pass through as
    proposals: the deterministic verifier rejects them (the parser is
    purely syntactic).
    """
    text = raw.strip()
    if not text:
        raise LLMResponseError("empty model response")
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"invalid JSON: {exc}") from exc
    if not isinstance(envelope, dict) or "proposals" not in envelope:
        raise LLMResponseError(
            "response must be an object with a 'proposals' list"
        )
    items = envelope["proposals"]
    if not isinstance(items, list):
        raise LLMResponseError("'proposals' must be a list")
    if method not in _PARSERS:
        raise LLMResponseError(f"unknown reasoner method {method!r}")
    parsed: list = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise LLMResponseError(
                f"proposal {index} must be an object, got {type(item).__name__}"
            )
        proposal_type = item.get("type")
        if not isinstance(proposal_type, str):
            raise LLMResponseError(
                f"proposal {index} has no 'type' string"
            )
        parser = _PARSERS.get(proposal_type)
        if parser is None:
            raise LLMResponseError(
                f"unknown proposal type {proposal_type!r}"
            )
        if proposal_type != method:
            raise LLMResponseError(
                f"proposal type {proposal_type!r} is not valid for method "
                f"{method!r}"
            )
        parsed.append(parser(item))
    return parsed


# ---------------------------------------------------------------------------
# Observability (never persisted)
# ---------------------------------------------------------------------------


class LLMReasonerStats:
    """Per-adapter call statistics (in-memory only; never persisted).

    Exposes the accepted/rejected surface and the last adapter error
    category without persisting arbitrary raw model output.
    """

    def __init__(self) -> None:
        self.calls = 0
        self.proposals_parsed = 0
        self.last_error_category: Optional[str] = None
        self.last_provider: Optional[str] = None
        self.last_model: Optional[str] = None


# ---------------------------------------------------------------------------
# The adapter
# ---------------------------------------------------------------------------


class LLMReasoner:
    """LLM-backed :class:`ReasonerPort` — proposal-only.

    Construction fails closed: without an injected transport AND without a
    configured model, :class:`LLMNotConfiguredError` is raised rather than
    guessing a model or silently degrading.
    """

    def __init__(
        self,
        transport: Optional[LLMTransport] = None,
        *,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        transport_config: Optional[TransportConfig] = None,
        deterministic_backend: Optional[DeterministicReasoner] = None,
    ):
        if transport is None:
            if transport_config is None:
                transport_config = resolve_transport_config(
                    provider=provider or config.LLM_DEFAULT_PROVIDER,
                    model=model or config.LLM_DEFAULT_MODEL,
                    base_url=base_url,
                    api_key=api_key,
                    timeout=timeout or config.LLM_DEFAULT_TIMEOUT_SECONDS,
                )
            if not transport_config.model:
                raise LLMNotConfiguredError(
                    "no LLM model configured: pass model=... or set "
                    "research.config.LLM_DEFAULT_MODEL; refusing to guess"
                )
            transport = build_transport(config=transport_config)
        self._transport = transport
        self._timeout = float(
            timeout or config.LLM_DEFAULT_TIMEOUT_SECONDS
        )
        self._backend = deterministic_backend or DeterministicReasoner()
        self._provider = provider or config.LLM_DEFAULT_PROVIDER
        self._model = model or config.LLM_DEFAULT_MODEL
        self.stats = LLMReasonerStats()
        self.stats.last_provider = self._provider
        self.stats.last_model = self._model

    # -- C2 methods: delegated to the deterministic backend ----------------

    def plan_queries(self, question: str, gaps: Sequence) -> list:
        return self._backend.plan_queries(question, gaps)

    def construct_claims(self, evidence: Sequence) -> list:
        return self._backend.construct_claims(evidence)

    def adjudicate(self, conflict) -> object:
        return self._backend.adjudicate(conflict)

    def synthesize(self, supported_claims: Sequence) -> object:
        return self._backend.synthesize(supported_claims)

    # -- C3 methods: LLM proposal generation -------------------------------

    def propose_decomposition(
        self, claim, *, evidence: Sequence = ()
    ) -> list[ClaimDecompositionProposal]:
        """Propose atomic child claims for a broad claim.

        ``evidence`` (optional, keyword-only) is the source claim's linked
        evidence; the model may only reference those ids. With no
        evidence the model proposes statements only.
        """
        blocks = {"claim": serialize_claim(claim)}
        if evidence:
            blocks["evidence"] = _stable_json(
                [json.loads(serialize_evidence(ev)) for ev in _sorted_by_id(evidence)]
            )
        return self._request("claim_decomposition", blocks)

    def propose_evidence_relations(
        self, claim, evidence: Sequence
    ) -> list[EvidenceRelationProposal]:
        """Propose claim<->evidence relationships (supporting/contradicting)."""
        if not evidence:
            return []
        blocks = {"claim": serialize_claim(claim)}
        blocks["evidence"] = _stable_json(
            [json.loads(serialize_evidence(ev)) for ev in _sorted_by_id(evidence)]
        )
        return self._request("evidence_relation", blocks)

    def propose_semantic_contradictions(
        self, claim_a, claim_b, *, evidence_a: Sequence = (),
        evidence_b: Sequence = (),
    ) -> list[SemanticContradictionProposal]:
        """Observe semantic contradictions between two claims.

        Without evidence for BOTH claims the model cannot produce a
        verifier-acceptable observation, so no model call is made and
        ``[]`` is returned (same as the deterministic baseline).
        """
        if not evidence_a or not evidence_b:
            return []
        blocks = {
            "claim_a": serialize_claim(claim_a),
            "claim_b": serialize_claim(claim_b),
        }
        blocks["evidence_a"] = _stable_json(
            [json.loads(serialize_evidence(ev)) for ev in _sorted_by_id(evidence_a)]
        )
        blocks["evidence_b"] = _stable_json(
            [json.loads(serialize_evidence(ev)) for ev in _sorted_by_id(evidence_b)]
        )
        return self._request("semantic_contradiction", blocks)

    def propose_context(
        self, claim
    ) -> Optional[ClaimContextProposal]:
        """Propose scope/context qualifiers for a claim (or None)."""
        proposals = self._request(
            "claim_context", {"claim": serialize_claim(claim)}
        )
        return proposals[0] if proposals else None

    def propose_coverage(self, claim) -> Optional[CoverageProposal]:
        """Propose an evidence-coverage assessment (or None)."""
        proposals = self._request(
            "coverage", {"claim": serialize_claim(claim)}
        )
        return proposals[0] if proposals else None

    # -- shared plumbing ---------------------------------------------------

    def _request(self, method: str, blocks: dict) -> list:
        """One transport round-trip: deterministic input -> typed proposals.

        Never applies anything. Transport failures raise
        :class:`LLMReasonerError` subclasses; the adapter is pure, so no
        session state can be corrupted by a model failure.
        """
        messages = [
            {"role": "system", "content": REASONER_SYSTEM_PROMPT_V1},
            {"role": "user", "content": build_model_input(method, **blocks)},
        ]
        self.stats.calls += 1
        try:
            raw = self._transport.complete(
                messages, timeout=self._timeout
            )
        except LLMTimeoutError as exc:
            self.stats.last_error_category = "timeout"
            raise LLMReasonerError(
                f"LLM reasoner {method}: timeout"
            ) from exc
        except LLMTransportError as exc:
            self.stats.last_error_category = "transport"
            raise LLMReasonerError(
                f"LLM reasoner {method}: transport failure: {exc}"
            ) from exc
        try:
            parsed = parse_proposals(raw, method=method)
        except LLMResponseError as exc:
            self.stats.last_error_category = "malformed"
            raise
        self.stats.proposals_parsed += len(parsed)
        self.stats.last_error_category = None
        return parsed
