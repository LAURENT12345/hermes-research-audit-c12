"""ReasonerPort + DeterministicReasoner — the C2 reasoning port.

The Reasoner only PROPOSES. It never decides claim status: transitions
like ``proposed -> supported`` belong exclusively to the deterministic
evaluator in later commits. This module therefore:

* never imports ``research.store``,
* never writes to the database,
* never calls an LLM and never touches the web.

The MVP implementation (:class:`DeterministicReasoner`) is fully
deterministic and reproducible: same input always yields the same
output. An LLM-backed implementation of :class:`ReasonerPort` (e.g. via
OpenCode Go) can be swapped in later without touching the loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, Sequence, runtime_checkable

from docpipe.search import STOPWORDS  # reuse the C0 stopword list
from evidence.model import Evidence

from research import config

# Adjudication decisions.
DECISION_SUPPORTING_STANDS = "supporting_stands"
DECISION_CONTRADICTION_WINS = "contradiction_wins"
DECISION_UNDETERMINED = "undetermined"

_WORD_RE = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True)
class Gap:
    """A claim that lacks sufficient evidence (input to plan_queries)."""

    claim_id: str
    statement: str


@dataclass(frozen=True)
class QueryProposal:
    """A retrieval query proposed by the reasoner (never executed here)."""

    query_text: str
    origin: str = "derived"


@dataclass(frozen=True)
class ClaimProposal:
    """A claim the reasoner proposes from evidence.

    Deliberately has NO ``status`` field: whether the claim is supported
    is decided by the deterministic evaluator, never by the reasoner.
    """

    statement: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Conflict:
    """A claim with both supporting and contradicting evidence."""

    claim_id: str
    statement: str
    supporting_evidence_ids: tuple[str, ...]
    contradicting_evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class Adjudication:
    """The reasoner's proposal for resolving a conflict."""

    claim_id: str
    decision: str  # one of DECISION_*
    note: str


@dataclass(frozen=True)
class SupportedClaim:
    """A claim the evaluator has already marked supported."""

    claim_id: str
    statement: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class SynthesisDraft:
    """Structured synthesis output (interface only; real logic is a later
    commit). Statements are sorted deterministically by claim_id."""

    summary: str
    statements: tuple[dict, ...] = field(
        default_factory=tuple
    )  # each: {"claim_id", "statement", "evidence_ids"}


# ---------------------------------------------------------------------------
# C3 proposal types (reasoner PROPOSES; the deterministic verifier decides)
# ---------------------------------------------------------------------------

#: Allowed claim scope/context keys (extensible; do not build an ontology).
CONTEXT_KEYS = (
    "population", "dataset", "task", "intervention", "condition",
    "method", "model", "setting",
)

COVERAGE_DIRECTLY_SUPPORTED = "directly_supported"
COVERAGE_PARTIALLY_SUPPORTED = "partially_supported"
COVERAGE_UNSUPPORTED = "unsupported"
COVERAGE_CONTRADICTORY = "contradictory"
COVERAGE_INSUFFICIENT = "insufficient"  # evidence exists but is insufficient
COVERAGE_IRRELEVANT = "irrelevant"      # reserved for C4; never derived here

COVERAGE_LABELS = frozenset({
    COVERAGE_DIRECTLY_SUPPORTED, COVERAGE_PARTIALLY_SUPPORTED,
    COVERAGE_UNSUPPORTED, COVERAGE_CONTRADICTORY,
    COVERAGE_INSUFFICIENT, COVERAGE_IRRELEVANT,
})


@dataclass(frozen=True)
class ClaimDecompositionProposal:
    """Propose splitting a broad claim into atomic child propositions.

    ``child_evidence_ids`` must reference evidence ALREADY linked to the
    source claim (the verifier checks this); children inherit that
    support. The reasoner may propose, the verifier + evaluator decide.
    """

    source_claim_id: str
    child_statements: tuple[str, ...]
    child_evidence_ids: tuple[str, ...]
    rationale: Optional[str] = None


@dataclass(frozen=True)
class EvidenceRelationProposal:
    """Propose a claim <-> evidence relationship."""

    claim_id: str
    evidence_id: str
    role: str  # 'supporting' | 'contradicting'


@dataclass(frozen=True)
class SemanticContradictionProposal:
    """OBSERVE that two claims are semantically contradictory.

    Observation only: the verifier records it without changing either
    claim's authoritative status; both findings are preserved.
    """

    claim_a_id: str
    claim_b_id: str
    evidence_a_ids: tuple[str, ...]
    evidence_b_ids: tuple[str, ...]
    rationale: str
    context: Optional[dict] = None
    strength: Optional[float] = None


@dataclass(frozen=True)
class ClaimContextProposal:
    """Propose scope/context qualifiers for a claim.

    Missing values are recorded as explicit None — NEVER interpreted as
    universal applicability."""

    claim_id: str
    context: dict  # keys subset of CONTEXT_KEYS; values str | None


@dataclass(frozen=True)
class CoverageProposal:
    """Propose an evidence-coverage observation for a claim.

    The deterministic verifier derives the authoritative coverage from
    persisted state; a proposal that contradicts derived state is
    rejected. ``irrelevant``/``insufficient`` are accepted labels for
    future reasoners but are never derived as "research gap"."""

    claim_id: str
    coverage: str  # one of COVERAGE_LABELS
    notes: Optional[str] = None


@runtime_checkable
class ReasonerPort(Protocol):
    """The reasoning capability C2 requires.

    All methods are pure: no DB writes, no state, no side effects.
    """

    def plan_queries(self, question: str, gaps: Sequence[Gap]) -> list[QueryProposal]:
        """Propose retrieval queries for the question and its gaps."""

    def construct_claims(self, evidence: Sequence[Evidence]) -> list[ClaimProposal]:
        """Propose claims derived from the given evidence."""

    def adjudicate(self, conflict: Conflict) -> Adjudication:
        """Propose a resolution for a supporting/contradicting conflict."""

    def synthesize(self, supported_claims: Sequence[SupportedClaim]) -> SynthesisDraft:
        """Draft the final synthesis from evaluator-approved claims."""

    # -- C3 proposal generation (all pure; the verifier decides) ----------

    def propose_decomposition(
        self, claim: "Claim"
    ) -> list[ClaimDecompositionProposal]:
        """Propose atomic child claims for a broad claim."""

    def propose_evidence_relations(
        self, claim: "Claim", evidence: Sequence[Evidence]
    ) -> list[EvidenceRelationProposal]:
        """Propose claim <-> evidence relationships."""

    def propose_semantic_contradictions(
        self, claim_a: "Claim", claim_b: "Claim"
    ) -> list[SemanticContradictionProposal]:
        """Observe a semantic contradiction between two claims (if any)."""

    def propose_context(self, claim: "Claim") -> Optional[ClaimContextProposal]:
        """Propose scope/context qualifiers for a claim."""

    def propose_coverage(self, claim: "Claim") -> Optional[CoverageProposal]:
        """Propose an evidence-coverage observation for a claim."""


class DeterministicReasoner:
    """Deterministic, reproducible MVP implementation of ReasonerPort."""

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _terms(text: str) -> list[str]:
        """Meaningful terms of a text (stopwords/short tokens removed)."""
        terms: list[str] = []
        for raw in _WORD_RE.findall(text):
            term = raw.lower()
            if term in STOPWORDS:
                continue
            if len(term) > 1 or term.isdigit():
                terms.append(term)
        return terms

    @staticmethod
    def _truncate_statement(text: str) -> str:
        """Cut at the last sentence boundary within the limit (hard cut
        otherwise). Deterministic."""
        if len(text) <= config.MAX_STATEMENT_CHARS:
            return text
        cut = text.rfind(". ", 0, config.MAX_STATEMENT_CHARS)
        if cut == -1:
            return text[: config.MAX_STATEMENT_CHARS]
        return text[: cut + 1]

    # -- ReasonerPort -----------------------------------------------------

    def plan_queries(self, question: str, gaps: Sequence[Gap]) -> list[QueryProposal]:
        """One query per source text (question first, then each gap),
        terms joined with spaces; deduped, capped and order-stable."""
        proposals: list[QueryProposal] = []
        seen: set[str] = set()
        for text in [question, *(gap.statement for gap in gaps)]:
            terms = self._terms(text)
            if not terms:
                continue
            query_text = " ".join(terms)
            if query_text in seen:
                continue
            seen.add(query_text)
            proposals.append(QueryProposal(query_text=query_text))
        return proposals[: config.MAX_QUERIES_PER_ITERATION]

    def construct_claims(self, evidence: Sequence[Evidence]) -> list[ClaimProposal]:
        """One verbatim quote-claim per evidence (truncated to the limit)."""
        proposals: list[ClaimProposal] = []
        for ev in evidence:
            proposals.append(
                ClaimProposal(
                    statement=self._truncate_statement(ev.content),
                    evidence_ids=(ev.evidence_id,),
                )
            )
        return proposals

    def adjudicate(self, conflict: Conflict) -> Adjudication:
        """Deterministic MVP rule: contradicting evidence wins until the
        conflict is explicitly resolved; otherwise supporting stands."""
        if conflict.contradicting_evidence_ids:
            return Adjudication(
                claim_id=conflict.claim_id,
                decision=DECISION_CONTRADICTION_WINS,
                note="deterministic rule: contradicting evidence present -> "
                "claim stays out of synthesis until resolved",
            )
        return Adjudication(
            claim_id=conflict.claim_id,
            decision=DECISION_SUPPORTING_STANDS,
            note="deterministic rule: no contradicting evidence -> supporting "
            "evidence stands",
        )

    def synthesize(self, supported_claims: Sequence[SupportedClaim]) -> SynthesisDraft:
        """Interface-validating MVP: deterministic ordering and summary.
        Full synthesis logic lands with the synthesis commit."""
        ordered = sorted(supported_claims, key=lambda c: c.claim_id)
        statements = tuple(
            {
                "claim_id": c.claim_id,
                "statement": c.statement,
                "evidence_ids": list(c.evidence_ids),
            }
            for c in ordered
        )
        return SynthesisDraft(
            summary=f"Deterministic synthesis of {len(ordered)} supported claim(s)",
            statements=statements,
        )

    # -- C3 proposals: deterministic baseline -------------------------------

    def propose_decomposition(
        self, claim: "Claim"
    ) -> list[ClaimDecompositionProposal]:
        """Baseline reasoner performs no semantic decomposition.

        Rich decomposition arrives with a future reasoner adapter; the
        verifier already supports the proposal type."""

        return []

    def propose_evidence_relations(
        self, claim: "Claim", evidence: Sequence[Evidence]
    ) -> list[EvidenceRelationProposal]:
        """Baseline: every evidence promotes a supporting relationship —
        the same deterministic policy the loop already applies."""
        return [
            EvidenceRelationProposal(
                claim_id=claim.claim_id, evidence_id=e.evidence_id,
                role="supporting",
            )
            for e in evidence
        ]

    def propose_semantic_contradictions(
        self, claim_a: "Claim", claim_b: "Claim"
    ) -> list[SemanticContradictionProposal]:
        """Baseline reasoner has no semantic insight; returns none."""
        return []

    def propose_context(self, claim: "Claim") -> Optional[ClaimContextProposal]:
        """Baseline reasoner proposes no context."""
        return None

    def propose_coverage(self, claim: "Claim") -> Optional[CoverageProposal]:
        """Baseline reasoner proposes no coverage (the verifier derives it
        deterministically from persisted state)."""
        return None
