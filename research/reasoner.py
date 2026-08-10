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
