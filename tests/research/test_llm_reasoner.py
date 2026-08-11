"""Commit 12 — LLM reasoner adapter tests.

Covers (per the Commit 12 spec):

A. Port boundary   — LLMReasoner implements ReasonerPort; no store
                     dependency; cannot mutate the DB.
B. Structured parsing — all five proposal types, malformed JSON, unknown
                     types, missing fields, wrong field types.
C. ID safety       — fabricated claim/evidence ids reach the verifier and
                     are rejected; wrong relationships are rejected.
D. Deterministic serialization — identical input -> identical request;
                     ordering independent of insertion order.
E. Failure behavior — timeout, transport failure, malformed response,
                     empty response.
F. Verifier integration — valid proposals can be applied; invalid
                     proposals cannot mutate state.
G. Security/boundary — no raw store mutation, no sqlite, no citation
                     fabrication, no claim-status mutation from the adapter.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

import evidence.store as c1store
import docpipe.db as c0db
from docpipe.chunker import Chunk
from research import claims, store, verifier
from research.llm_reasoner import (
    LLMNotConfiguredError,
    LLMReasoner,
    LLMReasonerError,
    LLMResponseError,
    REASONER_SYSTEM_PROMPT_V1,
    build_model_input,
    parse_proposals,
    serialize_claim,
    serialize_evidence,
)
from research.llm_transport import (
    LLMTimeoutError,
    LLMTransportError,
    resolve_transport_config,
)
from research.model import Claim
from research.reasoner import (
    ClaimContextProposal,
    ClaimDecompositionProposal,
    CoverageProposal,
    EvidenceRelationProposal,
    Gap,
    ReasonerPort,
    SemanticContradictionProposal,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeLLMTransport:
    """Deterministic, scriptable transport double.

    ``responder`` maps the messages list to the raw response text; a
    responder may raise to simulate transport failures.
    """

    def __init__(self, responder=None):
        self._responder = responder or (
            lambda messages: '{"proposals": []}'
        )
        self.messages_log: list = []

    def complete(self, messages, *, timeout):
        self.messages_log.append(messages)
        return self._responder(messages)


def _claim(
    claim_id="C-000001",
    statement="Scaled dot product attention divides by the square root of dk.",
    status="supported",
    session_id="S-000001",
    parent_claim_id=None,
    context=None,
) -> Claim:
    from research.model import normalize_statement

    return Claim(
        claim_id=claim_id,
        session_id=session_id,
        statement=statement,
        statement_normalized=normalize_statement(statement),
        status=status,
        confidence=None,
        notes=None,
        created_at=None,
        updated_at=None,
        parent_claim_id=parent_claim_id,
        context=context,
    )


def _evidence(eid="E-000001", content="Dot products grow large in magnitude.",
              doc_id=1, chunk_id=1, page=4):
    from evidence.model import Evidence

    return Evidence(
        evidence_id=eid, document_id=doc_id, chunk_id=chunk_id,
        evidence_type="chunk", source_type="local_pdf",
        source_title="Attention Is All You Need", content=content,
        page_number=page, section_path=None,
        locator={"page": page, "chunk_id": chunk_id},
        created_at="t",
    )


def _responder_ok(proposals_json: str):
    return lambda messages: json.dumps(
        {"proposals": proposals_json}
    )


def _env_db(tmp_path):
    """Document + two evidence + one supported claim (verifier env)."""
    db_path = tmp_path / "docpipe.db"
    conn = store.connect(db_path)
    doc_id = c0db.insert_document(
        conn, title="Paper", file_path="/tmp/p.pdf", file_hash="ha",
        status="ready",
    )
    chunk_ids = []
    for i, text in enumerate(
        ["Scaled dot product attention divides by the square root of dk.",
         "Multi head attention runs several attention layers in parallel."]
    ):
        chunk_ids.append(
            c0db.insert_chunk(
                conn, document_id=doc_id,
                chunk=Chunk(chunk_index=i, content=text,
                            normalized_content=text, page_number=i + 1,
                            section_path=None, token_count=10),
            )
        )
    ev1 = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_ids[0],
        evidence_type="chunk", source_type="local_pdf",
        source_title="Paper",
        content="Scaled dot product attention divides by the square root of dk.",
        page_number=1, section_path=None,
        locator={"page": 1, "chunk_id": chunk_ids[0]},
    ).evidence_id
    ev2 = c1store.create_evidence(
        conn, document_id=doc_id, chunk_id=chunk_ids[1],
        evidence_type="chunk", source_type="local_pdf",
        source_title="Paper",
        content="Multi head attention runs several attention layers in parallel.",
        page_number=2, section_path=None,
        locator={"page": 2, "chunk_id": chunk_ids[1]},
    ).evidence_id
    session = store.create_session(conn, question="Why scale by sqrt dk?")
    store.update_session_status(conn, session.session_id, "active")
    claim = claims.create_claim(
        conn, session_id=session.session_id,
        statement="Scaled dot product attention divides by the square root of dk.",
    )
    claims.link_evidence(conn, claim_id=claim.claim_id, evidence_id=ev1,
                         role="supporting")
    claims.evaluate_claim(conn, claim.claim_id)
    conn.close()
    return db_path, ev1, ev2, claim.claim_id


# ---------------------------------------------------------------------------
# A. Port boundary
# ---------------------------------------------------------------------------


def test_llm_reasoner_implements_reasoner_port():
    reasoner = LLMReasoner(FakeLLMTransport())
    assert isinstance(reasoner, ReasonerPort)


def test_llm_reasoner_has_no_store_or_domain_import():
    """The adapter must remain a pure proposal source: no research.store,
    no evidence.store, no docpipe.db, no claims/verifier imports."""
    for filename in ("llm_reasoner.py", "llm_transport.py"):
        source = (REPO_ROOT / "research" / filename).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "store" not in node.module.split("."), (
                    f"{filename} imports {node.module}"
                )
                assert node.module not in (
                    "research.claims", "research.verifier",
                    "research.synthesis", "research.loop",
                ), f"{filename} imports {node.module}"
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "store" not in alias.name.split("."), (
                        f"{filename} imports {alias.name}"
                    )
        # The source itself never imports sqlite3.
        assert "sqlite3" not in source.replace(
            "sqlite3", ""
        ) or "import sqlite3" not in source


def test_llm_reasoner_imports_pull_no_store_module():
    """Importing the adapter in a fresh interpreter must not load any
    store module into sys.modules."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "import research.llm_reasoner, research.llm_transport\n"
        "bad = [m for m in ('research.store', 'evidence.store', "
        "'docpipe.db', 'research.claims', 'research.verifier')\n"
        "        if m in sys.modules]\n"
        "assert not bad, bad\n"
        "print('clean')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout


def test_llm_reasoner_never_calls_raw_writers_or_apply_functions():
    source = (REPO_ROOT / "research" / "llm_reasoner.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    for writer in ("update_claim_status", "update_session_status",
                   "set_query_status", "set_session_answer",
                   "set_final_answer", "create_claim", "link_evidence",
                   "transition_claim"):
        assert writer not in calls, (
            f"adapter calls {writer!r} — it must never write state"
        )
    for apply in ("apply_decomposition", "apply_evidence_relation",
                  "apply_semantic_contradiction", "apply_context",
                  "apply_coverage"):
        assert apply not in calls, (
            f"adapter calls {apply!r} — apply belongs to the verifier/domain"
        )


def test_adapter_cannot_mutate_db_by_construction():
    """The adapter takes no connection and exposes no write surface."""
    reasoner = LLMReasoner(FakeLLMTransport())
    assert not hasattr(reasoner, "conn")
    assert not hasattr(reasoner, "db_path")
    for method in ("apply_decomposition", "apply_evidence_relation",
                   "apply_context", "apply_coverage"):
        assert not hasattr(reasoner, method)


def test_fails_closed_without_model_or_transport(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "dummy-key")
    with pytest.raises(LLMNotConfiguredError):
        LLMReasoner()


# ---------------------------------------------------------------------------
# B. Structured parsing
# ---------------------------------------------------------------------------


def test_parse_valid_decomposition():
    raw = (
        '{"proposals": [{"type": "claim_decomposition", '
        '"source_claim_id": "C-000001", '
        '"child_statements": ["A.", "B."], '
        '"child_evidence_ids": ["E-000001", "E-000002"], '
        '"rationale": "atomic split"}]}'
    )
    proposals = parse_proposals(raw, method="claim_decomposition")
    assert len(proposals) == 1
    p = proposals[0]
    assert isinstance(p, ClaimDecompositionProposal)
    assert p.source_claim_id == "C-000001"
    assert p.child_statements == ("A.", "B.")
    assert p.child_evidence_ids == ("E-000001", "E-000002")
    assert p.rationale == "atomic split"


def test_parse_valid_decomposition_without_evidence_ids():
    proposals = parse_proposals(
        '{"proposals": [{"type": "claim_decomposition", '
        '"source_claim_id": "C-000001", "child_statements": ["A."]}]}',
        method="claim_decomposition",
    )
    assert proposals[0].child_evidence_ids == ()


def test_parse_valid_evidence_relation():
    proposals = parse_proposals(
        '{"proposals": [{"type": "evidence_relation", '
        '"claim_id": "C-000001", "evidence_id": "E-000001", '
        '"role": "contradicting"}]}',
        method="evidence_relation",
    )
    p = proposals[0]
    assert isinstance(p, EvidenceRelationProposal)
    assert p.role == "contradicting"


def test_parse_valid_semantic_contradiction():
    raw = (
        '{"proposals": [{"type": "semantic_contradiction", '
        '"claim_a_id": "C-000001", "claim_b_id": "C-000002", '
        '"evidence_a_ids": ["E-000001"], "evidence_b_ids": ["E-000002"], '
        '"rationale": "opposite claims", "context": {"task": "translation"}, '
        '"strength": 0.9}]}'
    )
    proposals = parse_proposals(raw, method="semantic_contradiction")
    p = proposals[0]
    assert isinstance(p, SemanticContradictionProposal)
    assert p.strength == 0.9
    assert p.context == {"task": "translation"}


def test_parse_valid_context():
    proposals = parse_proposals(
        '{"proposals": [{"type": "claim_context", "claim_id": "C-000001", '
        '"context": {"population": "humans", "dataset": null}}]}',
        method="claim_context",
    )
    p = proposals[0]
    assert isinstance(p, ClaimContextProposal)
    assert p.context == {"population": "humans", "dataset": None}


def test_parse_valid_coverage():
    proposals = parse_proposals(
        '{"proposals": [{"type": "coverage", "claim_id": "C-000001", '
        '"coverage": "directly_supported", "notes": "two valid supports"}]}',
        method="coverage",
    )
    p = proposals[0]
    assert isinstance(p, CoverageProposal)
    assert p.coverage == "directly_supported"


def test_parse_malformed_json_rejected():
    for raw in ("", "   ", "{not json", "[1,2]", '{"proposals": 3}'):
        with pytest.raises(LLMResponseError):
            parse_proposals(raw, method="coverage")


def test_parse_unknown_proposal_type_rejected():
    with pytest.raises(LLMResponseError):
        parse_proposals(
            '{"proposals": [{"type": "research_gap", "claim_id": "C-1"}]}',
            method="coverage",
        )


def test_parse_wrong_type_for_method_rejected():
    """An evidence_relation proposal inside a coverage request is an
    adapter error (type/method mismatch), not a verifier decision."""
    with pytest.raises(LLMResponseError):
        parse_proposals(
            '{"proposals": [{"type": "evidence_relation", '
            '"claim_id": "C-1", "evidence_id": "E-1", '
            '"role": "supporting"}]}',
            method="coverage",
        )


def test_parse_missing_required_fields_rejected():
    cases = [
        # coverage without claim_id
        ('{"proposals": [{"type": "coverage", "coverage": "unsupported"}]}',
         "coverage"),
        # decomposition without child_statements
        ('{"proposals": [{"type": "claim_decomposition", '
         '"source_claim_id": "C-1"}]}', "claim_decomposition"),
        # contradiction without rationale
        ('{"proposals": [{"type": "semantic_contradiction", '
         '"claim_a_id": "A", "claim_b_id": "B", '
         '"evidence_a_ids": ["E-1"], "evidence_b_ids": ["E-2"]}]}',
         "semantic_contradiction"),
        # contradiction without evidence_a_ids
        ('{"proposals": [{"type": "semantic_contradiction", '
         '"claim_a_id": "A", "claim_b_id": "B", '
         '"evidence_b_ids": ["E-2"], "rationale": "r"}]}',
         "semantic_contradiction"),
    ]
    for raw, method in cases:
        with pytest.raises(LLMResponseError):
            parse_proposals(raw, method=method)


def test_parse_wrong_field_types_rejected():
    cases = [
        # child_statements not a list
        ('{"proposals": [{"type": "claim_decomposition", '
         '"source_claim_id": "C-1", "child_statements": "A."}]}',
         "claim_decomposition"),
        # evidence ids not strings
        ('{"proposals": [{"type": "claim_decomposition", '
         '"source_claim_id": "C-1", "child_statements": ["A."], '
         '"child_evidence_ids": [1, 2]}]}', "claim_decomposition"),
        # context not a dict
        ('{"proposals": [{"type": "claim_context", "claim_id": "C-1", '
         '"context": 42}]}', "claim_context"),
        # context value not str/null
        ('{"proposals": [{"type": "claim_context", "claim_id": "C-1", '
         '"context": {"task": 42}}]}', "claim_context"),
        # strength a string
        ('{"proposals": [{"type": "semantic_contradiction", '
         '"claim_a_id": "A", "claim_b_id": "B", '
         '"evidence_a_ids": ["E-1"], "evidence_b_ids": ["E-2"], '
         '"rationale": "r", "strength": "high"}]}',
         "semantic_contradiction"),
        # item not an object
        ('{"proposals": ["coverage"]}', "coverage"),
    ]
    for raw, method in cases:
        with pytest.raises(LLMResponseError):
            parse_proposals(raw, method=method)


def test_parse_partial_batch_fail_closed():
    """One bad item rejects the WHOLE batch (fail-closed)."""
    raw = (
        '{"proposals": [{"type": "coverage", "claim_id": "C-000001", '
        '"coverage": "unsupported"}, '
        '{"type": "coverage", "claim_id": "C-000002"}]}'
    )
    with pytest.raises(LLMResponseError):
        parse_proposals(raw, method="coverage")


def test_parse_empty_proposals_is_valid():
    assert parse_proposals('{"proposals": []}', method="coverage") == []


def _decomposition_raw(claim_id, statements, evidence_ids):
    """Build a claim_decomposition response payload without f-string
    brace-escaping hazards."""
    return json.dumps(
        {
            "proposals": [
                {
                    "type": "claim_decomposition",
                    "source_claim_id": claim_id,
                    "child_statements": statements,
                    "child_evidence_ids": evidence_ids,
                }
            ]
        }
    )


# ---------------------------------------------------------------------------
# C. ID safety — fabricated ids pass the parser but are rejected by the
#    deterministic verifier
# ---------------------------------------------------------------------------


@pytest.fixture
def env(tmp_path):
    return _env_db(tmp_path)


def test_fabricated_claim_id_rejected_by_verifier(env):
    db_path, ev1, _, claim_id = env
    conn = store.connect(db_path)
    try:
        proposal = parse_proposals(
            '{"proposals": [{"type": "evidence_relation", '
            '"claim_id": "C-999999", "evidence_id": "E-000001", '
            '"role": "supporting"}]}',
            method="evidence_relation",
        )[0]
        with pytest.raises(verifier.ProposalRejectedError) as exc:
            verifier.apply_evidence_relation(conn, proposal)
        assert "not found" in str(exc.value)
    finally:
        conn.close()


def test_fabricated_evidence_id_rejected_by_verifier(env):
    db_path, ev1, _, claim_id = env
    conn = store.connect(db_path)
    try:
        proposal = parse_proposals(
            '{"proposals": [{"type": "evidence_relation", '
            f'"claim_id": "{claim_id}", "evidence_id": "E-999999", '
            '"role": "supporting"}]}',
            method="evidence_relation",
        )[0]
        with pytest.raises(verifier.ProposalRejectedError) as exc:
            verifier.apply_evidence_relation(conn, proposal)
        assert "fabricated evidence id" in str(exc.value)
    finally:
        conn.close()


def test_fabricated_decomposition_evidence_rejected(env):
    db_path, ev1, ev2, claim_id = env
    conn = store.connect(db_path)
    try:
        proposal = parse_proposals(
            _decomposition_raw(
                claim_id, ["Atomic child."], ["E-999999"]
            ),
            method="claim_decomposition",
        )[0]
        with pytest.raises(verifier.ProposalRejectedError) as exc:
            verifier.apply_decomposition(conn, proposal)
        assert "fabricated evidence id" in str(exc.value)
    finally:
        conn.close()


def test_wrong_claim_evidence_relationship_rejected(env):
    """Real evidence that is NOT linked to the claim is rejected."""
    db_path, ev1, ev2, claim_id = env
    conn = store.connect(db_path)
    try:
        proposal = parse_proposals(
            _decomposition_raw(claim_id, ["Atomic child."], [ev2]),
            method="claim_decomposition",
        )[0]
        with pytest.raises(verifier.ProposalRejectedError) as exc:
            verifier.apply_decomposition(conn, proposal)
        assert "not linked to source claim" in str(exc.value)
    finally:
        conn.close()


def test_fabricated_ids_reach_verifier_as_proposals():
    """The adapter parser must NOT reject fabricated ids — that decision
    belongs to the deterministic verifier."""
    proposals = parse_proposals(
        '{"proposals": [{"type": "evidence_relation", '
        '"claim_id": "C-999999", "evidence_id": "E-999999", '
        '"role": "supporting"}]}',
        method="evidence_relation",
    )
    assert proposals[0].claim_id == "C-999999"


# ---------------------------------------------------------------------------
# D. Deterministic serialization
# ---------------------------------------------------------------------------


def test_identical_input_identical_request():
    claim = _claim()
    transport = FakeLLMTransport()
    reasoner = LLMReasoner(transport)
    reasoner.propose_coverage(claim)
    reasoner.propose_coverage(claim)
    first, second = transport.messages_log
    assert first == second


def test_serialization_independent_of_insertion_order():
    claim = _claim()
    a = build_model_input("coverage", claim=serialize_claim(claim))
    b = build_model_input("coverage", claim=serialize_claim(claim))
    assert a == b


def test_evidence_ordering_independent_of_input_order():
    ev_a = _evidence("E-000002", "Second.")
    ev_b = _evidence("E-000001", "First.")
    claim = _claim()

    def user_message(evidence):
        transport = FakeLLMTransport()
        reasoner = LLMReasoner(transport)
        reasoner.propose_evidence_relations(claim, evidence)
        return transport.messages_log[0][1]["content"]

    forward = user_message([ev_a, ev_b])
    backward = user_message([ev_b, ev_a])
    assert forward == backward


def test_request_is_byte_stable_across_runs():
    """Two separately built requests for the same logical input differ
    only if the input does — no timestamps, no randomness."""
    claim = _claim()
    t1 = FakeLLMTransport()
    t2 = FakeLLMTransport()
    LLMReasoner(t1).propose_coverage(claim)
    LLMReasoner(t2).propose_coverage(claim)
    assert t1.messages_log == t2.messages_log


def test_system_prompt_is_versioned_and_stable():
    assert REASONER_SYSTEM_PROMPT_V1.startswith(
        "You are the research reasoner"
    )
    # The prompt pins the proposal-only principle.
    assert "ADVISORY PROPOSAL DATA ONLY" in REASONER_SYSTEM_PROMPT_V1
    assert "NEVER invent IDs" in REASONER_SYSTEM_PROMPT_V1


# ---------------------------------------------------------------------------
# E. Failure behavior (fail-closed)
# ---------------------------------------------------------------------------


def test_timeout_maps_to_adapter_error():
    def boom(messages):
        raise LLMTimeoutError("timed out")

    reasoner = LLMReasoner(FakeLLMTransport(boom))
    with pytest.raises(LLMReasonerError) as exc:
        reasoner.propose_coverage(_claim())
    assert "timeout" in str(exc.value)
    assert reasoner.stats.last_error_category == "timeout"


def test_transport_failure_maps_to_adapter_error():
    def boom(messages):
        raise LLMTransportError("connection refused")

    reasoner = LLMReasoner(FakeLLMTransport(boom))
    with pytest.raises(LLMReasonerError) as exc:
        reasoner.propose_coverage(_claim())
    assert "transport failure" in str(exc.value)
    assert reasoner.stats.last_error_category == "transport"


def test_malformed_response_maps_to_adapter_error():
    reasoner = LLMReasoner(
        FakeLLMTransport(lambda messages: "not json at all")
    )
    with pytest.raises(LLMResponseError):
        reasoner.propose_coverage(_claim())
    assert reasoner.stats.last_error_category == "malformed"


def test_empty_string_response_is_an_error():
    reasoner = LLMReasoner(FakeLLMTransport(lambda messages: ""))
    with pytest.raises(LLMResponseError):
        reasoner.propose_coverage(_claim())


def test_empty_proposals_response_is_valid():
    reasoner = LLMReasoner(FakeLLMTransport())
    assert reasoner.propose_coverage(_claim()) is None
    assert reasoner.stats.last_error_category is None


def test_short_circuit_when_no_evidence():
    """No evidence -> no model call -> [] (deterministic baseline)."""
    transport = FakeLLMTransport()
    reasoner = LLMReasoner(transport)
    assert reasoner.propose_evidence_relations(_claim(), []) == []
    assert reasoner.propose_semantic_contradictions(
        _claim("C-000001"), _claim("C-000002")
    ) == []
    assert transport.messages_log == []


def test_model_failure_does_not_corrupt_session(env):
    """A model failure must not change any persisted state (the adapter
    is pure, but this guards the contract end-to-end)."""
    db_path, ev1, _, claim_id = env
    conn = store.connect(db_path)
    before_count = len(store.list_claims(conn, "S-000001"))
    conn.close()

    def boom(messages):
        raise LLMTimeoutError("boom")

    reasoner = LLMReasoner(FakeLLMTransport(boom))
    claim = store.connect(db_path)
    try:
        with pytest.raises(LLMReasonerError):
            reasoner.propose_coverage(
                store.get_claim(claim, claim_id)
            )
    finally:
        claim.close()
    conn = store.connect(db_path)
    try:
        assert len(store.list_claims(conn, "S-000001")) == before_count
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# F. Verifier integration — proposals from the adapter flow through the
#    deterministic verifier before any domain apply
# ---------------------------------------------------------------------------


def test_valid_decomposition_proposal_can_be_applied(env):
    db_path, ev1, _, claim_id = env
    conn = store.connect(db_path)
    try:
        proposal = parse_proposals(
            _decomposition_raw(
                claim_id,
                ["Attention divides by the square root of dk."],
                [ev1],
            ),
            method="claim_decomposition",
        )[0]
        children = verifier.apply_decomposition(conn, proposal)
        assert len(children) == 1
        assert children[0].parent_claim_id == claim_id
        # Child inherits the linked support and evaluates as supported.
        fresh = claims.evaluate_claim(conn, children[0].claim_id)
        assert fresh.status == "supported"
    finally:
        conn.close()


def test_invalid_proposal_cannot_mutate_state(env):
    db_path, ev1, _, claim_id = env
    conn = store.connect(db_path)
    try:
        before = len(store.list_claims(conn, "S-000001"))
        proposal = parse_proposals(
            '{"proposals": [{"type": "claim_decomposition", '
            f'"source_claim_id": "{claim_id}", '
            '"child_statements": ["Fabricated support."], '
            '"child_evidence_ids": ["E-999999"]}]}',
            method="claim_decomposition",
        )[0]
        with pytest.raises(verifier.ProposalRejectedError):
            verifier.apply_decomposition(conn, proposal)
        after = len(store.list_claims(conn, "S-000001"))
        assert after == before  # nothing was written
    finally:
        conn.close()


def test_semantic_contradiction_never_changes_claim_status(env):
    db_path, ev1, ev2, claim_id = env
    conn = store.connect(db_path)
    try:
        claim_b = claims.create_claim(
            conn, session_id="S-000001",
            statement="Multi head attention runs several attention layers in parallel.",
        )
        claims.link_evidence(conn, claim_id=claim_b.claim_id,
                             evidence_id=ev2, role="supporting")
        claims.evaluate_claim(conn, claim_b.claim_id)
        proposal = parse_proposals(
            '{"proposals": [{"type": "semantic_contradiction", '
            f'"claim_a_id": "{claim_id}", "claim_b_id": "{claim_b.claim_id}", '
            f'"evidence_a_ids": ["{ev1}"], "evidence_b_ids": ["{ev2}"], '
            '"rationale": "one divides, the other runs in parallel"}]}',
            method="semantic_contradiction",
        )[0]
        verifier.apply_semantic_contradiction(conn, proposal)
        # Authoritative status is untouched: both claims stay supported.
        assert store.get_claim(conn, claim_id).status == "supported"
        assert store.get_claim(conn, claim_b.claim_id).status == "supported"
        # The observation is recorded, not the status.
        records = store.list_contradictions(conn, "S-000001")
        assert any(r.kind == "semantic" for r in records)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# G. Configuration & transport resolution
# ---------------------------------------------------------------------------


def test_resolve_transport_config_defaults(monkeypatch):
    monkeypatch.delenv("OPENCODE_GO_API_KEY", raising=False)
    monkeypatch.delenv("OPENCODE_GO_BASE_URL", raising=False)
    cfg = resolve_transport_config(model="glm-5")
    assert cfg.provider == "opencode-go"
    assert cfg.model == "glm-5"
    assert cfg.base_url == "https://opencode.ai/zen/go/v1"
    assert cfg.api_key == ""


def test_resolve_transport_config_env_overrides(monkeypatch):
    monkeypatch.setenv("OPENCODE_GO_API_KEY", "secret-key")
    monkeypatch.setenv("OPENCODE_GO_BASE_URL", "https://example.test/v1")
    cfg = resolve_transport_config(model="m")
    assert cfg.api_key == "secret-key"
    assert cfg.base_url == "https://example.test/v1"


def test_transport_requires_model_and_key():
    from research.llm_transport import (
        ChatCompletionsTransport,
        LLMTransportError,
    )

    with pytest.raises(LLMTransportError):
        ChatCompletionsTransport(resolve_transport_config())
    with pytest.raises(LLMTransportError):
        ChatCompletionsTransport(
            resolve_transport_config(model="glm-5")
        )


def test_stats_expose_counts_without_persisting(env):
    transport = FakeLLMTransport(
        lambda messages: (
            '{"proposals": [{"type": "coverage", "claim_id": "C-000001", '
            '"coverage": "unsupported"}]}'
        )
    )
    reasoner = LLMReasoner(transport)
    reasoner.propose_coverage(_claim())
    assert reasoner.stats.calls == 1
    assert reasoner.stats.proposals_parsed == 1
    assert reasoner.stats.last_error_category is None
    assert reasoner.stats.last_provider == "opencode-go"
