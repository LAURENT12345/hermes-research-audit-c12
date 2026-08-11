"""C2 configuration: research-loop defaults (engineering constants).

These are defaults, not protocol rules. A session can override
``min_supporting`` and ``max_iterations`` at creation time; the rest are
loop-tuning constants used by later commits.
"""

from __future__ import annotations

#: Default number of valid supporting Evidence required for a claim to be
#: considered supported (sufficiency threshold, per session).
DEFAULT_MIN_SUPPORTING = 1

#: Evidence count that marks a claim as "strong" (reserved for C3).
MIN_STRONG_SUPPORTING = 2

#: Default cap on loop iterations per session.
DEFAULT_MAX_ITERATIONS = 5

#: How many loop iterations one ``start``/``iterate`` API call may run.
ITERATIONS_PER_REQUEST = 1

#: Max derived retrieval queries generated per planning phase.
MAX_QUERIES_PER_ITERATION = 3

#: Result limit passed to C0 search per query.
SEARCH_LIMIT_PER_QUERY = 10

#: Hard cap on claim statement length (characters).
MAX_STATEMENT_CHARS = 400

#: Per-session execution lease TTL (seconds). A crashed caller's lease
#: expires after this and the next caller may take over. Long enough that
#: a single iterate/start call completes well within it.
SESSION_LOCK_TTL_SECONDS = 300

# ---------------------------------------------------------------------------
# LLM reasoner adapter (Commit 12) — transport defaults.
#
# Credentials are NEVER configured here; they come from environment
# variables using the repository's canonical names (OPENCODE_GO_API_KEY /
# OPENCODE_GO_BASE_URL, OPENCODE_ZEN_API_KEY / OPENCODE_ZEN_BASE_URL).
# ---------------------------------------------------------------------------

#: Default provider for the LLM reasoner transport (OpenCode Go is the
#: repository's flat-namespace OpenAI-compatible endpoint).
LLM_DEFAULT_PROVIDER = "opencode-go"

#: Default model identifier. Empty means "not configured": LLMReasoner
#: fails closed at construction rather than guessing a model.
LLM_DEFAULT_MODEL = ""

#: Per-call model timeout (seconds).
LLM_DEFAULT_TIMEOUT_SECONDS = 60.0
