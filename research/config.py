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
