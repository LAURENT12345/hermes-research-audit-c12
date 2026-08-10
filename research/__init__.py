"""C2 — Hermes Research Engine.

Planned scope (commit-by-commit, per the approved design):

1. schema + model + store      <- THIS COMMIT
2. retriever + reasoner ports
3. claims + sufficiency
4. contradictions + validation flow
5. loop state machine
6. synthesis + answer
7. API endpoints
8. CLI
9. black-box acceptance
10. docs + regression

This commit adds only the persistence foundation: the ``research_*``
tables in the shared C0/C1 SQLite database, the data model, and the
store layer. No loop, no claims logic, no API, no LLM.
"""

__version__ = "0.1.0"
