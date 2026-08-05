"""Hybrid retrieval: vector kNN, Postgres full-text, and the fusion of both.

A layer CLAUDE.md 3 does not name, because the architecture sketch predates it.
It sits beside `ingest/` at the same level: `api -> services -> repositories`
still holds, and `retrieval/` is a service that owns no SQL of its own -- the
two legs are repository methods on `DocumentChunkRepository`.
"""
