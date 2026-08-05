"""The deterministic query path.

Phase 7 wraps this in a LangGraph agent with an LLM synthesising the prose.
Phase 4 answers the same questions with no model at all -- classification is
keyword rules, retrieval is hybrid search, breach decisions are the rules
engine, and the answer text is assembled from templates.

That ordering is deliberate (PLAN.md 6): building the deterministic path first
gives a baseline to measure LLM lift against, and means a budget bug in Phase 5
cannot take down a system that already works.
"""
