"""The deterministic query path.

Phase 7 wraps this in a LangGraph agent, which today is also deterministic --
the model-authored synthesis this docstring once promised has not been built.
Phase 4 answers the same questions with no model at all: classification is
keyword rules, retrieval is hybrid search, breach decisions are the rules
engine, and the answer text is assembled from templates.

The two paths share the parts that read a question -- `classify` and
`covenant_type_in` -- so they cannot drift into answering the same question
differently. They already had, once, before the agent was given an entry point
and anyone could compare them.

That ordering is deliberate (docs/plan.md 6): building the deterministic path first
gives a baseline to measure LLM lift against, and means a budget bug in Phase 5
cannot take down a system that already works.
"""
