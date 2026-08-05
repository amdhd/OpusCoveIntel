"""Model-facing code.

Phase 4 ships only the embedding *seam* and a deterministic offline embedder.
The budget guard, response cache, cost tracker and provider adapters land in
Phase 5 -- guards first, adapters second, so nothing can spend before the
ceiling exists (PLAN.md, Phase 5).
"""
