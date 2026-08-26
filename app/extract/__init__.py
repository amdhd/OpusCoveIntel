"""Extraction: candidate detection, extractors, citation verification, pipeline.

Phase 4 shipped the deterministic half -- regex extraction and citation checking.
Phase 6 adds the LLM extractor, versioned prompts, validation retry loop, parallel
rule+LLM extraction with disagreement detection, and review-queue routing.

The two extractors produce comparable results (docs/plan.md 3):
- RuleExtraction: from `rule_extractor.py` via regex patterns
- LLMCovenantExtraction: from `llm_extractor.py` via structured LLM output

The `ExtractionPipeline` in `pipeline.py` runs both and routes disagreements to
human review at no extra model cost.
"""
