"""Extraction: candidate detection, extractors, citation verification.

Phase 4 ships the deterministic half -- regex extraction and citation checking.
The Opus extractor, versioned Jinja2 prompts and the validation retry loop land
in Phase 6 and produce the same `RuleExtraction` shape, so the two can be
compared field by field (PLAN.md 3).
"""
