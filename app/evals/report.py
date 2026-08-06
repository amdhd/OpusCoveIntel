"""Rendering an `EvalReport` to `evals/results/`.

Two artefacts per run, because they have two readers:

* **JSON** is the record. Machine-diffable, complete, and the thing a later run
  is compared against.
* **Markdown** is for the person deciding something -- most immediately, "is a
  cheaper extraction model good enough?", which is a field-F1 delta and nothing
  else (HANDOVER §4).

Both are written under a timestamped name and copied to `latest.*`, so a run
never overwrites its own history but a reader always has one path to open.

Rates print as `n/a` when they are undefined, never as `0.00`. A precision of
"nothing was predicted" and a precision of "everything predicted was wrong" are
opposite findings, and rendering both as zero loses the distinction exactly
where it decides what to do next.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from app.evals.harness import CORPUS_CAVEAT, EvalReport
from app.evals.metrics import Score

DEFAULT_RESULTS_DIR = Path("evals/results")


def write_report(report: EvalReport, *, directory: Path | None = None) -> tuple[Path, Path]:
    """Write the JSON and Markdown artefacts. Returns both paths."""
    target = directory or DEFAULT_RESULTS_DIR
    target.mkdir(parents=True, exist_ok=True)

    stamp = report.generated_at.strftime("%Y%m%dT%H%M%SZ")
    json_path = target / f"eval-{stamp}.json"
    markdown_path = target / f"eval-{stamp}.md"

    json_path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")

    (target / "latest.json").write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
    (target / "latest.md").write_text(markdown_path.read_text(encoding="utf-8"), encoding="utf-8")
    return json_path, markdown_path


def render_markdown(report: EvalReport) -> str:
    lines: list[str] = [
        "# Evaluation report",
        "",
        f"Generated {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC · "
        f"extraction model `{report.extraction_model}`",
        "",
        f"> {CORPUS_CAVEAT}",
        "",
    ]

    lines += _corpus_section(report)
    lines += _answers_section(report)
    lines += _extraction_section(report)
    lines += _agreement_section(report)
    lines += _cost_section(report)

    if report.errors:
        lines += ["## Errors", ""]
        lines += [f"- {error}" for error in report.errors]
        lines.append("")

    return "\n".join(lines)


# -- sections ---------------------------------------------------------------


def _corpus_section(report: EvalReport) -> list[str]:
    lines = ["## Corpus", ""]
    scored = ", ".join(document.name for document in report.documents_scored) or "none"
    lines.append(f"- Scored: {scored}")
    if report.documents_missing:
        lines.append(
            f"- **Not in the corpus** (labelled but never ingested): "
            f"{', '.join(report.documents_missing)}"
        )
    if report.documents_unlabelled:
        lines.append(
            f"- In the corpus but unlabelled, so not scored: "
            f"{', '.join(report.documents_unlabelled)}"
        )
    lines.append("")
    return lines


def _answers_section(report: EvalReport) -> list[str]:
    if not report.answers:
        return ["## Golden questions", "", "No read path was run.", ""]

    lines = ["## Golden questions", ""]
    lines.append(
        "| Path | Passed | Target | Met | Faithfulness | Citations verified | Refusal F1 |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for path, scores in sorted(report.answers.items()):
        lines.append(
            f"| {path} | {scores.passed}/{scores.total} | {scores.target} | "
            f"{'yes' if scores.meets_target else '**no**'} | "
            f"{_rate(scores.faithfulness)} | "
            f"{scores.citations_verified}/{scores.citations_checked} | "
            f"{_rate(scores.refusal.f1)} |"
        )
    lines.append("")

    for path, scores in sorted(report.answers.items()):
        lines += [f"### {path}", ""]
        lines.append("| Q | Result | Intent | Citations | Note |")
        lines.append("|---|---|---|---|---|")
        for result in scores.results:
            note = result.notes or (
                f"missing {result.missing_terms}" if result.missing_terms else ""
            )
            intent = (
                result.intent
                if result.intent == result.expected_intent
                else f"{result.intent} (expected {result.expected_intent})"
            )
            lines.append(
                f"| {result.id} | {'PASS' if result.passed else 'FAIL'} | {intent} | "
                f"{result.citations_verified}/{result.citations} | {note} |"
            )
        if scores.unsupported_answers:
            lines += [
                "",
                f"**{scores.unsupported_answers} answer(s) carried no citation and no "
                "structured tool.** That is the failure mode CLAUDE.md 1.5 exists to "
                "prevent; each one is a refusal that did not happen.",
            ]
        lines.append("")
    return lines


def _extraction_section(report: EvalReport) -> list[str]:
    if not report.documents_scored:
        return [
            "## Extraction",
            "",
            "No labelled document is in the corpus, so nothing was scored. "
            "Run `make ingest-corpus index extract-sample` first.",
            "",
        ]

    lines = ["## Extraction", ""]
    for method in report.methods():
        totals = report.field_totals(method)
        # Which documents this extractor actually ran on. Without it, an LLM
        # recall of 0.71 over the one document it was run against reads as a
        # corpus-wide number, and the corpus is three documents.
        scored_on = ", ".join(
            document.name for document in report.documents_scored if method in document.by_method
        )
        lines += [f"### method: `{method}`", "", f"Scored on: {scored_on}.", ""]
        lines.append("| Field | P | R | F1 | TP | FP | FN |")
        lines.append("|---|---|---|---|---|---|---|")
        for name, score in sorted(totals.items()):
            lines.append(_score_row(name, score))
        micro = _micro(totals.values())
        lines.append(_score_row("**micro-average**", micro))
        lines.append("")

        citations = [
            document.by_method[method].citations
            for document in report.documents_scored
            if method in document.by_method
        ]
        predicted = sum(item.predicted for item in citations)
        matched = sum(item.evidence_matched for item in citations)
        reverified = sum(item.quote_reverified for item in citations)
        labelled = sum(item.labels for item in citations)
        lines += [
            f"Citations: {matched}/{predicted} quote the labelled clause "
            f"(precision {_rate(_ratio(matched, predicted))}), "
            f"{matched}/{labelled} labels were cited "
            f"(recall {_rate(_ratio(matched, labelled))}), "
            f"{reverified}/{predicted} still verify against their chunk.",
            "",
        ]

        languages: dict[str, tuple[int, int]] = {}
        for document in report.documents_scored:
            scores = document.by_method.get(method)
            if scores is None:
                continue
            for language, (found, total) in scores.recall_by_language.items():
                seen, counted = languages.get(language, (0, 0))
                languages[language] = (seen + found, counted + total)
        if len(languages) > 1:
            summary = " · ".join(
                f"{language}: {found}/{total} ({_rate(_ratio(found, total))})"
                for language, (found, total) in sorted(languages.items())
            )
            lines += [f"Covenant recall by document language — {summary}", ""]

    lines += _call_schedule_section(report)
    return lines


def _call_schedule_section(report: EvalReport) -> list[str]:
    totals = report.call_schedule_totals()
    if not totals:
        return []
    extracted_by = ", ".join(report.call_schedule_methods()) or "nothing"
    lines = [
        "### call schedules",
        "",
        f"Extracted by: {extracted_by}. Not split by method — only the rule "
        "extractor reads redemption tables, so a per-method table would score "
        "the LLM path at zero for a step it does not have.",
        "",
        "| Field | P | R | F1 | TP | FP | FN |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [_score_row(name, score) for name, score in sorted(totals.items())]
    lines.append("")
    return lines


def _agreement_section(report: EvalReport) -> list[str]:
    agreement = report.agreement
    lines = ["## Rules vs LLM", ""]
    if not agreement.measurable:
        lines += [
            "No LLM extraction is in the database, so agreement is undefined. "
            f"({agreement.rule_covenants} rule-extracted covenant(s) present.) "
            "Run `opuscovintel extract <document-id>` to make this measurable — "
            "it spends money.",
            "",
        ]
        return lines
    lines += [
        f"- LLM covenants: {agreement.llm_covenants}",
        f"- Rule covenants: {agreement.rule_covenants}",
        f"- Material disagreements routed to review: {agreement.disagreements}",
        f"- Agreement rate: {_rate(agreement.agreement_rate)}",
        "",
    ]
    return lines


def _cost_section(report: EvalReport) -> list[str]:
    cost = report.cost
    lines = ["## Cost", ""]
    if not cost.has_spend:
        lines += [
            "No provider call has been made against this database. "
            f"Ceilings stand at ${cost.budget_total_usd} total, "
            f"${cost.budget_per_document_usd} per document.",
            "",
        ]
        return lines

    lines += [
        # Two different caches, named apart because they are: `cache_hits` is
        # our own `llm_cache` table (a call that never left the process),
        # `cache_read_tokens` is the provider's prompt cache (a call that left,
        # and was billed at 0.1x on its prefix).
        f"- Total: **${cost.total_usd}** across {cost.calls} call(s) "
        f"(response-cache hits {cost.cache_hits}, rate {_rate(cost.cache_hit_rate)})",
        f"- Cost per document with spend: "
        f"{('$' + str(cost.cost_per_document)) if cost.cost_per_document else 'n/a'}",
        f"- Budget remaining: ${cost.budget_total_usd - cost.total_usd} of "
        f"${cost.budget_total_usd}",
    ]
    # PLAN.md 2: cache_read_tokens of zero across repeated extractions means a
    # silent cache invalidator in the prompt prefix. Say it here, where someone
    # is already reading the cost numbers.
    if cost.cache_read_tokens == 0:
        lines.append(
            "- **Prompt-cache reads: 0.** Across repeated extractions that means "
            "the cached prefix is being invalidated — a bug, not a tuning issue "
            "(PLAN.md 2)."
        )
    else:
        lines.append(
            f"- Provider prompt-cache read tokens: {cost.cache_read_tokens:,} "
            "(billed at 0.1x — the prefix is caching)"
        )
    lines.append("")

    if cost.by_stage:
        lines += ["| Stage | Spend |", "|---|---|"]
        lines += [f"| {stage} | ${amount} |" for stage, amount in sorted(cost.by_stage.items())]
        lines.append("")

    if cost.by_document:
        lines += ["| Document | Calls | Spend |", "|---|---|---|"]
        for item in cost.by_document:
            lines.append(f"| {item.filename} | {item.calls} | ${item.cost_usd} |")
        lines.append("")

    if cost.over_budget_documents:
        names = ", ".join(item.filename for item in cost.over_budget_documents)
        lines += [
            f"**Over the per-document ceiling of ${cost.budget_per_document_usd}: {names}.**",
            "",
        ]
    return lines


# -- formatting -------------------------------------------------------------


def _score_row(name: str, score: Score) -> str:
    return (
        f"| {name} | {_rate(score.precision)} | {_rate(score.recall)} | {_rate(score.f1)} | "
        f"{score.true_positives} | {score.false_positives} | {score.false_negatives} |"
    )


def _micro(scores: Iterable[Score]) -> Score:
    total = Score(name="micro")
    for score in scores:
        total = total + score
    return total


def _rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None
