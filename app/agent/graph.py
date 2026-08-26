"""LangGraph query agent — the Phase 7 read path.

docs/plan.md 5:

    classify_intent → plan → retrieve → tools → rules_eval → synthesize → verify → log
                              │                                                │
                              └──── insufficient evidence ──→ refuse ──────────┘

**No node in this graph calls a model.** `_synthesize` formats tool results
into prose with Python; the module imports nothing from `app/llm/`. CLAUDE.md's
routing table assigns answer synthesis to `claude-opus-5` at `effort: medium`,
and that remains the intended design — but it is not what runs today, and this
docstring used to say otherwise.

What the graph adds over the Phase 4 service is therefore structure rather than
language: an intent-directed plan, tool orchestration, the verify node, and a
logged and audited record of every question.

Two things stay deterministic even if synthesis is later handed to a model:
breach checks and portfolio queries (CLAUDE.md 1.1 — the LLM never computes a
breach), and the verify node, which is the guard against a fluent wrong answer
and is worth more, not less, once prose is model-authored.

Session access: nodes receive their AsyncSession via `config["configurable"]`,
which keeps the graph provider-agnostic and the nodes testable. There are two
of them, under `READ_SESSION_KEY` and `WRITE_SESSION_KEY` — see the comment on
those constants for why one session cannot serve both.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from decimal import Decimal
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.tools import (
    ToolResult,
    evaluate_covenant_rule,
    get_call_schedules,
    get_covenants,
    get_instrument,
    get_portfolio_holdings,
    list_documents,
    run_read_only_sql,
    search_clauses,
)
from app.core.logging import get_logger
from app.db.models.ops import AuditLog, QueryLog
from app.domain.enums import ActorType, QueryIntent
from app.domain.rules import Citation
from app.query.answerable import STRUCTURED_INTENTS, refusal_for, unsupported_terms
from app.query.intent import (
    classify,
    mentioned_documents,
    mentioned_entities,
    name_words,
)
from app.query.service import (
    NO_EVIDENCE,
    UNSUPPORTED_MESSAGE,
    covenant_type_in,
    no_covenants_for,
)

# RunnableConfig is langgraph's config dict. Defined here to keep imports
# clean — langgraph.types does not explicitly export it for mypy.
RunnableConfig = dict[str, Any]

logger = get_logger(__name__)

# -- graph state -------------------------------------------------------------


@dataclass
class AgentState:
    """State that flows through every node of the graph."""

    question: str = ""
    intent: QueryIntent = QueryIntent.UNSUPPORTED
    user_id: str | None = None
    request_id: str | None = None

    # Retrieval
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)

    # Reasoning
    plan: str = ""
    evidence_sufficient: bool = False
    # Words in the question that the structured path has no meaning for. Set by
    # `_retrieve`; non-empty is a refusal (app/query/answerable.py).
    unsupported_terms: list[str] = field(default_factory=list)
    # A refusal decided during retrieval, with the text to answer with. Set
    # when the question named a document the corpus holds but has not extracted
    # (docs/review.md, finding 15).
    refusal: str | None = None

    # Output
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False

    # Logging
    tools_called: list[str] = field(default_factory=list)
    sql_generated: str | None = None


# The graph runs against two sessions, because it has two jobs with opposite
# privilege requirements (CLAUDE.md 1.6).
#
#   READ_SESSION_KEY  -- retrieval, tools, and every generated SQL statement.
#                        Bound to the read-only role, so a stray write fails at
#                        the database rather than at code review. This is the
#                        session the invariant is about.
#   WRITE_SESSION_KEY -- `query_logs` and `audit_logs`, and nothing else. These
#                        are writes by definition, so they cannot go through the
#                        read-only role.
#
# One session cannot be both. Running the whole graph read-write to make the log
# work would silently drop the invariant; running it all read-only would make
# the audit trail fail. Splitting them is the only option that keeps both.
READ_SESSION_KEY = "session"
WRITE_SESSION_KEY = "log_session"


def _session_from_config(
    config: RunnableConfig, key: str = READ_SESSION_KEY
) -> AsyncSession | None:
    """Extract a session from LangGraph's RunnableConfig."""
    configurable: dict[str, Any] = config.get("configurable", {})
    result: object = configurable.get(key)
    if isinstance(result, AsyncSession):
        return result
    return None


def _log_session_from_config(config: RunnableConfig) -> tuple[AsyncSession | None, bool]:
    """The session the log node writes through, and whether it owns it.

    Falls back to the read session when no separate one was supplied, which is
    what a single-session caller (a test, or a deployment that has not split
    the roles yet) gets. `owned` says whether this session exists solely for
    logging: it decides whether the log node may commit, since committing a
    session it shares with the caller would end the caller's transaction.
    """
    write = _session_from_config(config, WRITE_SESSION_KEY)
    if write is not None:
        return write, True
    return _session_from_config(config, READ_SESSION_KEY), False


# -- node implementations ----------------------------------------------------


async def _classify_intent(state: AgentState) -> AgentState:
    """Classify the question into one of the six intents.

    Uses the free, deterministic classifier from Phase 4. Phase 8 may add an
    LLM override for truly ambiguous cases, but the rules cover the MVP space.
    """
    state.intent = classify(state.question)
    logger.info(
        "agent.classify_intent",
        extra={"intent": state.intent.value, "chars": len(state.question)},
    )
    return state


async def _plan(state: AgentState) -> AgentState:
    """Decide what to do based on the intent — a lookup table, not a model call."""
    match state.intent:
        case QueryIntent.UNSUPPORTED:
            state.plan = "refuse"
        case QueryIntent.COVENANT_BREACH_CHECK:
            state.plan = "retrieve → evaluate → deterministic_answer"
        case QueryIntent.PORTFOLIO_QUERY:
            state.plan = "retrieve → sql → deterministic_answer"
        case QueryIntent.INSTRUMENT_LOOKUP:
            state.plan = "retrieve → deterministic_answer"
        case QueryIntent.COVENANT_LOOKUP:
            state.plan = "retrieve → synthesize"
        case QueryIntent.DOCUMENT_SEARCH:
            state.plan = "retrieve → synthesize"
        case _:
            state.plan = "retrieve → deterministic_answer"

    logger.info("agent.plan", extra={"intent": state.intent.value, "plan": state.plan})
    return state


async def _retrieve(state: AgentState, config: RunnableConfig) -> AgentState:
    """Execute the retrieval step using deterministic tools."""
    session = _session_from_config(config)
    if session is None:
        state.answer = "Internal error: no database session available."
        state.refused = True
        return state

    match state.intent:
        case QueryIntent.COVENANT_BREACH_CHECK:
            result = await get_instrument(session)
            state.tool_results.append(result)
            state.tools_called.append("get_instrument")

        case QueryIntent.PORTFOLIO_QUERY:
            result = await get_portfolio_holdings(session)
            state.tool_results.append(result)
            state.tools_called.append("get_portfolio_holdings")

        case QueryIntent.INSTRUMENT_LOOKUP:
            result = await get_instrument(session)
            state.tool_results.append(result)
            state.tools_called.append("get_instrument")

        case QueryIntent.COVENANT_LOOKUP:
            # Narrow to what the question names: the covenant type, and the
            # document or instrument it is about. Retrieving all of them and
            # letting synthesis sort it out returns every covenant in the
            # corpus for a question about one of them, which reads as an answer
            # and is not one -- and, when the named document has no extracted
            # covenants, hands back another document's thresholds under this
            # document's name (docs/review.md, finding 15).
            named = await _named_sources(session, state.question)
            state.tools_called.append("list_documents")

            if named.document_without_covenants is not None:
                state.refusal = no_covenants_for(named.document_without_covenants)
                logger.info(
                    "agent.named_document_has_no_covenants",
                    extra={"document": named.document_without_covenants},
                )
                return state

            result = await get_covenants(
                session,
                covenant_type=covenant_type_in(state.question),
                document_ids=named.document_ids or None,
                instrument_ids=named.instrument_ids or None,
            )
            state.tool_results.append(result)
            state.tools_called.append("get_covenants")
            state.citations.extend(result.citations)

        case QueryIntent.DOCUMENT_SEARCH:
            result = await search_clauses(session, state.question)
            state.tool_results.append(result)
            state.tools_called.append("search_clauses")
            state.citations.extend(result.citations)

        case _:
            result = await search_clauses(session, state.question)
            state.tool_results.append(result)
            state.tools_called.append("search_clauses")
            state.citations.extend(result.citations)

    state.retrieved_chunk_ids = [c.chunk_id for c in state.citations if c.chunk_id]

    state.evidence_sufficient = bool(state.citations) or any(
        r.ok and r.data and r.data.get("count", 0) > 0 for r in state.tool_results
    )

    # Rows were found; whether they answer *this* question is a separate
    # question, and for the structured intents nothing downstream asks it. An
    # instrument lookup returns instruments however far the question is from
    # anything an instrument row records (app/query/answerable.py).
    if state.intent in STRUCTURED_INTENTS:
        state.unsupported_terms = list(
            unsupported_terms(state.question, known_names=_known_names(state.tool_results))
        )
        if state.unsupported_terms:
            state.evidence_sufficient = False
            logger.info(
                "agent.unsupported_terms",
                extra={"intent": state.intent.value, "terms": state.unsupported_terms},
            )

    if not state.evidence_sufficient and state.intent not in (
        QueryIntent.UNSUPPORTED,
        QueryIntent.COVENANT_BREACH_CHECK,
    ):
        logger.info("agent.insufficient_evidence", extra={"intent": state.intent.value})

    # Narrow to what the question actually named. The tools fetch everything
    # and the formatters print whatever they are handed, so a question about
    # one instrument was answered with the whole universe -- and a question
    # about one fund's exposure with another fund's holdings in the total.
    #
    # After the vocabulary check above, which needs every name to decide
    # whether the question is answerable at all.
    _narrow_to_named(state)

    return state


def _narrow_to_named(state: AgentState) -> None:
    """Drop rows the question did not ask about, in place.

    Only when the question names something. Naming nothing is a question about
    the whole book -- "which holdings breach their rating trigger?" -- and
    narrowing that to nothing would refuse the flagship query.
    """
    narrowed: list[ToolResult] = []
    for result in state.tool_results:
        rows = result.data if result.ok and isinstance(result.data, dict) else None
        if rows is None:
            narrowed.append(result)
            continue

        if instruments := rows.get("instruments"):
            names = [item.instrument_name for item in instruments]
            names += [item.issuer_name for item in instruments]
            mentioned = set(mentioned_entities(state.question, names))
            kept = [
                item
                for item in instruments
                if item.instrument_name in mentioned or item.issuer_name in mentioned
            ]
        elif holdings := rows.get("holdings"):
            names = [str(item.get("portfolio_name", "")) for item in holdings]
            names += [str(item.get("instrument_name", "")) for item in holdings]
            names += [str(item.get("issuer_name", "")) for item in holdings]
            mentioned = set(mentioned_entities(state.question, [n for n in names if n]))
            kept = [
                item
                for item in holdings
                if mentioned
                & {
                    str(item.get("portfolio_name", "")),
                    str(item.get("instrument_name", "")),
                    str(item.get("issuer_name", "")),
                }
            ]
        else:
            narrowed.append(result)
            continue

        if not kept or len(kept) == len(rows.get("instruments") or rows.get("holdings") or []):
            narrowed.append(result)
            continue

        key = "instruments" if "instruments" in rows else "holdings"
        narrowed.append(
            replace(result, data={**rows, key: kept, "count": len(kept)}),
        )
        logger.info(
            "agent.narrowed_to_named_entities",
            extra={"intent": state.intent.value, "kept": len(kept), "of": len(rows[key])},
        )

    state.tool_results = narrowed


@dataclass
class _NamedSources:
    """What a covenant question named, resolved against the corpus."""

    document_ids: list[uuid.UUID] = field(default_factory=list)
    instrument_ids: list[uuid.UUID] = field(default_factory=list)
    # Set when the question named exactly one document and that document has no
    # extracted covenants -- the case that must refuse rather than widen.
    document_without_covenants: str | None = None


async def _named_sources(session: AsyncSession, question: str) -> _NamedSources:
    """Resolve the documents and instruments a question names.

    Naming nothing resolves to nothing, and the caller then searches the whole
    corpus: "what cross-default thresholds do we have?" is a question about the
    book, and narrowing it to nothing would be a worse answer than the one this
    fixes.
    """
    named = _NamedSources()

    instruments = (await get_instrument(session)).data["instruments"]
    names = [item.instrument_name for item in instruments]
    names += [item.issuer_name for item in instruments]
    wanted = set(mentioned_entities(question, names))

    documents = (await list_documents(session)).data["documents"]
    by_filename = {item["filename"]: item["id"] for item in documents}
    # A word that names an instrument is not a word that names a document, even
    # when it happens to appear in exactly one filename.
    mentioned = mentioned_documents(
        question, list(by_filename), reserved=set(name_words(" ".join(names)))
    )
    named.document_ids = [by_filename[filename] for filename in mentioned]
    named.instrument_ids = [
        item.id
        for item in instruments
        if item.instrument_name in wanted or item.issuer_name in wanted
    ]

    # One named document with nothing extracted from it. Checked here rather
    # than by looking at an empty result, because an empty result is also what
    # a question about a covenant type nobody has looks like, and those two
    # deserve different answers.
    if len(named.document_ids) == 1 and not named.instrument_ids:
        held = await get_covenants(session, document_ids=named.document_ids)
        if held.data["count"] == 0:
            named.document_without_covenants = mentioned[0]

    return named


def _known_names(results: list[ToolResult]) -> list[str]:
    """Names the database holds, harvested from what retrieval already loaded.

    A question is free to name an instrument, an issuer or a portfolio, and
    those words are known even though no vocabulary could list them. Taken from
    the tool results rather than queried again: the rows are already in hand.
    """
    names: list[str] = []
    for result in results:
        if not (result.ok and result.data):
            continue
        for instrument in result.data.get("instruments", []):
            names.append(instrument.instrument_name)
            names.append(instrument.issuer_name)
        for holding in result.data.get("holdings", []):
            names.extend(
                str(holding.get(key, ""))
                for key in ("instrument_name", "issuer_name", "portfolio_name")
            )
    return [name for name in names if name]


async def _tools(state: AgentState, config: RunnableConfig) -> AgentState:
    """Run additional tools based on intent."""
    session = _session_from_config(config)
    if session is None:
        return state

    match state.intent:
        case QueryIntent.COVENANT_BREACH_CHECK:
            instruments_data = None
            for r in state.tool_results:
                if r.tool_name == "get_instrument" and r.ok and r.data:
                    instruments_data = r.data
                    break

            if instruments_data:
                for inst in instruments_data.get("instruments", []):
                    result = await evaluate_covenant_rule(session, instrument_id=inst.id)
                    state.tool_results.append(result)
                    state.tools_called.append("evaluate_covenant_rule")

        case QueryIntent.PORTFOLIO_QUERY:
            sql = _portfolio_sql(state.question)
            if sql:
                result = await run_read_only_sql(session, sql)
                state.tool_results.append(result)
                state.tools_called.append("run_read_only_sql")
                state.sql_generated = sql

        case QueryIntent.COVENANT_LOOKUP:
            result = await get_call_schedules(session)
            state.tool_results.append(result)
            state.tools_called.append("get_call_schedules")

    return state


async def _rules_eval(state: AgentState) -> AgentState:
    """Pass-through node — breach evaluation already happened in _tools.

    Exists as a separate node so the graph topology matches docs/plan.md, which
    matters for logging and future LLM-as-judge work.
    """
    return state


async def _synthesize(state: AgentState) -> AgentState:
    """Produce the final answer from tool results, deterministically."""
    # Checked before the intent, because every branch below would otherwise
    # format the rows it happens to hold into an answer to a question those
    # rows cannot address.
    if state.refusal:
        state.answer = state.refusal
        state.refused = True
        state.confidence = 0.0
        return state

    if state.unsupported_terms:
        state.answer = refusal_for(state.unsupported_terms)
        state.refused = True
        state.confidence = 0.0
        return state

    match state.intent:
        case QueryIntent.UNSUPPORTED:
            state.answer = UNSUPPORTED_MESSAGE
            state.refused = True
            state.confidence = 0.0

        case QueryIntent.COVENANT_BREACH_CHECK:
            state.answer, state.confidence = _format_breach_answer(state)
            state.tools_called.append("synthesize")

        case QueryIntent.PORTFOLIO_QUERY:
            state.answer, state.confidence = _format_portfolio_answer(state)
            state.tools_called.append("synthesize")

        case QueryIntent.INSTRUMENT_LOOKUP:
            state.answer, state.confidence = _format_instrument_answer(state)
            state.tools_called.append("synthesize")

        case QueryIntent.COVENANT_LOOKUP:
            state.answer, state.confidence = _format_covenant_answer(state)
            state.tools_called.append("synthesize")

        case QueryIntent.DOCUMENT_SEARCH:
            state.answer, state.confidence = _format_search_answer(state)
            state.tools_called.append("synthesize")

        case _:
            state.answer = NO_EVIDENCE
            state.refused = True
            state.confidence = 0.0

    return state


async def _verify(state: AgentState) -> AgentState:
    """Verify that every factual claim in the answer is traceable to a citation.

    docs/plan.md 5: "every factual claim in the drafted answer must map to a clause_id
    that was actually retrieved this turn."
    """
    if state.refused:
        return state

    if state.refusal:
        state.answer = state.refusal
        state.refused = True
        state.confidence = 0.0
        return state

    # The same signal `_synthesize` acts on, enforced again at the last gate.
    # Deliberate duplication: synthesize is a per-intent match statement and is
    # where a new intent will be added, and an intent added without its refusal
    # branch must not be able to answer a question the data cannot address.
    if state.unsupported_terms:
        state.answer = refusal_for(state.unsupported_terms)
        state.refused = True
        state.confidence = 0.0
        logger.warning(
            "agent.verify_caught_unsupported_answer",
            extra={"intent": state.intent.value, "terms": state.unsupported_terms},
        )
        return state

    if not state.citations and state.intent not in (
        QueryIntent.UNSUPPORTED,
        QueryIntent.PORTFOLIO_QUERY,
        QueryIntent.INSTRUMENT_LOOKUP,
        QueryIntent.COVENANT_BREACH_CHECK,
    ):
        state.answer = NO_EVIDENCE
        state.refused = True
        state.confidence = 0.0

    return state


async def _log(state: AgentState, config: RunnableConfig) -> AgentState:
    """Write a QueryLog row and an audit entry through the read-write session.

    This is the only node that writes, and the only one that may not use the
    read-only session. When it owns a dedicated logging session it commits it
    itself -- nothing else will, and an unflushed audit trail is the same as no
    audit trail. When it is sharing the caller's session it only flushes, and
    the caller's transaction boundary stands.
    """
    session, owned = _log_session_from_config(config)
    if session is None:
        return state

    try:
        query_log = QueryLog(
            user_id=state.user_id,
            question=state.question,
            intent=state.intent,
            retrieved_chunk_ids=state.retrieved_chunk_ids,
            tools_called=state.tools_called,
            sql_generated=state.sql_generated,
            answer=state.answer,
            citations_json=[c.model_dump(mode="json") for c in state.citations],
            confidence=state.confidence,
            refused=state.refused,
            request_id=state.request_id,
        )
        session.add(query_log)
        await session.flush()

        audit = AuditLog(
            actor_type=ActorType.AGENT,
            actor_id=f"agent:{state.user_id or 'anonymous'}",
            action="query",
            entity_type="query_log",
            entity_id=query_log.id,
            payload_json={
                "question": state.question,
                "intent": state.intent.value,
                "tools_called": state.tools_called,
                "confidence": state.confidence,
                "refused": state.refused,
            },
            request_id=state.request_id,
        )
        session.add(audit)
        await session.flush()

        if owned:
            await session.commit()

        logger.info(
            "agent.query_logged",
            extra={
                "intent": state.intent.value,
                "refused": state.refused,
                "confidence": state.confidence,
                "citations": len(state.citations),
                "committed": owned,
            },
        )
    except Exception as exc:  # noqa: BLE001
        # A failed log must not lose the answer, but it must not leave a
        # half-written transaction behind either.
        logger.error("agent.log_failed", extra={"error": str(exc)})
        if owned:
            try:
                await session.rollback()
            except Exception as rollback_exc:  # noqa: BLE001
                logger.error("agent.log_rollback_failed", extra={"error": str(rollback_exc)})

    return state


# -- edge routing ------------------------------------------------------------


def _route_after_plan(state: AgentState) -> Literal["refuse", "retrieve"]:
    if state.intent is QueryIntent.UNSUPPORTED:
        return "refuse"
    return "retrieve"


def _route_after_retrieve(state: AgentState) -> Literal["insufficient", "tools"]:
    # A question the data cannot address skips the tools entirely -- there is
    # no point evaluating covenants or running portfolio SQL for it, and the
    # breach check's exemption below must not let it through.
    if state.unsupported_terms or state.refusal:
        return "insufficient"
    if not state.evidence_sufficient and state.intent not in (
        QueryIntent.COVENANT_BREACH_CHECK,
        QueryIntent.UNSUPPORTED,
    ):
        return "insufficient"
    return "tools"


# -- graph construction ------------------------------------------------------


def build_graph() -> CompiledStateGraph[AgentState, None, AgentState, AgentState]:
    """Build and compile the LangGraph agent graph.

    Returns a compiled graph. The session is injected via
    `config["configurable"]["session"]` at invocation time.
    """
    builder: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)

    builder.add_node("classify_intent", _classify_intent)
    builder.add_node("plan", _plan)
    builder.add_node("retrieve", _retrieve)  # type: ignore[arg-type]
    builder.add_node("tools", _tools)  # type: ignore[arg-type]
    builder.add_node("rules_eval", _rules_eval)
    builder.add_node("synthesize", _synthesize)
    builder.add_node("verify", _verify)
    builder.add_node("log", _log)  # type: ignore[arg-type]

    builder.set_entry_point("classify_intent")
    builder.add_edge("classify_intent", "plan")

    builder.add_conditional_edges(
        "plan",
        _route_after_plan,
        {"refuse": "synthesize", "retrieve": "retrieve"},
    )

    builder.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {"insufficient": "synthesize", "tools": "tools"},
    )

    builder.add_edge("tools", "rules_eval")
    builder.add_edge("rules_eval", "synthesize")
    builder.add_edge("synthesize", "verify")
    builder.add_edge("verify", "log")
    builder.add_edge("log", END)

    return builder.compile()


# -- answer formatters (deterministic) ---------------------------------------


def _format_breach_answer(state: AgentState) -> tuple[str, float]:
    """Assemble breach-check results from tool outputs."""
    lines: list[str] = []
    breaching = 0
    at_risk = 0
    insufficient = 0
    instrument_count = 0

    for r in state.tool_results:
        tool_name: str = getattr(r, "tool_name", "")
        ok: bool = bool(getattr(r, "ok", False))
        data: object = getattr(r, "data", None)

        if tool_name == "get_instrument" and ok and data:
            instrument_count = data.get("count", 0) if isinstance(data, dict) else 0
        elif tool_name == "evaluate_covenant_rule" and ok and data:
            data_dict: dict[str, Any] = data if isinstance(data, dict) else {}
            inst_name = data_dict.get("instrument_name", "Unknown")
            for ev in data_dict.get("evaluations", []):
                status = ev["status"]
                lines.append(
                    f"{inst_name} [{status.upper()}] {ev['covenant_type']}: {ev['explanation']}"
                )
                if status == "breach":
                    breaching += 1
                elif status == "at_risk":
                    at_risk += 1
                elif status == "insufficient_data":
                    insufficient += 1

    if not lines:
        return NO_EVIDENCE, 0.0

    headline = (
        f"{breaching} covenant breach(es) found across {instrument_count} instrument(s)."
        if breaching
        else f"No covenant breaches found across {instrument_count} instrument(s)."
    )
    if at_risk:
        headline += f" {at_risk} at risk."
    body = "\n".join(f"  - {line}" for line in lines)
    text = f"{headline}\n{body}"

    if insufficient:
        text += (
            "\n\nNot evaluated for lack of reported facts "
            f"({insufficient} covenant(s) not reported)."
        )

    confidence = 0.95 if lines else 0.4
    return text, confidence


def _format_portfolio_answer(state: AgentState) -> tuple[str, float]:
    """Assemble portfolio aggregation results."""
    from app.rules.money import format_myr

    for r in state.tool_results:
        if r.tool_name == "run_read_only_sql" and r.ok and r.data:
            rows = r.data.get("rows", [])
            columns = r.data.get("columns", [])
            if not rows:
                return NO_EVIDENCE, 0.0

            out_lines = []
            for row in rows:
                parts = [str(row.get(c, "")) for c in columns]
                out_lines.append(" · ".join(parts))
            return "\n".join(f"- {line}" for line in out_lines), 0.95

        if r.tool_name == "get_portfolio_holdings" and r.ok and r.data:
            holdings = r.data.get("holdings", [])
            if not holdings:
                return NO_EVIDENCE, 0.0
            total = Decimal(0)
            out_lines = []
            for h in holdings:
                mv = h.get("market_value")
                if mv:
                    total += Decimal(str(mv))
                mv_str = format_myr(Decimal(str(mv))) if mv else "N/A"
                out_lines.append(
                    f"{h.get('portfolio_name', '')}: "
                    f"{h.get('instrument_name', '')} "
                    f"({h.get('issuer_name', '')}) — {mv_str}"
                )
            result = "\n".join(f"- {line}" for line in out_lines)
            return f"{result}\nTotal: {format_myr(total)}", 0.95

    return NO_EVIDENCE, 0.0


def _format_instrument_answer(state: AgentState) -> tuple[str, float]:
    """Assemble instrument lookup results, including rating-threshold filters."""
    import re

    from app.rules.money import format_myr
    from app.rules.ratings import UnknownRatingError, normalise, rank, try_rank

    for r in state.tool_results:
        if r.tool_name == "get_instrument" and r.ok and r.data:
            instruments_data = r.data.get("instruments", [])
            if not instruments_data:
                return NO_EVIDENCE, 0.0

            # Check if the question asks for instruments rated below a threshold
            threshold_match = re.search(
                r"\b(?:below|under|worse\s+than)\s+([Aa][A-Da-d]{0,2}[+-]?\d?)",
                state.question,
            )
            threshold_rating: str | None = None
            if threshold_match:
                try:
                    threshold_rating = normalise(threshold_match.group(1))
                except UnknownRatingError:
                    threshold_rating = None
            # Normalised above, so this cannot raise; computed once rather than
            # once per instrument.
            threshold_rank = rank(threshold_rating) if threshold_rating else None

            out_lines = []
            for inst in instruments_data:
                if threshold_rank is not None:
                    # An instrument the scale cannot place is **not** known to be
                    # below the threshold, and reporting it as such merges "could
                    # not evaluate" with a verdict somebody acts on -- in the
                    # direction that prompts action. Two holes used to do exactly
                    # that: an instrument with no rating never reached this filter,
                    # and an unrankable one was swallowed by `except
                    # UnknownRatingError: pass` and appended anyway.
                    #
                    # `try_rank` returns None for both cases. The deterministic
                    # read path excludes the same rows in SQL with
                    # `current_rating_rank IS NOT NULL` (app/query/service.py), and
                    # the two paths disagreeing is what findings 14 and 15 were.
                    instrument_rank = try_rank(inst.current_rating, inst.rating_agency)
                    if instrument_rank is None or instrument_rank <= threshold_rank:
                        continue

                parts = [
                    inst.instrument_name,
                    inst.issuer_name,
                ]
                if inst.current_rating:
                    parts.append(f"{inst.current_rating} ({inst.rating_agency.value})")
                if inst.issue_size is not None:
                    parts.append(format_myr(inst.issue_size))
                if inst.maturity_date is not None:
                    parts.append(f"matures {inst.maturity_date.isoformat()}")
                out_lines.append(" · ".join(parts))

            if threshold_rating:
                return (
                    f"{len(out_lines)} instrument(s) rated below "
                    f"{threshold_rating}.\n" + "\n".join(f"  - {line}" for line in out_lines),
                    0.95,
                )
            return (
                f"{len(out_lines)} instrument(s) matched.\n"
                + "\n".join(f"  - {line}" for line in out_lines),
                0.95,
            )

    return NO_EVIDENCE, 0.0


def _format_covenant_answer(state: AgentState) -> tuple[str, float]:
    """Assemble covenant lookup results, including call schedules."""
    out_lines: list[str] = []
    confidences: list[float] = []

    for r in state.tool_results:
        if r.tool_name == "get_covenants" and r.ok and r.data:
            rows = r.data.get("covenants", [])
            for covenant, clause in rows:
                parts = [covenant.covenant_type.value]
                if covenant.threshold_amount is not None:
                    from app.rules.money import format_myr

                    parts.append(f"threshold {format_myr(covenant.threshold_amount)}")
                thresholds = covenant.thresholds_json or {}
                ratio = thresholds.get("threshold_ratio")
                if ratio:
                    parts.append(f"ratio {ratio} times")
                trigger = thresholds.get("trigger_rating")
                if trigger:
                    parts.append(f"trigger rating {trigger}")
                parts.append(f"(page {clause.page_number})")
                out_lines.append(" · ".join(str(p) for p in parts))
                confidences.append(covenant.confidence or 0.5)

        elif r.tool_name == "get_call_schedules" and r.ok and r.data:
            calls = r.data.get("calls", [])
            if calls:
                call_lines = [
                    f"{c.call_date.isoformat()} at {c.call_price}% ({c.call_type.value})"
                    for c in calls
                ]
                out_lines.append("Call schedule: " + "; ".join(call_lines))

    if not out_lines:
        return NO_EVIDENCE, 0.0

    confidence = min(confidences) if confidences else 0.55
    return "\n".join(f"- {line}" for line in out_lines), confidence


def _format_search_answer(state: AgentState) -> tuple[str, float]:
    """Format document search results."""
    for r in state.tool_results:
        if r.tool_name == "search_clauses" and r.ok and r.data:
            hits = r.data.get("hits", [])
            count = r.data.get("count", 0)
            if not hits:
                return NO_EVIDENCE, 0.0

            best = hits[0]
            section = f", {best.chunk.section_title}" if best.chunk.section_title else ""
            text = (
                f"Found {count} supporting passage(s). "
                f"Best match (page {best.chunk.page_number}{section}):\n"
                f"{best.chunk.chunk_text[:600]}"
            )
            return text, 0.55

    return NO_EVIDENCE, 0.0


def _portfolio_sql(question: str) -> str | None:
    """Generate a safe SQL template for portfolio aggregation."""
    text = question.lower()
    if "exposure" in text or "fund" in text or "portfolio" in text:
        return (
            "SELECT p.name AS portfolio_name, "
            "COUNT(ph.id) AS holding_count, "
            "COALESCE(SUM(ph.market_value), 0) AS total_market_value "
            "FROM portfolios p "
            "JOIN portfolio_holdings ph ON ph.portfolio_id = p.id "
            "GROUP BY p.name "
            "ORDER BY total_market_value DESC "
            "LIMIT 100"
        )
    return None
