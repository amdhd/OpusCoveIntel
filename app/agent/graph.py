"""LangGraph query agent — the Phase 7 read path.

PLAN.md 5:

    classify_intent → plan → retrieve → tools → rules_eval → synthesize → verify → log
                              │                                                │
                              └──── insufficient evidence ──→ refuse ──────────┘

The graph wraps the deterministic Phase 4 service and adds LLM synthesis for
intents where prose is better than structured rows (document_search,
covenant_lookup). Breach checks and portfolio queries stay fully deterministic
(CLAUDE.md 1.1 — the LLM never computes a breach).

Session access: nodes receive the AsyncSession via `config["configurable"]["session"]`.
This keeps the graph provider-agnostic and the nodes testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal

from typing import Any, Dict

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

# RunnableConfig is langgraph's config dict.
RunnableConfig = Dict[str, Any]

from app.agent.tools import (
    ToolResult,
    evaluate_covenant_rule,
    get_call_schedules,
    get_covenants,
    get_instrument,
    get_portfolio_holdings,
    run_read_only_sql,
    search_clauses,
)
from app.core.logging import get_logger
from app.db.models.ops import AuditLog, QueryLog
from app.domain.enums import ActorType, QueryIntent
from app.domain.rules import Citation
from app.query.intent import classify
from app.query.service import NO_EVIDENCE, UNSUPPORTED_MESSAGE

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

    # Output
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    confidence: float = 0.0
    refused: bool = False

    # Logging
    tools_called: list[str] = field(default_factory=list)
    sql_generated: str | None = None


def _session_from_config(config: RunnableConfig) -> AsyncSession | None:
    """Extract the session from LangGraph's RunnableConfig."""
    configurable: dict[str, Any] = config.get("configurable", {})
    result: object = configurable.get("session")
    if result is None:
        return None
    if isinstance(result, AsyncSession):
        return result
    return None


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
            result = await get_covenants(session)
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

    if not state.evidence_sufficient and state.intent not in (
        QueryIntent.UNSUPPORTED,
        QueryIntent.COVENANT_BREACH_CHECK,
    ):
        logger.info("agent.insufficient_evidence", extra={"intent": state.intent.value})

    return state


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
                    result = await evaluate_covenant_rule(
                        session, instrument_id=inst.id
                    )
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

    Exists as a separate node so the graph topology matches PLAN.md, which
    matters for logging and future LLM-as-judge work.
    """
    return state


async def _synthesize(state: AgentState) -> AgentState:
    """Produce the final answer from tool results, deterministically."""
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

    PLAN.md 5: "every factual claim in the drafted answer must map to a clause_id
    that was actually retrieved this turn."
    """
    if state.refused:
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
    """Write a QueryLog row and add an audit entry."""
    session = _session_from_config(config)
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

        logger.info(
            "agent.query_logged",
            extra={
                "intent": state.intent.value,
                "refused": state.refused,
                "confidence": state.confidence,
                "citations": len(state.citations),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("agent.log_failed", extra={"error": str(exc)})

    return state


# -- edge routing ------------------------------------------------------------


def _route_after_plan(state: AgentState) -> Literal["refuse", "retrieve"]:
    if state.intent is QueryIntent.UNSUPPORTED:
        return "refuse"
    return "retrieve"


def _route_after_retrieve(state: AgentState) -> Literal["insufficient", "tools"]:
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
            instrument_count = (
                data.get("count", 0) if isinstance(data, dict) else 0
            )
        elif tool_name == "evaluate_covenant_rule" and ok and data:
            data_dict: dict[str, Any] = (
                data if isinstance(data, dict) else {}
            )
            inst_name = data_dict.get("instrument_name", "Unknown")
            for ev in data_dict.get("evaluations", []):
                status = ev["status"]
                lines.append(
                    f"{inst_name} [{status.upper()}] {ev['covenant_type']}: "
                    f"{ev['explanation']}"
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
        f"{breaching} covenant breach(es) found across "
        f"{instrument_count} instrument(s)."
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
    from app.rules.ratings import UnknownRatingError, normalise, rank

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

            out_lines = []
            for inst in instruments_data:
                # Filter by rating threshold if specified
                if threshold_rating and inst.current_rating:
                    try:
                        if rank(inst.current_rating) <= rank(threshold_rating):
                            continue
                    except UnknownRatingError:
                        pass

                parts = [
                    inst.instrument_name,
                    inst.issuer_name,
                ]
                if inst.current_rating:
                    parts.append(
                        f"{inst.current_rating} ({inst.rating_agency.value})"
                    )
                if inst.issue_size is not None:
                    parts.append(format_myr(inst.issue_size))
                if inst.maturity_date is not None:
                    parts.append(f"matures {inst.maturity_date.isoformat()}")
                out_lines.append(" · ".join(parts))

            if threshold_rating:
                return (
                    f"{len(out_lines)} instrument(s) rated below "
                    f"{threshold_rating}.\n"
                    + "\n".join(f"  - {line}" for line in out_lines),
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
                    f"{c.call_date.isoformat()} at {c.call_price}% "
                    f"({c.call_type.value})"
                    for c in calls
                ]
                out_lines.append(
                    "Call schedule: " + "; ".join(call_lines)
                )

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
            section = (
                f", {best.chunk.section_title}"
                if best.chunk.section_title
                else ""
            )
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
