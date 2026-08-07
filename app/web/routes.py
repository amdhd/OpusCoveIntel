"""HTML routes.

These render through the same services the JSON API uses -- `CatalogService`,
`AgentQueryService`, the review repositories, the rules engine's own tool. The
UI holds no second opinion about what a covenant is, what a breach is, or who
may decide a review item.

Two things differ from the API on purpose:

* an unauthenticated request **redirects** to the login form rather than
  returning 401, because a person following a bookmark cannot act on JSON, and
* every mutation replies with a 303 to a GET, so a refresh after approving
  something does not approve it twice.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.agent.service import AgentQueryService
from app.agent.tools import evaluate_covenant_rule
from app.api.deps import get_auth_service
from app.auth.service import AuthService
from app.catalog.service import CatalogService
from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.db.repositories.ops import HumanReviewRepository
from app.db.session import get_readonly_session, get_session
from app.web.deps import PageUser, login_redirect, safe_next
from app.web.templates import templates

logger = get_logger(__name__)

router = APIRouter(prefix="/ui", tags=["ui"], include_in_schema=False)


def _catalog(session: Annotated[AsyncSession, Depends(get_readonly_session)]) -> CatalogService:
    return CatalogService(session)


CatalogDep = Annotated[CatalogService, Depends(_catalog)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


async def _pending_count(session: AsyncSession) -> int:
    """Badge for the nav. Cheap enough to run per page; a count, not a list."""
    return await HumanReviewRepository(session).count_pending()


def _page(
    request: Request,
    name: str,
    context: dict[str, Any],
    *,
    status_code: int = 200,
) -> HTMLResponse:
    return templates.TemplateResponse(request, name, context, status_code=status_code)


# -- login -------------------------------------------------------------------


@router.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, next: str = "") -> HTMLResponse:
    return _page(request, "login.html", {"user": None, "next_path": safe_next(next)})


@router.post("/login")
async def login_submit(
    request: Request,
    service: Annotated[AuthService, Depends(get_auth_service)],
    settings: SettingsDep,
    username: Annotated[str, Form()],
    password: Annotated[str, Form()],
    next: Annotated[str, Form()] = "",
) -> Response:
    destination = safe_next(next)
    user = await service.authenticate(username, password)
    if user is None:
        # 401 with the form re-rendered. The message does not distinguish an
        # unknown user from a wrong password (see app/auth/service.py).
        return _page(
            request,
            "login.html",
            {
                "user": None,
                "next_path": destination,
                "error": "Invalid username or password.",
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    issued = await service.start_session(
        user,
        ttl=settings.session_ttl,
        user_agent=request.headers.get("user-agent"),
        client_ip=request.client.host if request.client else None,
    )
    response = RedirectResponse(destination, status_code=status.HTTP_303_SEE_OTHER)
    response.set_cookie(
        settings.SESSION_COOKIE_NAME,
        issued.token,
        max_age=settings.SESSION_TTL_HOURS * 3600,
        httponly=True,
        samesite="lax",
        secure=settings.SESSION_COOKIE_SECURE,
        path="/",
    )
    return response


@router.post("/logout")
async def logout(
    request: Request,
    service: Annotated[AuthService, Depends(get_auth_service)],
    settings: SettingsDep,
) -> Response:
    token = request.cookies.get(settings.SESSION_COOKIE_NAME, "")
    if token:
        await service.end_session(token)
    response = RedirectResponse("/ui/login", status_code=status.HTTP_303_SEE_OTHER)
    response.delete_cookie(settings.SESSION_COOKIE_NAME, path="/")
    return response


# -- ask ---------------------------------------------------------------------


@router.get("/ask", response_class=HTMLResponse)
async def ask_form(
    request: Request,
    user: PageUser,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    return _page(
        request,
        "ask.html",
        {"user": user, "active": "ask", "pending_count": await _pending_count(session)},
    )


@router.post("/ask", response_class=HTMLResponse)
async def ask_submit(
    request: Request,
    user: PageUser,
    question: Annotated[str, Form()],
    read_session: Annotated[AsyncSession, Depends(get_readonly_session)],
    log_session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    """Answer inline rather than redirecting.

    A POST that renders its own result would normally be a re-submission
    hazard, but re-asking a question is harmless and free -- and a redirect
    would mean putting the question in a URL, where it would land in access
    logs and browser history. Questions about a portfolio are not something to
    scatter around.
    """
    agent = AgentQueryService(read_session, log_session=log_session)
    answer = await agent.answer(question, user_id=user.username)

    return _page(
        request,
        "ask.html",
        {
            "user": user,
            "active": "ask",
            "pending_count": await _pending_count(log_session),
            "question": question,
            "answer": answer,
        },
    )


# -- catalogue ---------------------------------------------------------------


@router.get("/instruments", response_class=HTMLResponse)
async def instruments(
    request: Request,
    user: PageUser,
    catalog: CatalogDep,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    return _page(
        request,
        "instruments.html",
        {
            "user": user,
            "active": "instruments",
            "pending_count": await _pending_count(session),
            "instruments": await catalog.list_instruments(limit=200),
        },
    )


@router.get("/instruments/{instrument_id}", response_class=HTMLResponse)
async def instrument_detail(
    request: Request,
    instrument_id: uuid.UUID,
    user: PageUser,
    catalog: CatalogDep,
    settings: SettingsDep,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    detail = await catalog.get_instrument(instrument_id)
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="instrument not found")

    return _page(
        request,
        "instrument.html",
        {
            "user": user,
            "active": "instruments",
            "pending_count": await _pending_count(session),
            "i": detail,
            "confidence_threshold": settings.DEFAULT_CONFIDENCE_THRESHOLD,
        },
    )


@router.get("/clauses/{clause_id}", response_class=HTMLResponse)
async def clause_provenance(
    request: Request,
    clause_id: uuid.UUID,
    user: PageUser,
    catalog: CatalogDep,
    settings: SettingsDep,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    """The provenance viewer — a citation you can check rather than trust."""
    provenance = await catalog.clause_provenance(clause_id)
    if provenance is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="clause not found")

    return _page(
        request,
        "provenance.html",
        {
            "user": user,
            "active": "instruments",
            "pending_count": await _pending_count(session),
            "p": provenance,
            "confidence_threshold": settings.DEFAULT_CONFIDENCE_THRESHOLD,
        },
    )


# -- portfolios --------------------------------------------------------------


@router.get("/portfolios", response_class=HTMLResponse)
async def portfolios(
    request: Request,
    user: PageUser,
    catalog: CatalogDep,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    return _page(
        request,
        "portfolios.html",
        {
            "user": user,
            "active": "portfolios",
            "pending_count": await _pending_count(session),
            "portfolios": await catalog.list_portfolios(),
        },
    )


@router.get("/portfolios/{portfolio_id}", response_class=HTMLResponse)
async def portfolio_detail(
    request: Request,
    portfolio_id: uuid.UUID,
    user: PageUser,
    catalog: CatalogDep,
    read_session: Annotated[AsyncSession, Depends(get_readonly_session)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    """The breach board.

    Evaluation runs through `evaluate_covenant_rule` -- the same deterministic
    tool the agent calls -- rather than a second implementation for the UI.
    Two rules engines would eventually disagree, and the one on screen is the
    one a person would act on.
    """
    holdings = await catalog.portfolio_holdings(portfolio_id)
    if holdings is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="portfolio not found")

    rows: list[dict[str, Any]] = []
    breach_total = at_risk_total = 0
    for holding in holdings.holdings:
        result = await evaluate_covenant_rule(read_session, instrument_id=holding.instrument.id)
        data = result.data if result.ok and result.data else {}
        breaches = int(data.get("breach_count", 0))
        at_risk = int(data.get("at_risk_count", 0))
        breach_total += breaches
        at_risk_total += at_risk
        rows.append(
            {
                "holding": holding,
                "evaluations": data.get("evaluations", []),
                "breaches": breaches,
                "at_risk": at_risk,
                "insufficient": int(data.get("insufficient_data_count", 0)),
            }
        )

    # Worst first: a breach board that buries the breach is a list.
    rows.sort(key=lambda row: (-row["breaches"], -row["at_risk"]))

    return _page(
        request,
        "portfolio.html",
        {
            "user": user,
            "active": "portfolios",
            "pending_count": await _pending_count(session),
            "h": holdings,
            "rows": rows,
            "breach_total": breach_total,
            "at_risk_total": at_risk_total,
        },
    )


# -- review queue ------------------------------------------------------------


@router.get("/review", response_class=HTMLResponse)
async def review_queue(
    request: Request,
    user: PageUser,
    session: Annotated[AsyncSession, Depends(get_session)],
) -> HTMLResponse:
    repo = HumanReviewRepository(session)
    items = await repo.list_pending(limit=100)
    return _page(
        request,
        "review.html",
        {
            "user": user,
            "active": "review",
            "pending_count": await repo.count_pending(),
            "total_pending": await repo.count_pending(),
            "items": [
                {
                    "id": str(item.id),
                    "entity_type": item.entity_type,
                    "field_name": item.field_name,
                    "old_value": item.old_value,
                    "source_quote": item.source_quote,
                    "page_number": item.page_number,
                    "confidence": item.confidence,
                    "trigger_reason": item.trigger_reason.value,
                }
                for item in items
            ],
            # Read-only for an analyst. The server enforces this too; hiding
            # the buttons is courtesy, not the control.
            "can_review": user.role.may_review,
        },
    )


@router.post("/review/{review_id}/{action}")
async def review_action(
    review_id: uuid.UUID,
    action: str,
    user: PageUser,
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    new_value: Annotated[str, Form()] = "",
    reason: Annotated[str, Form()] = "",
) -> Response:
    """Approve, correct or reject, then redirect.

    The role check is repeated here rather than trusted from the template: the
    form is hidden from analysts, and a hidden form is not a permission.
    """
    if not user.role.may_review:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail=f"role {user.role.value!r} may not decide review items",
        )

    from app.api.routes.review import (
        ApproveRequest,
        CorrectRequest,
        RejectRequest,
        approve_review,
        correct_review,
        reject_review,
    )

    try:
        if action == "approve":
            await approve_review(review_id, ApproveRequest(), user, session)
        elif action == "correct":
            await correct_review(review_id, CorrectRequest(new_value=new_value), user, session)
        elif action == "reject":
            await reject_review(review_id, RejectRequest(reason=reason), user, session)
        else:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="unknown action")
    except HTTPException as exc:
        if exc.status_code == status.HTTP_409_CONFLICT:
            # Someone else decided it first, or a double submit. Not an error
            # worth a stack trace -- reload and show the current queue.
            logger.info("ui.review_conflict", extra={"review_id": str(review_id)})
            return RedirectResponse("/ui/review", status_code=status.HTTP_303_SEE_OTHER)
        raise

    # 303 to a GET so a refresh does not resubmit the decision.
    return RedirectResponse("/ui/review", status_code=status.HTTP_303_SEE_OTHER)


__all__ = ["login_redirect", "router"]
