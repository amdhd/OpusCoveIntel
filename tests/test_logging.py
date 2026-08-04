"""Structured logging tests."""

from __future__ import annotations

import json
import logging

from app.core.logging import JsonFormatter, request_id_var


def _record(**kwargs: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="hello %s",
        args=("world",),
        exc_info=None,
    )
    for key, value in kwargs.items():
        setattr(record, key, value)
    return record


def test_output_is_single_line_json() -> None:
    line = JsonFormatter().format(_record())
    assert "\n" not in line

    payload = json.loads(line)
    assert payload["level"] == "INFO"
    assert payload["logger"] == "app.test"
    assert payload["message"] == "hello world"
    assert payload["ts"].endswith("+00:00")


def test_extra_fields_are_merged() -> None:
    payload = json.loads(JsonFormatter().format(_record(document_id="doc-1", cost_usd=0.42)))
    assert payload["document_id"] == "doc-1"
    assert payload["cost_usd"] == 0.42


def test_request_id_is_injected_from_context() -> None:
    token = request_id_var.set("req-xyz")
    try:
        payload = json.loads(JsonFormatter().format(_record()))
    finally:
        request_id_var.reset(token)
    assert payload["request_id"] == "req-xyz"


def test_request_id_is_absent_when_unset() -> None:
    assert "request_id" not in json.loads(JsonFormatter().format(_record()))


def test_non_serializable_values_do_not_raise() -> None:
    """A logging call must never take down the request that made it."""
    payload = json.loads(JsonFormatter().format(_record(obj=object())))
    assert "obj" in payload


def test_exception_info_is_captured() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record()
        record.exc_info = sys.exc_info()
        payload = json.loads(JsonFormatter().format(record))

    assert "ValueError: boom" in payload["exc_info"]
