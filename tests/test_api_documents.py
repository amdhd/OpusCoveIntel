"""The document API.

Route handlers are meant to be thin (CLAUDE.md 3), so these tests assert the
HTTP contract -- status codes, shapes, error mapping -- and leave the pipeline
behaviour to `test_ingest_service.py`.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from tests.fixtures.synthetic_pdf import build_mixed_document, build_prospectus

# (filename, content, content-type) as httpx expects for a multipart part.
FileSpec = tuple[str, bytes, str]

pytestmark = pytest.mark.usefixtures("storage_root")


def pdf_upload(data: bytes | None = None, name: str = "im.pdf") -> dict[str, FileSpec]:
    return {"file": (name, data if data is not None else build_prospectus(), "application/pdf")}


async def upload(client: AsyncClient, data: bytes | None = None, name: str = "im.pdf") -> Response:
    return await client.post("/documents/upload", files=pdf_upload(data, name))


async def test_upload_creates_a_document(api_client: AsyncClient) -> None:
    response = await upload(api_client)

    assert response.status_code == 201
    body = response.json()
    assert body["duplicate"] is False
    assert body["document"]["status"] == "uploaded"
    assert len(body["document"]["sha256"]) == 64


async def test_uploading_the_same_bytes_again_returns_the_existing_document(
    api_client: AsyncClient,
) -> None:
    data = build_prospectus()
    first = await upload(api_client, data, name="original.pdf")

    second = await upload(api_client, data, name="copy.pdf")

    # 200, not 409: a duplicate is a correct outcome (CLAUDE.md 1.7).
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert second.json()["document"]["id"] == first.json()["document"]["id"]


async def test_a_non_pdf_upload_is_rejected(api_client: AsyncClient) -> None:
    response = await upload(api_client, b"PK\x03\x04 zip", name="archive.zip")

    assert response.status_code == 415


async def test_metadata_from_the_form_reaches_the_row(api_client: AsyncClient) -> None:
    response = await api_client.post(
        "/documents/upload",
        files=pdf_upload(),
        data={"document_type": "trust_deed", "uploaded_by": "analyst@example.com"},
    )

    document = response.json()["document"]
    assert document["document_type"] == "trust_deed"
    assert document["uploaded_by"] == "analyst@example.com"


async def test_an_invalid_document_type_is_rejected(api_client: AsyncClient) -> None:
    response = await api_client.post(
        "/documents/upload", files=pdf_upload(), data={"document_type": "napkin"}
    )

    assert response.status_code == 422


async def test_documents_can_be_listed_and_fetched(api_client: AsyncClient) -> None:
    created = (await upload(api_client)).json()["document"]

    listing = await api_client.get("/documents")
    fetched = await api_client.get(f"/documents/{created['id']}")

    assert listing.status_code == 200
    assert created["id"] in [item["id"] for item in listing.json()]
    assert fetched.status_code == 200
    assert fetched.json()["sha256"] == created["sha256"]


async def test_listing_can_be_filtered_by_status(api_client: AsyncClient) -> None:
    await upload(api_client)

    uploaded = await api_client.get("/documents", params={"status_filter": "uploaded"})
    extracted = await api_client.get("/documents", params={"status_filter": "extracted"})

    assert len(uploaded.json()) == 1
    assert extracted.json() == []


async def test_an_unknown_document_is_a_404(api_client: AsyncClient) -> None:
    missing = uuid.uuid4()

    assert (await api_client.get(f"/documents/{missing}")).status_code == 404
    assert (await api_client.get(f"/documents/{missing}/pages")).status_code == 404
    assert (await api_client.get(f"/documents/{missing}/chunks")).status_code == 404
    assert (await api_client.post(f"/documents/{missing}/process")).status_code == 404


async def test_processing_reports_pages_chunks_and_flagged_pages(
    api_client: AsyncClient,
) -> None:
    document_id = (await upload(api_client)).json()["document"]["id"]

    response = await api_client.post(f"/documents/{document_id}/process")

    assert response.status_code == 200
    body = response.json()
    assert body["page_count"] == 4
    assert body["chunk_count"] > 0
    assert body["pages_flagged_for_vlm"] == 0
    assert body["skipped"] is False


async def test_processing_twice_reports_the_second_run_as_skipped(
    api_client: AsyncClient,
) -> None:
    document_id = (await upload(api_client)).json()["document"]["id"]
    await api_client.post(f"/documents/{document_id}/process")

    response = await api_client.post(f"/documents/{document_id}/process")

    assert response.json()["skipped"] is True


async def test_page_telemetry_is_exposed_with_its_vlm_reason(
    api_client: AsyncClient,
) -> None:
    document_id = (await upload(api_client, build_mixed_document())).json()["document"]["id"]
    await api_client.post(f"/documents/{document_id}/process")

    pages = (await api_client.get(f"/documents/{document_id}/pages")).json()

    assert [page["page_number"] for page in pages] == [1, 2]
    assert pages[0]["vlm_reason"] is None
    assert "no_text_layer" in pages[1]["vlm_reason"]
    assert pages[1]["vlm_used"] is False


async def test_chunks_are_exposed_with_their_spans(api_client: AsyncClient) -> None:
    document_id = (await upload(api_client)).json()["document"]["id"]
    await api_client.post(f"/documents/{document_id}/process")

    chunks = (await api_client.get(f"/documents/{document_id}/chunks")).json()

    assert chunks
    for chunk in chunks:
        assert chunk["char_end"] - chunk["char_start"] == len(chunk["chunk_text"])
        assert chunk["page_number"] >= 1
        assert chunk["fts_config"] in {"english", "simple"}


async def test_the_openapi_document_advertises_the_upload_endpoint(
    api_client: AsyncClient,
) -> None:
    schema = (await api_client.get("/openapi.json")).json()

    assert "/documents/upload" in schema["paths"]


# -- progress ----------------------------------------------------------------
#
# What the upload screen polls. Everything here was already in the database and
# none of it was reachable over HTTP, which is why the UI had no upload screen
# worth building (docs/review.md, finding 7).


async def test_a_fresh_upload_is_queued_and_not_terminal(api_client: AsyncClient) -> None:
    """`uploaded` means the worker has not picked it up yet.

    A poller that treated this as finished would report an unparsed document as
    ingested, which is the one thing this endpoint must not allow.
    """
    document_id = (await upload(api_client)).json()["document"]["id"]

    status = (await api_client.get(f"/documents/{document_id}/status")).json()

    assert status["status"] == "uploaded"
    assert status["terminal"] is False
    assert status["chunk_count"] == 0
    assert status["error"] is None
    assert [job["job_type"] for job in status["jobs"]] == ["parse"]
    assert status["jobs"][0]["status"] == "queued"


async def test_a_processed_document_reports_terminal_with_its_counts(
    api_client: AsyncClient,
) -> None:
    document_id = (await upload(api_client)).json()["document"]["id"]
    await api_client.post(f"/documents/{document_id}/process")

    status = (await api_client.get(f"/documents/{document_id}/status")).json()

    assert status["terminal"] is True
    assert status["status"] == "chunked"
    assert status["page_count"] and status["page_count"] > 0
    assert status["chunk_count"] > 0
    assert status["jobs"][0]["status"] == "succeeded"
    assert status["jobs"][0]["finished_at"] is not None
    assert status["error"] is None


async def test_a_flagged_page_is_counted_for_the_screen(api_client: AsyncClient) -> None:
    """The VLM count is on this response so the operator sees the cost coming."""
    document_id = (await upload(api_client, build_mixed_document())).json()["document"]["id"]
    await api_client.post(f"/documents/{document_id}/process")

    status = (await api_client.get(f"/documents/{document_id}/status")).json()

    assert status["pages_flagged_for_vlm"] == 1


async def test_a_failure_reports_its_reason_rather_than_just_failing(
    api_client: AsyncClient, db_session: AsyncSession
) -> None:
    """A document stuck at "failed" with no reason sends someone to the logs.

    The service already writes the message down; this is the assertion that it
    reaches the screen.
    """
    from app.db.models.documents import Document
    from app.db.models.ops import ExtractionJob
    from app.domain.enums import DocumentStatus, JobStatus

    document_id = (await upload(api_client)).json()["document"]["id"]
    document = await db_session.get(Document, uuid.UUID(document_id))
    assert document is not None
    document.status = DocumentStatus.FAILED
    job = (
        await db_session.execute(
            select(ExtractionJob).where(ExtractionJob.document_id == uuid.UUID(document_id))
        )
    ).scalar_one()
    job.status = JobStatus.FAILED
    job.error_message = "page 3: no text layer and the VLM cap is 0"
    await db_session.flush()

    status = (await api_client.get(f"/documents/{document_id}/status")).json()

    assert status["terminal"] is True
    assert status["error"] == "page 3: no text layer and the VLM cap is 0"


async def test_status_is_404_for_a_document_that_does_not_exist(
    api_client: AsyncClient,
) -> None:
    assert (await api_client.get(f"/documents/{uuid.uuid4()}/status")).status_code == 404
