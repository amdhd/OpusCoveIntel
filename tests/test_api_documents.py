"""The document API.

Route handlers are meant to be thin (CLAUDE.md 3), so these tests assert the
HTTP contract -- status codes, shapes, error mapping -- and leave the pipeline
behaviour to `test_ingest_service.py`.
"""

from __future__ import annotations

import uuid

import pytest
from httpx import AsyncClient, Response

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
