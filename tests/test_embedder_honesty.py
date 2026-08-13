"""Running on the placeholder embedder is a fact the deployment should know.

`HashingEmbedder` makes the vector leg of hybrid retrieval *run*, and makes it
meaningless: it has no semantics, so "gearing" and "leverage" are unrelated to
it. Without a Qwen key that is what answers every question, and the symptom is
not an error -- it is a page of legal advisers ranked above the negative-pledge
clause, which reads as an ordinary bad result (docs/review.md, finding 11).

Two things are pinned here, both about *knowing*:

* falling back says so, once, with the consequence spelled out; and
* a query whose vector leg matched nothing because the corpus was indexed by a
  different model says that, rather than silently degrading to keyword-only.

Neither test asserts retrieval quality. That needs a real key and a re-baseline
(PLAN.md Phase 10.4); what these defend is that nobody has to guess which
embedder they are running.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.repositories.documents import DocumentChunkRepository
from app.llm import embeddings as embeddings_module
from app.llm.embeddings import FAKE_EMBEDDING_MODEL, HashingEmbedder, get_embedder
from app.retrieval.hybrid import HybridSearcher

pytestmark = pytest.mark.usefixtures("storage_root")


def _fields(caplog: pytest.LogCaptureFixture, needle: str) -> dict[str, Any]:
    """The structured fields of the matching record.

    `caplog.text` renders the message only; the detail this module cares about
    travels as `extra`, which lands on the record as plain attributes.
    """
    record = next(r for r in caplog.records if needle in r.getMessage())
    return vars(record)


@pytest.fixture(autouse=True)
def _unannounce() -> None:
    """The warning is once-per-process; each test wants its own first time."""
    embeddings_module._placeholder_announced = False


class TestTheFallbackAnnouncesItself:
    def test_a_missing_key_is_reported_with_its_consequence(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # `test` short-circuits to the placeholder without a word, because CI
        # is where that is the correct and expected choice.
        monkeypatch.setenv("ENVIRONMENT", "local")
        monkeypatch.delenv("QWEN_API_KEY", raising=False)

        with caplog.at_level(logging.WARNING):
            embedder = get_embedder()

        assert isinstance(embedder, HashingEmbedder)
        assert "placeholder embedder" in caplog.text
        # The point is not that a fallback happened but what it costs, and the
        # detail travels as structured fields rather than in the message --
        # `caplog.text` renders only the message, so assert on the record.
        record = _fields(caplog, "placeholder")
        assert record["reason"] == "QWEN_API_KEY is not set"
        assert "no semantic signal" in record["consequence"]

    def test_it_is_said_once_rather_than_per_chunk(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Indexing calls this per document; ten thousand warnings is none."""
        monkeypatch.setenv("ENVIRONMENT", "local")
        monkeypatch.delenv("QWEN_API_KEY", raising=False)

        with caplog.at_level(logging.WARNING):
            for _ in range(5):
                get_embedder()

        assert caplog.text.count("placeholder embedder") == 1

    def test_the_test_environment_stays_silent(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setenv("ENVIRONMENT", "test")

        with caplog.at_level(logging.WARNING):
            get_embedder()

        assert "placeholder embedder" not in caplog.text


class TestAModelMismatchIsNotSilent:
    async def test_a_corpus_indexed_by_another_model_says_so(
        self,
        db_session: AsyncSession,
        indexed_corpus: list[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The trap waiting for the day a real key is added.

        `search_by_vector` filters to chunks embedded by the same model --
        correctly, since comparing two models' vectors is meaningless -- so a
        corpus embedded by the placeholder and queried with a real model
        returns nothing at all, and the answer silently becomes keyword-only.
        """

        class OtherModel(HashingEmbedder):
            @property
            def model_id(self) -> str:
                return "text-embedding-v4"

        searcher = HybridSearcher(db_session, embedder=OtherModel())

        with caplog.at_level(logging.WARNING):
            hits = await searcher.search_vector("gearing covenant", limit=5)

        assert hits == []
        assert "indexed by another model" in caplog.text
        record = _fields(caplog, "another model")
        assert record["querying_with"] == "text-embedding-v4"
        assert record["corpus_indexed_by"] == [FAKE_EMBEDDING_MODEL]
        assert "opuscovintel index" in record["remedy"]

    async def test_an_agreeing_corpus_says_nothing(
        self,
        db_session: AsyncSession,
        indexed_corpus: list[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The normal case must not warn, or the warning becomes noise."""
        with caplog.at_level(logging.WARNING):
            hits = await HybridSearcher(db_session).search_vector("gearing covenant", limit=5)

        assert hits
        assert "indexed by another model" not in caplog.text

    async def test_an_empty_corpus_is_not_a_mismatch(
        self, db_session: AsyncSession, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Nothing indexed is nothing to disagree with."""
        with caplog.at_level(logging.WARNING):
            hits = await HybridSearcher(db_session).search_vector("gearing covenant", limit=5)

        assert hits == []
        assert "indexed by another model" not in caplog.text

    async def test_the_index_reports_which_models_built_it(
        self, db_session: AsyncSession, indexed_corpus: list[str]
    ) -> None:
        models = await DocumentChunkRepository(db_session).embedding_models()

        assert models == {FAKE_EMBEDDING_MODEL}
