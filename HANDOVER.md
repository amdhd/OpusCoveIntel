# Handover — starting Phase 8 (Evaluation)

Written 2026-08-06, at commit `0606d84`. That commit is on branch
`feat/persist-rule-extractions` (PR #15), **not yet on `main`** — PR #15 is green
and mergeable as of writing but was left open intentionally, for the user to
merge. `main` itself is at `f4fad7d` (merged PR #14). Delete this file when
Phase 8 lands.

Read [CLAUDE.md](CLAUDE.md) first — it holds the architectural invariants, and its
model-routing table now has a `Status` column that tells you what's actually wired
versus designed. Read [PLAN.md §6](PLAN.md) for the Phase 8 scope this document
elaborates. This file covers only what a fresh session cannot infer from the code.

---

## 1. Where the project is

Phases 1–7 are built. Not just written — **run against live providers**, which
matters because several defects only surfaced that way (see §4). PRs #10–#15 (all
merged except #15) did the wiring and the review; their descriptions are the
detailed record if you need it. `gh pr view <n>` on any of them.

| | State |
|---|---|
| Tests | **625 passing** locally; lint and `mypy --strict` clean on 137 files |
| CI | green on `main`; `REQUIRE_POSTGRES=1` set, zero paid API calls anywhere |
| Extraction | **live-verified**: 7 calls, $0.069, prompt caching confirmed (`cache_read_input_tokens=5027` every call) |
| Agent | wired (`opuscovintel ask`), deterministic, two-session split confirmed live |
| VLM | wired (`opuscovintel ocr`) but **not live-verified** — OpenAI account has no credit |
| Eval harness | **does not exist**. `app/evals/` holds only `golden.py` (10 questions, no metrics runner) |

**Two commands now spend real money**, both refuse to run without an explicit
target and both have `--dry-run`:

```bash
uv run opuscovintel extract <document-id>   # covenant extraction, ~$0.01-0.02/candidate
uv run opuscovintel ocr <document-id>       # page OCR, ~$0.02/page (blocked: no OpenAI credit)
```

Nothing else in the CLI spends anything. `make check` is $0 by construction — see
`app/cli.py`'s module docstring, which states this as an invariant of the file.

## 2. Resuming (about two minutes)

```bash
cd /Users/hadi/OpusCovIntel
open -a Docker
make up
make migrate
make seed
make check                # -> expect 625 passed
```

If `make check` is green you're in a known-good state. To confirm the live paths
still work (optional, costs ~$0.10):

```bash
make extract-llm-dry-run  # free — prices what extract would send
uv run opuscovintel ask "What is the cross-default threshold?"   # free — agent has no LLM calls
```

**Check PR #15 before starting new work.** It's the rule-extraction-persistence
fix, opened at the end of the last session. If it's still open, merge it first —
Phase 8's cost/doc and rules-vs-LLM-agreement metrics both depend on rule
extractions actually being rows in the database, which is what #15 fixes.

## 3. Phase 8 scope

From PLAN.md §6. The metrics it names, in the order I'd build them (cheapest and
most load-bearing first):

1. **Field-level F1** — for each extracted field (`covenant_type`, `threshold_amount`,
   `operator`, ...), precision/recall against a golden-labeled set. This is the
   metric that lets you answer "is Sonnet good enough for extraction?" — the
   question flagged as open at the end of the last session. Nothing else in the
   harness matters more.
2. **Enum exact match** — `covenant_type`, `clause_type`, `rating_agency` etc.
   should match exactly; there's no partial credit for an enum.
3. **Numeric tolerance** — for `threshold_amount`/`threshold_ratio`, exact match
   is too strict (rounding) and no tolerance is too loose. Decimal, per CLAUDE.md
   §6 — never float, including inside the eval harness itself.
4. **Date tolerance** — same idea for `call_date`, `maturity_date`.
5. **Citation precision/recall** — reuses `app.extract.citations.verify_quote`
   directly rather than reimplementing verification; the harness's citation
   metric and the pipeline's citation gate must be the same code, or the eval
   measures something the pipeline doesn't actually enforce.
6. **Answer faithfulness** — every claim in an answer must trace to a citation
   that was actually retrieved. `app/agent/graph.py`'s `_verify` node already does
   this at runtime; the eval-time version is likely "run `_verify`'s logic
   against golden answers" rather than a new implementation. Look there first.
7. **Refusal correctness** — `golden.py` already has this encoded (`expect_refusal`,
   one deliberately unanswerable question, Q10). The harness needs to score
   against it; the fixture doesn't need building.
8. **Rules-vs-LLM agreement** — `app/extract/pipeline.py::_fields_disagree`
   already computes this per-extraction and increments `outcome.disagreements`.
   The eval-time version is aggregating that across a run, not reimplementing
   the comparison.
9. **Cost/doc** — `app/db/repositories/ops.py::LLMCallRepository.cost_by_stage`
   already exists and groups by `LLMStage`. `make cost-report` (named in
   CLAUDE.md §8) has no target yet — this is probably a thin CLI command over
   that repository method, not new aggregation logic.

**Acceptance** (PLAN.md): `make eval` emits metrics to `evals/results/` · CI green
· deploy/operate docs exist.

**Only after the harness works**, PLAN.md §6 says to build the deferred infra:
Celery/Redis, S3/MinIO, RBAC/OIDC, OTel/Prometheus. Don't start there — CLAUDE.md
§9 is explicit that these are deferred, and nothing in Phases 1–7 needs them yet.

## 4. What the last session learned that Phase 8 should not re-learn

**Read code, then run it — reading alone missed real bugs, every time.** Across
PRs #10–#15, four separate defects were found only by executing the path, never
by review: an `asyncio` event-loop bug in the `extract` CLI, a wrong storage key
in `VlmService`, two schema faults that made every live Anthropic call 400 until
fixed, and a config bug where `QWEN_API_KEY=` (blank) was treated as "key present"
instead of "key absent." None of these were visible in code review or unit tests
with mocks. **The eval harness is exactly the kind of thing that needs a real run
against real output before you trust it** — write it, then run `make eval` against
whatever's in the corpus and read the actual numbers, don't just get it to import
cleanly.

**A "fix" test can pass against the unfixed code.** In PR #15, a formatter
collapsed a multi-line statement onto one line between when a regression test was
written and when it was verified; the "disable the fix and confirm the test
fails" check silently ran against the *fixed* code because the string-replace
patch no longer matched. One of three new tests also asserted `x >= 0`, which is
true of everything and caught nothing. Both were only caught by actually
re-reading the diff of the "disabled" state before trusting the red. If Phase 8
adds regression protection for these bugs (it should — an eval harness is a
different kind of regression test than pytest), apply the same discipline:
confirm the check fails before you trust that it passes.

**Model swap decision is blocked on this harness, not on opinion.** The last
session's cost research found `claude-opus-5` → a cheaper Claude model is a
one-line `EXTRACTION_MODEL` env change with zero engineering cost (same caching,
same structured-output path), but explicitly declined to recommend it without a
way to measure the F1 delta. That's item 1 above. Once it exists, that's a real
decision to bring back to the user, not a guess.

**VLM OCR provider is worth revisiting, separately from Phase 8.** Research
surfaced `mistral-ocr-latest` at ~$0.004/page (vs. ~$0.02 for gpt-4o), with
native PDF input, structured table output, and real per-page/per-word confidence
scores — the last of which we currently fake (`VLM_PAGE_CONFIDENCE = 0.85`,
hardcoded, in `app/llm/vlm.py`). This is unrelated to Phase 8's scope but was
flagged as a candidate for a follow-up adapter behind the existing provider seam
in `app/llm/adapters/`. Not blocking — OpenAI has no credit anyway, so there's no
live gpt-4o baseline to compare against yet yourself.

## 5. Decisions already made that constrain Phase 8

- **The mock provider must stay realistic.** `app/llm/mock.py::MockLLMProvider`
  quotes its own input verbatim and fills gated enum fields (`_ground_in_input`,
  `_FORCED_ENUM_FIELDS`) specifically so it exercises the *success* path, not just
  the review-queue path. If eval fixtures need mock LLM output, extend that
  pattern rather than hand-rolling fixture JSON that happens to validate.
- **Money is `Decimal`, never `float`, including in eval code** — CLAUDE.md §6,
  and it's not a suggestion; the ratings/money modules are held to `mypy --strict`.
  A numeric-tolerance metric written in `float` will drift silently on exactly
  the values it's supposed to be checking.
- **No new provider calls in `make eval`.** CLAUDE.md §7: CI must never hit a paid
  API. If "answer faithfulness" or any other metric is tempted to use an LLM
  judge (PLAN.md's routing table reserves `claude-opus-5` for this, marked "not
  built" in the Status column), that call needs the same `RUN_LIVE_LLM_TESTS=1`
  gate that `live_llm`-marked tests already use — see `tests/conftest.py`'s
  `pytest_collection_modifyitems`.
- **The golden set (`app/evals/golden.py`) already carries a Phase-4-vs-Phase-7
  target split** (`PHASE_4_TARGET`, `PHASE_7_TARGET`). Whatever the eval harness's
  overall pass threshold is, it should be able to report per-path scores the same
  way — that's what makes "did the agent's verify node actually help?" answerable
  from the harness rather than assumed.
- **Repositories never commit; services own the transaction.** Unchanged from
  Phase 2 onward. An eval harness that reads extraction output should go through
  the existing repositories (`ClauseRepository`, `CovenantRepository`), not raw
  SQL — consistency with the rest of the codebase, and it means the harness
  automatically respects the read-only role split if it's ever run against
  production data.

## 6. Open questions carried forward, unresolved

From PLAN.md §9, still open: real prospectus availability (biggest schedule risk
— every regex and chunking heuristic is tuned on synthetic fixtures only),
Qwen embedding region/dimensionality, portfolio holdings source, reviewer
identity for the audit trail, Bahasa Malaysia corpus share. None of these block
Phase 8, but the eval harness's synthetic-only golden set means its numbers will
need re-baselining the day real documents arrive — worth saying explicitly in
whatever `evals/results/` output format gets built, so a future reader doesn't
mistake a synthetic-corpus F1 for a production one.
