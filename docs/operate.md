# Operating OpusCovIntel — runbook

Day-to-day tasks and the things that go wrong, with the command that answers
each. Deployment is in [deploy.md](deploy.md).

**The one rule worth memorising:** two commands spend money, `extract` and
`ocr`. Both refuse to run without an explicit target, both take `--dry-run`,
and both print the ceilings and ask before dispatching. Everything else in this
document is free, including `make eval`.

---

## 1. Bringing a new document in

```bash
uv run opuscovintel ingest path/to/document.pdf     # $0 — parse, score, chunk
uv run opuscovintel index                           # $0 — embed + full-text index
uv run opuscovintel extract-rules                   # $0 — regex extractor
uv run opuscovintel extract <document-id> --dry-run # $0 — prices the LLM pass
uv run opuscovintel extract <document-id>           # SPENDS MONEY
```

Ingestion is idempotent by content hash: the same bytes under a different
filename are the same document, and re-uploading is reported as a duplicate
rather than as an error.

The whole chain except the last line costs nothing and already answers most
questions. Run it first and look at the result before deciding whether the
document is worth an LLM pass.

**Check what a document will cost before spending.** `--dry-run` prices the
candidate spans without dispatching anything:

```bash
make extract-llm-dry-run     # prices every document in the corpus
```

## 2. Watching spend

```bash
uv run opuscovintel cost-report
```

Reads `llm_calls`, which every provider call is written to by
`app/llm/router.py`. There is no other ledger.

What to look at, in order of what it tells you:

- **Budget remaining.** At zero the circuit breaker opens and every call is
  refused, including ones you want.
- **Cost per document.** Should be cents. PLAN.md 2 budgets ~$0.35/document
  after candidate narrowing; the live prospectus run came in at $0.069.
- **Provider prompt-cache read tokens.** If this is **zero across repeated
  extractions, that is a bug, not a tuning issue** — something in the prompt
  prefix is changing between calls (a timestamp, a UUID, an unsorted `json.dumps`)
  and every call is paying full price for a prefix that should bill at 0.1×.
- **Documents over the per-document ceiling.** Should be none; the guard aborts
  before it happens. One appearing means the guard was bypassed or the ceiling
  was lowered after the fact.

## 3. Working the review queue

Items land there by policy, not by accident (CLAUDE.md 5): confidence below
`DEFAULT_CONFIDENCE_THRESHOLD`, rules and LLM disagreeing, a validation retry, a
citation that would not verify, a value read off a VLM page, or any monetary
threshold above RM100m.

Reviewers normally work the queue in the UI at `/ui/review`, which shows the
flagged value, its quote and its trigger. Over HTTP (both need a session cookie
— see `POST /auth/login`):

```bash
curl -s -b cookies.txt localhost:8000/review/pending \
  | jq '{total: .total_pending, by_reason: (.items | group_by(.trigger_reason)
        | map({(.[0].trigger_reason): length}) | add)}'
```

Approve, correct or reject through `POST /review/{id}/approve`, `/correct` or
`/reject`. **The reviewer is taken from the session, not from the request body** —
it used to be a client-supplied `reviewer_id`, which meant the audit trail
recorded whatever the caller typed. Only the `reviewer` role may decide an item;
an `analyst` reading the same queue gets a 403. The resolved-review CHECK
constraint is the backstop: it refuses a decision that names nobody.

Triage by `trigger_reason`, because the reasons need different work:

| Trigger | What it means | What to do |
|---|---|---|
| `rule_llm_disagreement` | The two extractors read the clause differently | Open the quote. One of them is wrong and the other is usually right |
| `citation_unverified` | The quote is not in the chunk it cites | **Nothing was persisted.** The clause was dropped, not saved wrong |
| `high_value_threshold` | Over RM100m | Always a human. A misread here is a portfolio-level error |
| `low_confidence` | Below the threshold | Usually a genuine ambiguity in the document |
| `validation_retry` | The model's output failed the schema twice | Look at the candidate span; often a chunking problem |

A correction keeps the previous value, the reviewer and the reason. That history
is the audit trail, and re-running extraction does not touch anything a human
has approved, corrected or rejected — machine output is disposable, human
judgement is not.

## 4. Re-extracting after a change

Extraction identity is `(document_sha256, prompt_version, model_id,
extractor_version)`. Re-running an unchanged pipeline is a no-op and costs $0 —
that is the design, and it is also the trap:

**If you change what an extractor produces, bump its version in the same
commit.** Otherwise every already-processed document is skipped as "identity
already satisfied" and keeps its old rows for ever, while your change works
perfectly on documents nobody has seen yet.

- rule extractor → `EXTRACTOR_VERSION` in `app/extract/rule_extractor.py`
- LLM pipeline → `LLM_EXTRACTOR_VERSION` in `app/extract/pipeline.py`
- prompts → `PROMPT_VERSION` in `app/extract/prompts/`

To force one document without a version bump: `extract <document-id> --force`.
It discards the previous machine output first, so the re-run replaces the old
clauses rather than doubling them.

## 5. Measuring whether it still works

```bash
make eval             # scores extraction + answers -> evals/results/
```

$0, no model calls, and it writes both a JSON record and a Markdown summary,
plus `latest.json` / `latest.md`. It exits non-zero when a read path misses its
PLAN.md target (9/13 deterministic, 11/13 agent). Extraction F1 is reported and
never gated — PLAN.md sets no target for it, and a threshold invented against a
synthetic corpus would be a gate that says nothing about production.

Read it in this order:

1. **Golden questions.** Below target is a regression in retrieval, intent
   classification or the rules engine.
2. **`unsupported_answers`.** Should be zero. Anything else is an answer with no
   citation and no structured tool behind it — the failure CLAUDE.md 1.5 exists
   to prevent.
3. **Extraction, per method.** `rule` and `llm` are scored separately on
   purpose. Their difference is the answer to "did the LLM actually help?", and
   pooling them destroys it.
4. **Citations.** `still verify against their chunk` should be 100 per cent. Less
   means a stored quote no longer occurs in the text it names — chunking changed
   under existing clauses, and those citations are now unopenable.
5. **Cost.** As §2.

**These numbers are from synthetic documents.** Every report says so. No
licensed prospectus is in the corpus (CLAUDE.md 7), so this is a regression
baseline, not a production accuracy estimate, and it needs re-baselining the day
real documents arrive.

To measure the LLM path you have to have run it — the agreement and LLM-method
sections report "no data" rather than a flattering default when nobody has spent
anything.

## 6. When something is wrong

### A document is stuck

```sql
SELECT document_id, job_type, status, error_message, started_at
FROM extraction_jobs WHERE status IN ('running', 'failed') ORDER BY created_at DESC;
```

A job stuck in `running` means a process died mid-work. Both extraction services
mark the job `failed` on an exception and roll back, so a durable `running` is a
killed process, not a caught error. Re-run the command; the job is re-claimed.

### `status = 'budget_exceeded'`

Not a failure. The document hit a ceiling part-way through and stopped, and it
is marked so downstream readers do not treat it as fully processed. **The
deterministic extractor's clauses are already persisted** — that happens before
any billable call, precisely so a budget-exhausted document ends up with
something rather than with nothing.

Raise `MAX_COST_PER_DOCUMENT_USD` if the document is genuinely worth it, then
re-run with `--force`. If the *global* ceiling opened the breaker, every call is
refused until `MAX_TOTAL_COST_USD` is raised.

### The provider is out of credit

`ocr` exits 3 and says so rather than retrying. Backoff cannot fix a billing
state, and every remaining page would fail identically.

### An answer looks wrong

```bash
uv run opuscovintel ask "<the question>" --show-tools
```

Then read the citations it printed. Every claim traces to a clause, a page and a
verbatim quote; if it does not, that is the bug, and the `verify` node in the
agent graph is what should have caught it. The full record is in `query_logs` —
question, intent, retrieved chunk ids, tools called, generated SQL, answer and
citations.

### "No supporting evidence in the corpus"

Usually correct and not a bug (CLAUDE.md 1.5). Check the document was indexed:
retrieval reads embeddings and the FTS column, which `ingest` does not write —
`index` does. An ingested but unindexed document is invisible to every question.

## 7. Routine checks

| When | Command | Looking for |
|---|---|---|
| Every deploy | `opuscovintel check-schema` | Enum drift Alembic cannot see |
| Every deploy | `make eval` | Extraction or answer regression |
| After any LLM run | `opuscovintel cost-report` | Spend, and zero cache reads |
| Weekly | `/review/pending` count | A queue nobody is working is a queue that is not a control |
