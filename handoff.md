# PAC RAG: Project Handoff

This document is the entry point for any agent or developer picking up the PAC RAG project. Read it end to end before touching code.

## Mission

Build a retrieval-augmented question-answering system over the Indiana Public Access Counselor's published advisory opinions. A user describes a public-records or open-meetings scenario and gets back a curated analysis grounded in past PAC opinions, with citations.

The Indiana PAC is a state office that issues non-binding advisory opinions interpreting Indiana's Access to Public Records Act (APRA, IC 5-14-3) and Open Door Law (ODL, IC 5-14-1.5). Opinions are persuasive, not precedential. The system must reflect that posture in its outputs.

Source of truth for opinions: https://www.in.gov/pac/advisory/

## Current state (do not redo)

The repo, environment, and an end-to-end smoke test are already in place. Everything below is verified working on macOS / Apple Silicon as of the handoff.

What's built:

- Workspace at `~/code/pac-rag` (path may differ on the target machine; treat the workspace root as `$REPO`).
- `uv` workspace with four Python member packages: `pac_crawler`, `pac_pipeline`, `pac_api`, `pac_eval`.
- A virtual root `pyproject.toml` (`package = false`) holding workspace config, ruff, pyright, pytest config, and a `dev` dependency group.
- `apps/web/` Next.js 15 app with shadcn/ui v4 already initialized (Radix base, TypeScript, App Router, src dir).
- Embedded Qdrant under `data/qdrant/`, no service required.
- Embedding (`BAAI/bge-large-en-v1.5`) and reranker (`BAAI/bge-reranker-v2-m3`) downloaded and cached under `data/models/` via `scripts/warm_models.py`. MPS is detected and used.
- Anthropic SDK working; `ANTHROPIC_API_KEY` lives in `.env` and is read through `pac_api.config.settings`.
- A working `scripts/smoke_test.py` that takes one PDF (`25-FC-118`), runs LiteParse, naive-chunks the text, embeds, indexes into Qdrant, retrieves, and generates an answer with Claude. The smoke test has been run successfully and produces a correctly-grounded answer about the Bloomington Transportation Commission case.
- `justfile` with stub targets (`sync`, `lint`, `format`, `test`, `crawl`, `parse`, `extract`, `embed`, `api`, `web`).
- Pre-commit hooks installed (ruff, ruff-format, basic hygiene checks).
- `.env.example` is checked in; `.env` is gitignored.

Do not re-scaffold any of the above. Read `pyproject.toml`, the package `pyproject.toml` files, `justfile`, and `scripts/smoke_test.py` before writing new code.

## Architecture

A query flows through five stages. Each stage is independently testable.

```
1. Filter      DuckDB metadata catalog narrows candidates.
2. Retrieve    Qdrant (dense vectors) + Tantivy (BM25) run in parallel.
3. Fuse        Reciprocal Rank Fusion combines the two ranked lists.
4. Rerank      bge-reranker-v2-m3 cross-encodes (query, chunk) pairs.
5. Generate    Claude Sonnet writes the answer with required citations.
```

Indexing is a separate offline pipeline:

```
Crawl listing pages -> download PDFs -> parse with LiteParse ->
  LLM-extract structured metadata -> chunk by section ->
    embed chunks -> upsert into Qdrant + Tantivy + DuckDB
```

Both pipelines are idempotent and resumable via the manifest (DuckDB).

## Tech stack reference

| Concern | Choice | Notes |
|---|---|---|
| Python runtime | 3.12 via uv | Workspace lockfile; one venv |
| Package manager | uv (workspace mode) | `uv add --package <name> <dep>` |
| Web framework | FastAPI | `pac_api` package |
| Vector DB | Qdrant in embedded mode | `QdrantClient(path=...)`; no service |
| Lexical index | tantivy-py | Embedded; `data/tantivy/` |
| Catalog DB | DuckDB | Single-writer; `data/catalog/pac.duckdb` |
| PDF parsing | LiteParse CLI (`lit`) | Already installed; shell out from Python |
| Embeddings | `BAAI/bge-large-en-v1.5`, dim 1024 | sentence-transformers on MPS |
| Reranker | `BAAI/bge-reranker-v2-m3` | CrossEncoder on MPS |
| Extraction LLM | `claude-haiku-4-5-20251001` | Cheap, batchable, structured output |
| Generation LLM | `claude-sonnet-4-6` | Citation-disciplined answers |
| Frontend | Next.js 15 + shadcn/ui v4 | App Router, Tailwind, Radix base |
| Lint / format | ruff | `ruff check`, `ruff format` |
| Type check | pyright | strict on `packages/` |
| Tests | pytest, pytest-recording | VCR cassettes for HTTP and Anthropic |
| Task runner | just | Targets in `justfile` at root |

Do not introduce LangChain, LlamaIndex, Celery, Redis, Postgres, or Docker. The architecture intentionally avoids them. If a concrete need arises later (e.g., concurrent indexer + serving), the upgrade path is documented at the bottom of this file.

## Repo layout

```
pac-rag/
├── pyproject.toml              # workspace root (virtual, package = false)
├── uv.lock
├── justfile
├── .env / .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── packages/
│   ├── pac_crawler/            # listing scrape + PDF download + manifest
│   ├── pac_pipeline/           # parse, extract, chunk, embed, index
│   ├── pac_api/                # FastAPI service, shared config
│   └── pac_eval/               # gold set + scoring
├── apps/
│   └── web/                    # Next.js 15 frontend
├── ops/                        # deployment, GH Actions, Cloudflare Tunnel
├── scripts/
│   ├── warm_models.py          # cache embedding + reranker
│   ├── check_anthropic.py      # API key smoke test
│   └── smoke_test.py           # end-to-end on one PDF
├── tests/                      # cross-package integration tests
├── notebooks/                  # exploratory; never imported
└── data/                       # gitignored
    ├── pdfs/<year>/<opinion_id>.pdf
    ├── parsed/<year>/<opinion_id>.json
    ├── catalog/pac.duckdb
    ├── qdrant/                 # embedded Qdrant storage
    ├── tantivy/                # BM25 index
    └── models/                 # HF cache, set HF_HOME here
```

`pac_api` owns the shared `config.py` with `Settings`. Every other package imports `from pac_api.config import settings`. Do not create a fifth "common" package; the dependency direction is fine as is.

## Conventions

These are project rules, not stylistic suggestions.

1. **Idempotence everywhere.** Every script and pipeline stage must be safe to rerun. Use the DuckDB manifest as the single source of truth for "has this been done."
2. **Opinion ID is the canonical key.** Filenames on the source site are inconsistent and sometimes misleading (see Known data quirks). Always derive the opinion ID from the listing page text, never from the PDF filename.
3. **No hallucinated metadata.** If a complainant is redacted, store `null`. Do not let the LLM extractor invent it.
4. **Validate LLM extractions with Pydantic.** Reject invalid responses and retry with a clarifying note. Do not silently coerce.
5. **Citation discipline at generation time.** The system prompt for the answer LLM must require the form `(opinion_id, p. N)` for every claim, and must instruct the model to say so explicitly when retrieved opinions don't address the question.
6. **Always include the advisory disclaimer.** PAC opinions are non-binding. Generated answers must surface this caveat. The user-facing UI should also display it.
7. **Type everything that crosses a package boundary.** pyright is configured strict on `packages/`. Public APIs use Pydantic models, not loose dicts.
8. **Logging is structlog, JSON.** No `print()` in library code. Scripts under `scripts/` may print.
9. **Tests use pytest-recording for HTTP.** Crawler and Anthropic calls record cassettes the first time and replay thereafter. Do not write tests that hit the live network in CI.
10. **Voice and tone in user-visible text.** Direct, warm, concrete, occasionally dry. Avoid em-dashes with surrounding spaces. Avoid hedging language stacks ("it's important to note that potentially..."). One disclaimer about advisory status is enough; don't sprinkle the same caveat through every paragraph.

## Remaining work

The work is broken into nine milestones. Each milestone has acceptance criteria. Milestones 1–5 are sequential; 6 can start once 4 is done; 7 once 6 has at least a `/ask` endpoint; 8 and 9 are parallelizable at the end.

---

### Milestone 1: Crawler and manifest

**Package:** `pac_crawler`

**Goal:** Produce a complete, accurate catalog of every published PAC advisory opinion, with PDFs on disk under `data/pdfs/<year>/<opinion_id>.pdf` and metadata in `data/catalog/pac.duckdb`.

**Subtasks:**

1. **Year-page enumeration.** Two index pages anchor the crawl:
   - Formal complaints: `https://www.in.gov/pac/advisory/` lists year-pages from 1999 onward at `/pac/advisory/<year>-advisory-opinions` (e.g. `/pac/advisory/2025-advisory-opinions`).
   - Informal opinions: a separate listing at `https://www.in.gov/pac/advisory/informal-inquiries/` with similar per-year structure. Verify the URL pattern at crawl time; the site has been reorganized at least once.
2. **Listing parser.** For each year-page, parse out one row per opinion. The row provides: title text, opinion ID (regex: `(\d{2})-(FC|INF)-(\d+)(A|B)?(\s*REVISED)?`), source agency name, and the link to the PDF. Use `selectolax` (already a dep) for speed; the site is plain HTML.
3. **Manifest.** A DuckDB table `opinions` with columns:
   ```
   opinion_id        TEXT PRIMARY KEY    e.g. "25-FC-118"
   year              INTEGER
   kind              TEXT                'FC' or 'INF'
   sub_letter        TEXT                'A' | 'B' | NULL
   revision          INTEGER             0 for original, 1+ for revised
   parent_opinion_id TEXT                links revisions to originals
   raw_title         TEXT                from the listing page
   agency_name       TEXT                parsed from title
   listing_url       TEXT                year-page URL
   pdf_url           TEXT                exact URL from listing page
   pdf_path          TEXT                local path after download
   pdf_sha256        TEXT                content hash
   first_seen_at     TIMESTAMP
   last_checked_at   TIMESTAMP
   download_status   TEXT                'pending' | 'ok' | 'http_error' | 'missing'
   ```
4. **Downloader.** Polite httpx client with `tenacity` retries, rate-limited to roughly 1–2 req/sec. Save PDFs to `data/pdfs/<year>/<opinion_id>.pdf` (canonical, not source filename). Skip if hash matches the recorded one. Update `last_checked_at` on every run.
5. **Diffing.** A second invocation re-fetches the listing pages and adds new rows, marks now-missing opinions as `'missing'` (don't delete; preserve history).
6. **CLI entry point.** `pac-crawl` registered via `[project.scripts]` in `pac_crawler/pyproject.toml`. Accept `--year YYYY`, `--kind FC|INF|all`, `--listing-only` (skip downloads), `--limit N`.

**Acceptance criteria:**

- `just crawl` with no flags pulls every listing-page entry into the manifest and downloads every PDF.
- Re-running is idempotent: no duplicate rows, no re-downloads of unchanged PDFs.
- The manifest correctly handles the 25-FC-138 case: the opinion ID is `25-FC-138` even though the actual file lives at `26-FC-138-LaFontaine.pdf` (see Known data quirks).
- Total opinion count is roughly 1,500–2,500 across both kinds. Sanity-check by spot-comparing five random year-pages against the manifest.
- Tests use cassettes; CI never hits in.gov.

---

### Milestone 2: Parse pipeline

**Package:** `pac_pipeline`

**Goal:** Convert every PDF in the manifest into a normalized text JSON sidecar at `data/parsed/<year>/<opinion_id>.json`. Use LiteParse.

**Subtasks:**

1. **LiteParse wrapper.** A Python module that shells out to `lit parse <pdf> --format json -o <out>`. Capture stderr, parse JSON, return a typed Pydantic model. Look at the actual JSON shape produced by LiteParse on `25-FC-118` (already cached at `data/parsed/2025/25-FC-118.json`) before writing the wrapper. Common shape: `{"pages": [{"page_num": 1, "text": "..."}]}` or similar. Don't trust documentation; trust the file.
2. **OCR routing.** For each page, count extracted characters. If a page is below ~50 chars and contains no obvious markers ("BACKGROUND", "ANALYSIS", a date), re-run LiteParse with OCR enabled (`lit parse --ocr ...`) for that document. Pre-2008 opinions are most likely to need this.
3. **Section detection.** PAC opinions follow a near-template: caption block (with "ISSUED:", "Office of the Indiana Public Access Counselor", an opinion ID), then "BACKGROUND", "ANALYSIS", "CONCLUSION", then signature. Detect these section boundaries by regex against the full text. Store start/end character offsets as part of the parsed JSON.
4. **Output schema.**
   ```python
   class ParsedPage(BaseModel):
       page_num: int
       text: str
       char_count: int
       ocr_used: bool

   class ParsedOpinion(BaseModel):
       opinion_id: str
       full_text: str
       pages: list[ParsedPage]
       sections: dict[str, tuple[int, int]]   # name -> (start, end) char offsets
       parser_version: str                     # for reprocessing later
   ```
5. **Manifest update.** Add `parse_status`, `parse_path`, `parsed_at`, `parser_version` columns to `opinions`.
6. **CLI entry point.** `pac-parse`. Accept `--opinion-id`, `--year`, `--reparse` (force).

**Acceptance criteria:**

- `just parse` parses every PDF in the manifest with `download_status = 'ok'`.
- Section detection succeeds on >95% of post-2010 opinions. Log and inspect failures.
- OCR fallback triggers automatically for scanned documents and produces non-empty text.
- Parser version is recorded so you can re-run only the documents parsed by an older version.

---

### Milestone 3: Structured metadata extraction

**Package:** `pac_pipeline`

**Goal:** For each parsed opinion, extract structured metadata via Claude Haiku and store it in DuckDB. This is the layer that turns plain text into a queryable legal database.

**Subtasks:**

1. **Schema.**
   ```python
   class OpinionMetadata(BaseModel):
       opinion_id: str
       issue_date: date | None
       counselor: str | None              # e.g., "Luke H. Britt"
       complainant: str | None            # null if redacted
       public_agency: str
       agency_type: Literal[
           "county", "city", "town", "school_corporation", "state_agency",
           "law_enforcement", "court", "library", "utility", "other"
       ]
       statutes_invoked: list[str]        # e.g., ["IC 5-14-3-4(b)(8)"]
       statute_areas: list[Literal["APRA", "ODL"]]
       issue_tags: list[str]              # controlled vocab; see below
       finding: Literal[
           "violation", "no_violation", "mixed",
           "moot", "no_jurisdiction", "withdrawn"
       ]
       summary: str                       # 2–3 sentences, model-written
   ```
2. **Issue-tag vocabulary.** Define a closed set of ~30 tags covering the recurring issue types: `deliberative-materials`, `executive-session-purpose`, `executive-session-notice`, `fee-charging`, `response-timeline`, `reasonable-particularity`, `personnel-files`, `investigatory-records`, `attorney-client-privilege`, `commercial-trade-secrets`, `serial-meetings`, `electronic-meetings`, etc. The LLM must pick from this list. Add new tags only after seeing the same novel concept three times.
3. **Extractor.** Send the BACKGROUND + ANALYSIS + CONCLUSION sections (not the full document; the caption and signature add noise) to Haiku with a system prompt that defines each enum value precisely. Use Pydantic with the SDK to enforce structure. Retry on validation failure with the validation error pasted back as a follow-up.
4. **Counselor identification.** Counselor names are stable across many opinions; cache them so Haiku doesn't have to re-derive. The signature line at the end is the most reliable source.
5. **Validation pass.** Hand-check 30–50 extractions against the source. Most-frequent failure modes will be `finding` (especially `mixed` vs `violation`) and `agency_type`. Adjust the prompt and re-run only the failures.
6. **Storage.** Write metadata to a DuckDB table `opinion_metadata` keyed on `opinion_id`. Keep history: never overwrite without recording `extracted_at` and `extractor_version`.
7. **CLI entry point.** `pac-extract`. Accept `--opinion-id`, `--reextract`, `--limit`.

**Acceptance criteria:**

- 100% of parsed opinions have a metadata row.
- A sample of 30 hand-checked extractions has ≥90% correct `finding`, ≥95% correct `statute_areas`, ≥85% correct primary `issue_tags`.
- Re-running with a new `extractor_version` updates rows but preserves the older extraction in a history table for diffing.

---

### Milestone 4: Chunking, embedding, indexing

**Package:** `pac_pipeline`

**Goal:** Produce three search artifacts that stay in lockstep: a Qdrant collection of chunk vectors, a Tantivy index of chunk text, and a DuckDB table of chunk-level metadata.

**Subtasks:**

1. **Chunking strategy.** Use the section offsets from Milestone 2.
   - Within ANALYSIS, split on paragraph breaks. Pack paragraphs greedily into chunks of 300–600 tokens with ~75-token overlap.
   - BACKGROUND and CONCLUSION each become single chunks unless they exceed 600 tokens, in which case split on paragraphs.
   - Generate a synthetic "**document card**" chunk per opinion: the model-generated summary + caption fields + finding. Tag this chunk with `section = "card"`.
2. **Chunk metadata.** Every chunk carries the full document metadata denormalized (opinion_id, year, agency_type, statute_areas, finding, issue_tags). This enables filter pushdown at query time.
   ```python
   class Chunk(BaseModel):
       chunk_id: str                # f"{opinion_id}#c{idx}"
       opinion_id: str
       chunk_index: int
       section: str                 # "card" | "background" | "analysis" | "conclusion"
       page_num: int                # for citation; first page the chunk starts on
       text: str
       token_count: int
       # denormalized:
       year: int
       agency_type: str
       statute_areas: list[str]
       finding: str
       issue_tags: list[str]
   ```
3. **Embedding.** Use sentence-transformers with `device="mps"`. Batch size 32. Always `normalize_embeddings=True`. The model is `BAAI/bge-large-en-v1.5`, dim 1024, distance Cosine. Embed once; persist.
4. **Qdrant collection.** Name: `pac_opinions_v1`. Vector params: size 1024, distance Cosine. Payload: the full chunk metadata above. Create payload indexes on `year`, `agency_type`, `statute_areas`, `finding`, `issue_tags` for filter pushdown.
5. **Tantivy index.** Schema fields: `chunk_id` (raw, stored), `opinion_id` (raw, stored), `text` (en_stem, stored), and the same filterable scalars as Qdrant for symmetry. Tantivy and Qdrant are queried in parallel; their result sets must be joinable on `chunk_id`.
6. **DuckDB chunks table.** Mirror of the chunk metadata for offline analysis and to support filter-only retrieval (no semantic, no lexical, just SQL).
7. **CLI entry point.** `pac-embed`. Accept `--reindex` (drop and rebuild) and `--opinion-id` for partial reindex.

**Acceptance criteria:**

- All three indexes have the same chunk count.
- A targeted query against Qdrant with a payload filter (`year >= 2022`, `agency_type = "school_corporation"`) returns only matching chunks.
- A Tantivy query for `IC 5-14-3-4` returns chunks containing that exact citation.
- Reindexing one opinion updates all three indexes atomically (or as close to it as the underlying systems allow; document any window of inconsistency).

---

### Milestone 5: Retrieval pipeline

**Package:** `pac_pipeline`

**Goal:** A `retrieve(question, filters)` function that returns the top-k most relevant chunks with scores and provenance.

**Subtasks:**

1. **Filter inference.** A small classifier (Haiku, structured output) reads the user question and proposes filters where the question implies them (e.g., "school board executive sessions" -> `agency_type = "school_corporation"`, `issue_tags ∋ "executive-session-*"`). Filters are *suggestions*; if applying them returns fewer than 8 candidates, retry without filters and note the relaxation in the response.
2. **Hybrid retrieval.** Run Qdrant (top 30) and Tantivy (top 30) in parallel via `asyncio.gather`. Embed the question once; reuse for Qdrant.
3. **RRF fusion.** Combined score = Σ over lists of `1 / (60 + rank)`. Take top 30 fused.
4. **Reranker.** Score every (question, chunk) pair with bge-reranker-v2-m3 on MPS. Sort descending, keep top 8–10.
5. **Return contract.**
   ```python
   class RetrievedChunk(BaseModel):
       chunk: Chunk
       dense_rank: int | None
       lexical_rank: int | None
       fused_score: float
       rerank_score: float

   class RetrievalResult(BaseModel):
       question: str
       applied_filters: dict
       relaxed: bool
       chunks: list[RetrievedChunk]
   ```
6. **Caching.** Cache embedding-of-question and rerank-scores on a content hash. In-memory LRU is enough for v1; do not add Redis.

**Acceptance criteria:**

- For 10 hand-written test questions, the top-3 reranked chunks always include a chunk from at least one obviously-relevant opinion.
- Median end-to-end retrieval latency under 800 ms on the dev machine for the warm path (model loaded, no Anthropic call).
- Filter inference correctly identifies `statute_areas` for queries that mention "open meetings" or "public records" without naming the statute.

---

### Milestone 6: FastAPI service

**Package:** `pac_api`

**Goal:** A minimal HTTP API that the frontend (and future API consumers) calls.

**Endpoints:**

- `POST /ask` — body: `{question: str, filters?: dict}`. Returns: the model's answer, the structured citations, the retrieved chunks, and the applied filters. Streams tokens via SSE.
- `GET /opinions/{opinion_id}` — full metadata + parsed text + link to original PDF.
- `GET /opinions` — paginated browse with filters (year, agency_type, finding, statute_areas, issue_tags).
- `GET /search` — pure retrieval, no generation. Same retrieval contract as Milestone 5.
- `GET /healthz` — liveness, plus index sizes and last-update timestamps.

**Generation prompt requirements:**

The system prompt for `/ask` must:

1. State that the model is answering questions about Indiana PAC advisory opinions.
2. Require every factual claim to be cited in the form `(opinion_id, p. N)`.
3. Require the model to say "the retrieved opinions don't directly address this" when they don't, rather than guess.
4. Require an advisory-status disclaimer at the end: "PAC opinions are advisory and persuasive only, not binding on agencies or courts."
5. Forbid fabricated citations. If the model cites an opinion ID that is not in the retrieved set, post-process and reject the response.

**Validation:** A response post-processor parses citations from the answer, checks every `opinion_id` against the retrieval result, and fails the request (with a logged warning) if any citation doesn't match. This catches hallucinated cites at the API boundary.

**Acceptance criteria:**

- `just api` starts the service on port 8000 with auto-reload.
- `/ask` streams a coherent, citation-bearing answer for a real question.
- The citation validator rejects a deliberately corrupted response (test it).
- `/healthz` returns chunk counts that match across Qdrant, Tantivy, and DuckDB.

---

### Milestone 7: Frontend

**App:** `apps/web`

**Goal:** A clean single-page interface for asking questions and browsing opinions. Editorial, not flashy. Justin's design preferences: navy/cream palette, Literata for body type, restrained chrome, no gradients.

**Routes:**

- `/` — the question box. After submit: streamed answer at the top, retrieved-opinion sidebar at right with citation links, applied-filter chips. Empty state shows three or four example questions.
- `/opinions` — filterable, paginated list. Filters: year range, agency type, finding, statute area, issue tags.
- `/opinions/[id]` — single-opinion view. Header: caption, counselor, finding. Body: the model's summary, then the parsed sections rendered as prose. Sidebar: link to original PDF, related opinions.
- `/about` — what the system is, what it isn't, advisory-status disclaimer in full.

**Components:** the shadcn primitives already installed (Button, Card, Input, Textarea, Badge, Separator, Skeleton). Add `dialog`, `command`, `scroll-area`, and `sonner` (toast) as needed.

**API client:** TypeScript fetch wrappers in `apps/web/src/lib/api.ts`. Use `EventSource` or `ReadableStream` reading for the SSE on `/ask`.

**Acceptance criteria:**

- `just web` runs the dev server.
- Asking a question streams an answer with clickable citations that scroll the sidebar to the matching opinion.
- The opinions list filters correctly and pagination works.
- The site has no flash of unstyled content and passes basic Lighthouse accessibility.

---

### Milestone 8: Eval harness

**Package:** `pac_eval`

**Goal:** A small gold set and an automated scorecard so changes to chunking, retrieval, or generation can be evaluated empirically.

**Subtasks:**

1. **Gold set.** 30–50 Q&A pairs, each with: a question, the *expected* opinion(s) that should appear in the retrieved set, a free-form ideal-answer summary, and the key facts the answer must contain. Hand-write these by reading 50 random opinions across years and statute areas.
2. **Retrieval metrics.** For each question, compute:
   - Recall@k (k=8, 16): does the expected opinion appear in the top-k chunks?
   - MRR of the first relevant chunk.
3. **Answer metrics.** For each question:
   - Citation precision: of citations the model produced, what fraction point to chunks actually retrieved?
   - Fact coverage: an LLM judge (Claude Sonnet) compares the answer to the ideal-answer summary and reports % of key facts present. Use a fresh prompt per question; no shared context.
   - Hedging score: heuristic check that the advisory disclaimer is present and the model doesn't overclaim authority.
4. **CLI.** `pac-eval run` runs the gold set and writes a Parquet scorecard to `data/eval/<timestamp>.parquet`. `pac-eval diff <a> <b>` compares two runs.

**Acceptance criteria:**

- Baseline scorecard exists.
- Running it twice with no code changes produces near-identical numbers.
- Future changes to chunking or retrieval can be A/B tested by running the eval before and after.

---

### Milestone 9: Update pipeline and deployment

**Goal:** Keep the system current with new opinions, and run it on the homelab.

**Subtasks:**

1. **Weekly crawl.** A GitHub Actions cron workflow that runs `pac-crawl` and, if the manifest changed, runs the rest of the pipeline. Commit the updated manifest summary (counts only; the DuckDB itself stays out of git) and open a PR if any new opinions were ingested.
2. **Homelab deployment.** Two long-running processes on a Proxmox VM, managed via systemd:
   - `pac-api.service`: uvicorn serving the FastAPI app.
   - `pac-web.service`: `next start` serving the prod build.
   Indexing runs as a separate timer-triggered unit (`pac-index.timer`) so it never collides with the API on the embedded Qdrant. Acceptable: the API briefly returns 503 while reindexing, since indexing is rare.
3. **Public access.** Cloudflare Tunnel pointing at the homelab. No port forwarding. Free hostname.
4. **Secrets.** `.env` on the homelab is managed by 1Password CLI (`op run -- ...`) and never committed. Anthropic API key is the only real secret.
5. **Backup.** Nightly rsync of `data/catalog/`, `data/qdrant/`, `data/tantivy/` to the NAS. PDFs are reproducible from in.gov, so they don't need to be backed up, but parsed JSON should be.

**Concurrent indexer + serving (deferred):** if at some point the API needs to serve while indexing is running, swap embedded Qdrant for the native Qdrant binary managed by systemd. The Python code change is essentially `QdrantClient(path=...)` -> `QdrantClient(host="localhost", port=6333)`. On-disk format is compatible; no re-embedding needed.

**Acceptance criteria:**

- New opinion appears on the PAC site -> within a week, it's in the system and answerable.
- Restarting the homelab brings everything back up automatically.
- The site is reachable from the public internet via the Cloudflare hostname with HTTPS.

---

## Known data quirks

These are the bugs in the source data, not in the system. Plan for them.

1. **Opinion ID vs filename mismatch.** The 25-FC-138 listing entry links to a file named `26-FC-138-LaFontaine.pdf`. There are likely other instances. Always derive the canonical opinion ID from the listing-page text, never from the PDF filename. When mismatches are detected, log them and flag the row.
2. **REVISED suffixes.** Some opinions have a REVISED version published later. Treat REVISED as an increment of the `revision` column on the same `opinion_id`. RAG should prefer the latest revision but keep older ones searchable for completeness.
3. **A/B sub-opinions.** Some opinion IDs are split (e.g., `24-FC-77A`, `24-FC-77B`). These are companion opinions, often cross-referenced. Store as separate rows; track the relationship via `parent_opinion_id`.
4. **Cross-linked listing entries.** Sub-opinions sometimes link to each other's PDFs by mistake on the listing page. Detect this by comparing the file's caption block (which always contains the canonical opinion ID) against the listing-page row.
5. **Redacted complainants.** Many recent opinions are titled "Letter-Redacted." The PDF often will not contain the unredacted name. The extractor should set `complainant = null`, not invent.
6. **Pre-2008 scans.** Older opinions may be image-only PDFs requiring OCR. LiteParse handles this automatically with the `--ocr` flag; the parse pipeline should detect low-text-density pages and re-run with OCR.
7. **Counselor turnover.** Different PACs (Heather Willis Neal, Joe Hoage, Luke H. Britt, Britt Prendergast, etc.) have recognizably different interpretive styles. Surface counselor as a filterable field; users may want to track how interpretations have shifted across administrations.
8. **Two listing pages, one corpus.** Formal complaints and informal opinions are on separate index pages. Both belong in the system. They are differentiated by the `kind` column (`FC` vs `INF`). Informal opinions are often shorter and don't always have a "BACKGROUND/ANALYSIS/CONCLUSION" structure; the section detector should handle missing sections gracefully.

---

## Out of scope for v1

Deliberately excluded. Don't build these without an explicit go-ahead.

- Cross-opinion citation graph (which opinions cite which other opinions). Useful, not urgent.
- A graph database (Kuzu, Neo4j). The relational layer is sufficient for v1.
- Personalization, accounts, saved searches.
- Comparison to actual court rulings or statutory text. PAC opinions are non-binding; mixing in binding sources changes the legal posture in ways that need careful design.
- Indiana Court of Appeals / Supreme Court opinions on APRA/ODL disputes. Same reason.
- A general-purpose legal research tool. Stay focused on PAC opinions.

---

## Quick start for the next session

```bash
cd $REPO
uv sync --all-packages
cp .env.example .env       # then drop in your ANTHROPIC_API_KEY
uv run python scripts/warm_models.py
uv run python scripts/check_anthropic.py
uv run python scripts/smoke_test.py
```

If all four pass, the environment is healthy. Begin with Milestone 1.

When in doubt, prefer simplicity, idempotence, and tests over ambition.
