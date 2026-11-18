# Data notes

## Sources

All data is from [GOV.UK](https://www.gov.uk), retrieved via two public APIs:

- **Search API** (`https://www.gov.uk/api/search.json`) — used to discover pages by format and topic.
- **Content API** (`https://www.gov.uk/api/content/<path>`) — used to fetch the full text of each page.

Licence: [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).
Credit: Contains public sector information licensed under the Open Government Licence v3.0.

## Date collected

18 November 2026

## Scope

### Formats included

| Format | Topics | Reason |
|--------|--------|--------|
| `answer` | all | Short factual pages; directly answerable questions |
| `guide` | all | Multi-part pages covering a topic in depth |
| `detailed_guide` | benefits, tax, working, housing, driving | Detailed policy guidance for citizens |

### Formats excluded

Transactions, smart answers, local transactions, place pages, licences, completed transactions, simple smart answers, calendars — these are interactive tools or forms, not guidance text.

Non-English pages (locale ≠ `en`) are excluded.

## Page counts

| Format | Count |
|--------|-------|
| answer | 769 |
| guide | 860 |
| detailed_guide | 1,684 |
| **Total** | **3,313** |

The detailed_guide count comes from four taxonomy topics: welfare/benefits (88), work/employment (195), housing and local community (1,026), transport/driving (404), minus 29 pages that were deduplicated across topics.

## Fetch results

- Pages successfully fetched and cached: **3,313**
- Pages that failed: **0** (see `data/fetch_failures.csv`)

## Chunk counts and token statistics

| Chunk size | Chunks | Min tokens | Avg tokens | Max tokens | Over limit |
|------------|--------|-----------|-----------|-----------|-----------|
| 400 | 48,855 | 8 | 126 | 1,001 | 104 (0.2%) |
| 800 | 46,990 | 8 | 129 | 1,001 | 36 (0.1%) |

The low average (126 tokens) reflects GOV.UK's structure: many short sections with focused factual content. The 0.2% over-limit cases are single long paragraphs or tables with no sentence boundaries — they cannot be split further without breaking meaning.

## Exclusions and decisions

1. **Non-English pages:** GOV.UK hosts Welsh (`cy`) and other languages. We index English only because the LLM and embedding model are English-only.
2. **Transaction-type formats:** These are links to external services, not guidance text — no body to index.
3. **`detailed_guide` topic filter:** Without a topic filter this format returns tens of thousands of pages (many are specialist technical guidance). We restrict to the five citizen-facing topic areas to keep the corpus focused and within the 3,000–6,000 page target.
4. **Parts vs body:** For `guide` format, we index each part separately to preserve per-part URLs in citations. For `answer` and `detailed_guide`, we index the single body field.

## Refresh

`src/citewise/ingest/refresh.py` compares the `public_timestamp` field in the manifest against what is stored in `data/raw/`. Pages whose timestamp has changed, or which are not cached at all, are re-fetched. This means re-running `make ingest` after a manifest update is safe and incremental.
