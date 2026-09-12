# Selvage — Clothing Search Engine

## What this is
A ranked search engine over the 100-document clothing corpus, built as a
single self-contained web app (`app.html`) — no server, no install. Open it
in any browser.

It implements every part of the assignment:

- **Part A** — tokenize, normalize, remove stop-words, stem (`source_code/build_index.py`)
- **Part B** — VSM ranked retrieval with lnc.ltc cosine similarity, top 10 results
- **Part C** — positional index, exact phrase search, ordered proximity search
- **Part D** — the app itself: free-text search, phrase/proximity search, an
  index explorer, and a results panel you click into (like a search-engine
  results page) to see the full product text with matched terms highlighted
  and their exact token positions shown as evidence

## How to use `app.html`
Open the file in a browser. Three tabs:

1. **Search** — free-text queries, ranked by lnc.ltc cosine similarity.
   **Type of cloth / Colour / Size** filter dropdowns (each defaulting to
   **Any**) narrow which documents are searched before ranking runs — a
   custom addition beyond the base spec, see "Search filters" below. Click
   any result to open it in a centered detail view with the full
   description, matched terms highlighted, and their positions in the
   document.
2. **Phrase & proximity** — type `"cotton shirt"` for an exact phrase, or
   `cotton WITHIN/3 shirt` for ordered proximity (first term must occur
   before the second, within the given number of token positions). Click a
   result to see the exact matched span highlighted.
3. **Index** — look up any term to see its document frequency and its full
   postings list (docID, tf, positions) — the dictionary and positional
   index in browsable form.

To exercise Part E's mandatory test queries, run `python3 source_code/generate_report.py`
(regenerates `test_report.md`) or simply type each query into the Search /
Phrase & proximity tabs and note the results — the app has no separate
test-runner UI.

## Search filters (custom addition, beyond the base assignment spec)
The Search tab has three dropdowns — **Type of cloth**, **Colour**, and
**Size** — each defaulting to **Any**. Setting one or more of them narrows
the candidate document set *before* lnc.ltc cosine ranking runs, so the
query is only scored against documents that match every active filter.
The values come straight from the corpus's own data: type of cloth is the
real `CATEGORY` field (Shirt, T-Shirt, Jeans, Kurta, Saree, Dress, Hoodie,
Jacket, Leggings, Sweatshirt), colour is read from the end of each title
(e.g. "... - Navy Blue"), and size is read from each description's
"Available in size ..." sentence. A **Reset filters** button clears all
three back to Any.

This replaces an earlier version of the app that instead added a flat +1
to the cosine score of documents in a category named in the query text
(e.g. boosting Shirt above T-Shirt for a "shirt" query). Explicit filters
give the same control without silently reweighting the ranking, and they
also work on colour and size, which the query text alone couldn't express.
Filtering only applies to free-text search (Part B); phrase/proximity
search (Part C) is unaffected and remains purely structural, per spec. The
lnc.ltc cosine ranking itself is never modified — filtering only changes
which documents are considered.

## Why a web app instead of a terminal program
Search results you can click into — where clicking a ranked result opens
the actual product description, centered on screen with the matching terms
highlighted — is far easier to read, demo, and screenshot than scrolling
terminal output, and matches how a real search engine's results page lets
you preview an item without leaving the results.

## Stop-word policy
A standard list of English function words (articles, prepositions,
conjunctions, pronouns, auxiliary/copula verbs — ~90 words) is applied
identically to documents and queries. The corpus is short, templated
product-description text ("Made from ... this ... is designed for ...")
full of connective language with no retrieval value, so removing it keeps
the dictionary focused on clothing-relevant terms. The exact list lives in
`source_code/build_index.py` (Python, used to build the index) and is
duplicated verbatim in `source_code/pipeline.js` (JS, used for query-time
processing in the browser) — a parity check (below) confirms both produce
identical stemmed token streams.

## Stemmer
A small, transparent suffix-stripping stemmer (not full Porter): strips
`-ies→y`, `-ing`, `-ed`, `-es`, `-s`, with simple doubled-consonant undoing
(`fitting→fit`). It's deliberately simple so its behavior is easy to justify
in a viva. Known limitation: because it's not a true linguistic stemmer, a
few related forms don't fully collapse (e.g. `size` and `sizes` stem to
`size` and `siz` respectively) — documented here rather than hidden.

## Project structure
```
app.html                    the application — open this in a browser
test_report.md              static Part E report (mirrors the in-app Test report tab)
index_output/
  inverted_index.txt        Part A/B deliverable: term -> df -> postings(doc, tf)
  positional_index.txt      Part C deliverable: term -> df -> postings(doc, tf, positions)
source_code/
  build_index.py            parses the corpus, builds the index, writes index_data.json
                             (embedded into app.html) and the two .txt files above
  pipeline.js                the exact same tokenize/stopword/stem pipeline, in JS,
                             used by app.html for query-time processing
  app.js                    the search engine logic (VSM, phrase, proximity, rendering) —
                             embedded into app.html; kept here separately for readability
  corpus_100.txt            the original supplied corpus
README.md                   this file
```

## How the index was built
```
python3 source_code/build_index.py
```
reads `corpus_100.txt`, applies the pipeline described above, and writes:
- `index_data.json` (embedded into `app.html` at build time — this is what
  the browser actually queries against; it is not re-derived at page load)
- `index_output/inverted_index.txt`
- `index_output/positional_index.txt`

Only three things happen client-side in the browser: tokenizing/stemming the
*query* text (identical pipeline to the corpus), computing cosine similarity
against the precomputed document vectors, and walking the precomputed
position lists for phrase/proximity search. A Node-based parity check
(`verify_parity.js`, not included in this package but described here for
transparency) confirmed the JS pipeline reproduces the Python-built index
exactly, token-for-token, across all 100 documents before this package was
finalized.

## Weighting scheme (Part B)
- Document weight: `w(d,t) = 1 + log10(tf)`, no idf, cosine-normalized.
- Query weight: `w(q,t) = (1 + log10(tf)) × log10(N/df)`, N = 100,
  cosine-normalized.
- Score = dot product of the normalized vectors.
- Ties broken by increasing document ID; top 10 returned.

## Screenshots
Included for quick look on final state
