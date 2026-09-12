"""
build_index.py
Information Retrieval Assignment 1 (CSD358)

Parses the 100-document clothing corpus, applies a tokenize -> normalize ->
stop-word-removal -> stem pipeline, and builds:
  1. An inverted index: term -> df -> postings [(docID, tf), ...]
  2. A positional index: term -> df -> postings [(docID, tf, [pos1, pos2, ...]), ...]
  3. Per-document lnc (log-tf, no idf, cosine-normalized) weight vectors for VSM.

All of this is dumped to a single JSON file (index_data.json) that is embedded
into the web application (app.html), so the browser does not need to redo any
corpus parsing or index construction -- only *query-side* tokenization (ltc
weighting) happens in the browser, using the identical stemming/stopword
pipeline reimplemented in JS (see the <script> in app.html).

Run: python3 build_index.py
"""

import re
import json
import math
from collections import defaultdict

CORPUS_PATH = "/mnt/user-data/uploads/corpus_100.txt"
OUT_JSON = "/home/claude/build/index_data.json"
OUT_INVERTED_TXT = "/home/claude/build/inverted_index.txt"
OUT_POSITIONAL_TXT = "/home/claude/build/positional_index.txt"

# ---------------------------------------------------------------------------
# Stop-word policy
# ---------------------------------------------------------------------------
# We use a standard, fairly compact English stop-word list (function words:
# articles, prepositions, conjunctions, auxiliary/copula verbs, pronouns).
# Justification: the corpus is short, templated product-description text
# ("Made from ... this ... is designed for ... It features ...") full of
# generic connective language that carries no discriminative retrieval
# value for clothing search (e.g. a query for "cotton shirt" should not
# match on "this" or "for"). We deliberately do NOT stop domain-relevant
# words that a naive stop-list sometimes includes (e.g. we keep "s" is not
# a real word so n/a). The same fixed list is applied consistently to both
# documents (Part A) and queries (Part B/C) so index and query vocabularies
# line up. This exact list is duplicated verbatim in app.html's JS so that
# document-side and query-side processing never diverge.
STOPWORDS = set("""
a an the and or but if then else for nor so yet
is are was were be been being am
of to in on at by with from into onto over under
this that these those it its it's
as well also can could should would will shall may might must
not no do does did doing done
i you he she we they him her them his hers our ours your yours their theirs
which who whom what when where why how
than too very just only own same
there here
""".split())

# ---------------------------------------------------------------------------
# Stemmer: simple, transparent suffix-stripping stemmer (not full Porter).
# Order of rules matters -- longer/more specific suffixes checked first.
# Mirrored exactly in JS (see app.html) so document-side and query-side
# stems always agree.
# ---------------------------------------------------------------------------
def stem(word):
    w = word
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ing"):
        stripped = w[:-3]
        # undo simple doubled final consonant, e.g. "fitting" -> "fitt" -> "fit"
        if len(stripped) > 2 and stripped[-1] == stripped[-2] and stripped[-1] not in "aeiou":
            stripped = stripped[:-1]
        return stripped
    if len(w) > 4 and w.endswith("ed"):
        stripped = w[:-2]
        if len(stripped) > 2 and stripped[-1] == stripped[-2] and stripped[-1] not in "aeiou":
            stripped = stripped[:-1]
        return stripped
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize_normalize(text):
    """Lowercase, strip punctuation (keep alnum tokens), split on whitespace/punct."""
    return TOKEN_RE.findall(text.lower())


def preprocess(text):
    """Full pipeline: tokenize -> normalize case -> remove stopwords -> stem.
    Returns list of stemmed tokens IN ORIGINAL ORDER (so positions are
    positions within this stemmed-token stream, matching Part C's
    requirement that positions are stored 'after the same
    tokenization/normalization/stemming pipeline used in Part A')."""
    tokens = tokenize_normalize(text)
    out = []
    for t in tokens:
        if t in STOPWORDS:
            continue
        if t.isdigit():
            continue
        out.append(stem(t))
    return out


# ---------------------------------------------------------------------------
# Parse corpus
# ---------------------------------------------------------------------------
def parse_corpus(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    docs = []
    for block in re.findall(r"<DOC>(.*?)</DOC>", raw, re.S):
        docid = re.search(r"<DOCID>(.*?)</DOCID>", block, re.S).group(1).strip()
        category = re.search(r"<CATEGORY>(.*?)</CATEGORY>", block, re.S).group(1).strip()
        title = re.search(r"<TITLE>(.*?)</TITLE>", block, re.S).group(1).strip()
        text = re.search(r"<TEXT>(.*?)</TEXT>", block, re.S).group(1).strip()
        docs.append({"docid": docid, "category": category, "title": title, "text": text})
    return docs


def main():
    docs = parse_corpus(CORPUS_PATH)
    N = len(docs)
    assert N == 100, f"Expected 100 docs, found {N}"

    # doc_tokens[docid] = list of stemmed tokens (title + text combined,
    # title tokens are also indexed so a query on the title words still hits)
    doc_tokens = {}
    doc_display = {}
    for d in docs:
        full_text = d["title"] + ". " + d["text"]
        toks = preprocess(full_text)
        doc_tokens[d["docid"]] = toks
        # Colour and size are read straight off the corpus's own text so the
        # in-app type/colour/size filters (see app.html) work off real data:
        # colour is the segment of the title after the final " - ", and size
        # comes from the "Available in size X" sentence in the description.
        color = d["title"].rsplit(" - ", 1)[-1].strip()
        size_match = re.search(r"Available in size (\w+)", d["text"])
        size = size_match.group(1) if size_match else None
        doc_display[d["docid"]] = {
            "category": d["category"],
            "title": d["title"],
            "text": d["text"],
            "color": color,
            "size": size,
        }

    # ---------------- inverted + positional index ----------------
    # term -> {docid: {"tf": int, "positions": [..]}}
    index = defaultdict(lambda: defaultdict(lambda: {"tf": 0, "positions": []}))
    for docid, toks in doc_tokens.items():
        for pos, term in enumerate(toks):
            entry = index[term][docid]
            entry["tf"] += 1
            entry["positions"].append(pos)

    # doc vector lengths for VSM (lnc: log-tf, no idf, cosine normalized)
    doc_vectors = {}  # docid -> {term: weight}  (already normalized)
    doc_norms = {}
    for docid, toks in doc_tokens.items():
        tf_counts = defaultdict(int)
        for t in toks:
            tf_counts[t] += 1
        raw_weights = {t: 1 + math.log10(tf) for t, tf in tf_counts.items()}
        norm = math.sqrt(sum(w * w for w in raw_weights.values())) or 1.0
        doc_vectors[docid] = {t: w / norm for t, w in raw_weights.items()}
        doc_norms[docid] = norm

    # df per term
    df = {term: len(postings) for term, postings in index.items()}

    # ---------------- serialize JSON for the web app ----------------
    # Build compact structures.
    index_json = {}
    for term, postings in index.items():
        plist = []
        for docid, info in sorted(postings.items()):
            plist.append({"doc": docid, "tf": info["tf"], "pos": info["positions"]})
        index_json[term] = {"df": df[term], "postings": plist}

    doc_vectors_json = {
        docid: {t: round(w, 6) for t, w in vec.items()} for docid, vec in doc_vectors.items()
    }

    data = {
        "N": N,
        "docs": doc_display,          # docid -> category/title/text (for display + highlighting)
        "docTokenCount": {d: len(doc_tokens[d]) for d in doc_tokens},
        "index": index_json,          # term -> df, postings(doc, tf, pos[])
        "docVectors": doc_vectors_json,  # docid -> {term: normalized lnc weight}
        "stopwords": sorted(STOPWORDS),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f)

    print(f"Wrote {OUT_JSON}  ({len(index_json)} unique terms, {N} docs)")

    # ---------------- human-readable deliverable: inverted index ----------------
    with open(OUT_INVERTED_TXT, "w", encoding="utf-8") as f:
        f.write("INVERTED INDEX (term -> df -> postings[(docID, tf), ...])\n")
        f.write("=" * 70 + "\n\n")
        for term in sorted(index_json.keys()):
            entry = index_json[term]
            postings_str = ", ".join(f"({p['doc']}, tf={p['tf']})" for p in entry["postings"])
            f.write(f"{term}  (df={entry['df']})\n    -> {postings_str}\n\n")
    print(f"Wrote {OUT_INVERTED_TXT}")

    # ---------------- human-readable deliverable: positional index ----------------
    with open(OUT_POSITIONAL_TXT, "w", encoding="utf-8") as f:
        f.write("POSITIONAL INDEX (term -> df -> [(docID, tf, [positions]), ...])\n")
        f.write("=" * 70 + "\n\n")
        for term in sorted(index_json.keys()):
            entry = index_json[term]
            f.write(f"{term}  (df={entry['df']})\n")
            for p in entry["postings"]:
                f.write(f"    ({p['doc']}, tf={p['tf']}, positions={p['pos']})\n")
            f.write("\n")
    print(f"Wrote {OUT_POSITIONAL_TXT}")


if __name__ == "__main__":
    main()
