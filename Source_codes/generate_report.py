"""
generate_report.py
Re-implements the same VSM / phrase / proximity search logic used in
app.html (JS) in Python, against the same precomputed index_data.json,
and writes a static Markdown test report satisfying Part E of the
assignment. No document IDs are hard-coded -- every ID below was produced
by actually running these queries against the corpus.
"""
import json
import math

DATA = json.load(open("/home/claude/build/index_data.json", encoding="utf-8"))
N = DATA["N"]
INDEX = DATA["index"]
DOC_VECTORS = DATA["docVectors"]
DOCS = DATA["docs"]

# ---- must match build_index.py's preprocess() exactly ----
import re
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

def stem(w):
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith("ing"):
        s = w[:-3]
        if len(s) > 2 and s[-1] == s[-2] and s[-1] not in "aeiou":
            s = s[:-1]
        return s
    if len(w) > 4 and w.endswith("ed"):
        s = w[:-2]
        if len(s) > 2 and s[-1] == s[-2] and s[-1] not in "aeiou":
            s = s[:-1]
        return s
    if len(w) > 4 and w.endswith("es"):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w

TOKEN_RE = re.compile(r"[a-z0-9]+")

def preprocess(text):
    out = []
    for t in TOKEN_RE.findall(text.lower()):
        if t in STOPWORDS or t.isdigit():
            continue
        out.append(stem(t))
    return out


def vsm_search(query_text):
    q_stems = preprocess(query_text)
    q_counts = {}
    unknown = []
    for t in q_stems:
        if t not in INDEX:
            if t not in unknown:
                unknown.append(t)
            continue
        q_counts[t] = q_counts.get(t, 0) + 1

    q_weights = {}
    for t, tf in q_counts.items():
        df = INDEX[t]["df"]
        idf = math.log10(N / df)
        q_weights[t] = (1 + math.log10(tf)) * idf
    norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0
    for t in q_weights:
        q_weights[t] /= norm

    candidates = set()
    for t in q_weights:
        for p in INDEX[t]["postings"]:
            candidates.add(p["doc"])

    scored = []
    for docid in candidates:
        dvec = DOC_VECTORS[docid]
        score = sum(q_weights[t] * dvec[t] for t in q_weights if t in dvec)
        if score > 0:
            scored.append((docid, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored[:10], len(scored), list(q_counts.keys()), unknown


def phrase_search(words):
    stems = preprocess(" ".join(words))
    if any(t not in INDEX for t in stems):
        return [], 0, stems, [t for t in stems if t not in INDEX]
    first_postings = {p["doc"]: p["pos"] for p in INDEX[stems[0]]["postings"]}
    results = []
    for docid, pos0_list in first_postings.items():
        pos_lists = []
        ok_doc = True
        for t in stems:
            entry = next((p for p in INDEX[t]["postings"] if p["doc"] == docid), None)
            if entry is None:
                ok_doc = False
                break
            pos_lists.append(set(entry["pos"]))
        if not ok_doc:
            continue
        hits = [p for p in pos_lists[0] if all((p + i) in pos_lists[i] for i in range(1, len(pos_lists)))]
        if hits:
            results.append((docid, len(hits)))
    results.sort(key=lambda x: (-x[1], x[0]))
    return results[:10], len(results), stems, []


def proximity_search(a, b, k):
    stems = preprocess(a + " " + b)
    if len(stems) < 2 or stems[0] not in INDEX or stems[1] not in INDEX:
        return [], 0, stems, [t for t in stems if t not in INDEX]
    tA, tB = stems[0], stems[1]
    postB = {p["doc"]: p["pos"] for p in INDEX[tB]["postings"]}
    results = []
    for p in INDEX[tA]["postings"]:
        if p["doc"] not in postB:
            continue
        hits = [(pa, pb) for pa in p["pos"] for pb in postB[p["doc"]] if 0 < pb - pa <= k]
        if hits:
            results.append((p["doc"], len(hits)))
    results.sort(key=lambda x: (-x[1], x[0]))
    return results[:10], len(results), stems, []


FREE_TEXT_TESTS = [
    "cotton shirt", "denim jeans", "festive saree", "winter jacket", "breathable fabric",
    "regular fit kurta", "printed dress", "zip closure jacket", "stretch leggings", "sweatshirt hoodie",
]
UNKNOWN_TERM_TEST = "silk lehenga"
PHRASE_TESTS = ["cotton shirt", "stretch denim", "festive wear", "winter wear", "regular fit"]
PROXIMITY_TESTS = [("cotton", "shirt", 3), ("stretch", "denim", 5), ("festive", "kurta", 4)]

lines = []
lines.append("# Part E — Test Report (generated, no hard-coded document IDs)\n")
lines.append("All results below were produced by running `generate_report.py` against "
              "`index_data.json`, the same precomputed index the web app (`app.html`) uses. "
              "The identical query engine is also runnable live from the app's **Test report** tab.\n")

lines.append("## 1. Free-text queries (VSM, lnc.ltc) — top 10 by cosine similarity\n")
for q in FREE_TEXT_TESTS:
    res, total, terms, unknown = vsm_search(q)
    lines.append(f"**\"{q}\"** — {total} matching document(s)\n")
    lines.append("| Rank | DocID | Category | Title | Score |")
    lines.append("|---|---|---|---|---|")
    for i, (docid, score) in enumerate(res, 1):
        d = DOCS[docid]
        lines.append(f"| {i} | {docid} | {d['category']} | {d['title']} | {score:.4f} |")
    lines.append("")

lines.append("## 2. Query containing a term absent from the corpus\n")
res, total, terms, unknown = vsm_search(UNKNOWN_TERM_TEST)
lines.append(f"**\"{UNKNOWN_TERM_TEST}\"** — recognized terms: {terms or '(none)'}; "
             f"unrecognized (not in dictionary): {unknown or '(none)'}; {total} matching document(s).\n")
if res:
    lines.append("| Rank | DocID | Score |")
    lines.append("|---|---|---|")
    for i, (docid, score) in enumerate(res, 1):
        lines.append(f"| {i} | {docid} | {score:.4f} |")
else:
    lines.append("_No results — the recognized term(s), if any, do not co-occur with anything scoreable, "
                 "or none of the query terms exist in the dictionary._")
lines.append("")

lines.append("## 3. Exact phrase queries (positional index)\n")
phrase_outcomes = {}
for q in PHRASE_TESTS:
    res, total, stems, missing = phrase_search(q.split(" "))
    phrase_outcomes[q] = (res, total, stems, missing)
    lines.append(f"**\"{q}\"** — {total} document(s) contain this exact adjacent sequence\n")
    lines.append("| DocID | Occurrences |")
    lines.append("|---|---|")
    for docid, occ in res:
        lines.append(f"| {docid} | {occ} |")
    lines.append("")

lines.append("## 4. Ordered proximity queries (different k)\n")
for a, b, k in PROXIMITY_TESTS:
    res, total, stems, missing = proximity_search(a, b, k)
    lines.append(f"**{a} WITHIN/{k} {b}** (ordered: \"{a}\" must precede \"{b}\") — {total} document(s)\n")
    lines.append("| DocID | Occurrences |")
    lines.append("|---|---|")
    for docid, occ in res:
        lines.append(f"| {docid} | {occ} |")
    lines.append("")

lines.append("## 5. Where positional information changes the result set/order\n")
case_num = 1
for q in PHRASE_TESTS:
    vsm_res, vsm_total, _, _ = vsm_search(q)
    ph_res, ph_total, _, _ = phrase_outcomes[q][0], phrase_outcomes[q][1], None, None
    vsm_docs = set(d for d, _ in vsm_res)
    ph_docs = set(d for d, _ in phrase_outcomes[q][0])
    only_vsm = sorted(vsm_docs - ph_docs)
    if only_vsm or (ph_docs - vsm_docs):
        top_vsm = vsm_res[0] if vsm_res else None
        lines.append(f"**Case {case_num}: \"{q}\"**")
        lines.append(f"- Free-text VSM: {vsm_total} document(s) match because they contain both stems "
                     f"anywhere in the description" + (f" (top match {top_vsm[0]} at {top_vsm[1]:.4f})." if top_vsm else "."))
        lines.append(f"- Exact phrase: only {phrase_outcomes[q][1]} document(s) contain the terms adjacent and in order.")
        if only_vsm:
            lines.append(f"- Documents ranked under free-text but absent from the phrase result set: "
                         f"{', '.join(only_vsm[:5])}{'…' if len(only_vsm) > 5 else ''} — "
                         f"the words occur separately in these descriptions, not next to each other.")
        lines.append("")
        case_num += 1
    if case_num > 5:
        break

open("/home/claude/deliverables/test_report.md", "w", encoding="utf-8").write("\n".join(lines))
print("wrote test_report.md")
