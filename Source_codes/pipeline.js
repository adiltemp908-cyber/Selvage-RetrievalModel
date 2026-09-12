// Query-side / display-side text pipeline.
// Mirrors build_index.py EXACTLY (same stopword list, same stemmer rules,
// same tokenizer regex, same digit filter, same "title + '. ' + text"
// combination) so that stemmed tokens and their positions always agree
// with the precomputed index shipped in DATA.

const STOPWORDS = new Set(`
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
`.trim().split(/\s+/));

function stem(w) {
  if (w.length > 4 && w.endsWith('ies')) return w.slice(0, -3) + 'y';
  if (w.length > 4 && w.endsWith('ing')) {
    let s = w.slice(0, -3);
    if (s.length > 2 && s[s.length - 1] === s[s.length - 2] && !'aeiou'.includes(s[s.length - 1])) s = s.slice(0, -1);
    return s;
  }
  if (w.length > 4 && w.endsWith('ed')) {
    let s = w.slice(0, -2);
    if (s.length > 2 && s[s.length - 1] === s[s.length - 2] && !'aeiou'.includes(s[s.length - 1])) s = s.slice(0, -1);
    return s;
  }
  if (w.length > 4 && w.endsWith('es')) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith('s') && !w.endsWith('ss')) return w.slice(0, -1);
  return w;
}

const TOKEN_RE = /[a-z0-9]+/g;

// Tokenize + keep character offsets into the ORIGINAL (not lowercased)
// string, so callers can slice out the original-cased word for highlighting.
function tokenizeWithOffsets(text) {
  const lower = text.toLowerCase();
  const out = [];
  let m;
  TOKEN_RE.lastIndex = 0;
  while ((m = TOKEN_RE.exec(lower)) !== null) {
    out.push({ token: m[0], start: m.index, end: m.index + m[0].length });
  }
  return out;
}

function isDigits(s) {
  return /^[0-9]+$/.test(s);
}

// Full pipeline: tokenize -> normalize -> remove stopwords/pure-numbers -> stem.
// Returns array of {stem, raw, start, end, pos} where `pos` is the index in
// the FILTERED stream (matches positions stored in the precomputed index).
function preprocessWithOffsets(text) {
  const toks = tokenizeWithOffsets(text);
  const out = [];
  let pos = 0;
  for (const t of toks) {
    if (STOPWORDS.has(t.token)) continue;
    if (isDigits(t.token)) continue;
    out.push({
      stem: stem(t.token),
      raw: text.slice(t.start, t.end),
      start: t.start,
      end: t.end,
      pos: pos,
    });
    pos++;
  }
  return out;
}

// Simple stems-only pipeline, for query processing.
function preprocess(text) {
  return preprocessWithOffsets(text).map(t => t.stem);
}

module.exports = { STOPWORDS, stem, tokenizeWithOffsets, preprocessWithOffsets, preprocess };
