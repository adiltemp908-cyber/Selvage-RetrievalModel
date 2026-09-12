/* ---------- small helpers ---------- */
function escapeHtml(s){
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function fmtScore(x){ return x.toFixed(4); }

/* ---------- view switching ---------- */
const modeButtons = document.querySelectorAll('nav.modes button');
modeButtons.forEach(btn=>{
  btn.addEventListener('click', ()=>{
    modeButtons.forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    document.querySelectorAll('.view').forEach(v=>v.classList.remove('active'));
    document.getElementById('view-'+btn.dataset.view).classList.add('active');
  });
});

/* ---------- stats strip ---------- */
(function initStats(){
  const el = document.getElementById('statsStrip');
  const terms = Object.keys(DATA.index).length;
  el.innerHTML = `
    <span>${DATA.N} products indexed</span>
    <span>${terms} stemmed terms in the dictionary</span>
    <span>weighting: lnc.ltc</span>
  `;
})();

/* =========================================================
   Search filters (custom addition, beyond the base assignment spec)
   ---------------------------------------------------------
   Assignment Part B defines ranking purely by lnc.ltc cosine similarity
   over the full corpus. This optional addition lets the person narrow
   *which documents are searched* before that ranking runs, using three
   facets read straight off the corpus's real fields: CATEGORY ("type of
   cloth"), the colour named at the end of the title, and the size found
   in each description's "Available in size ..." line. Each facet can be
   set to "Any" to opt out of filtering on it. Filtering happens on the
   candidate set — the lnc.ltc cosine ranking itself (Part B) is
   untouched, and phrase/proximity search (Part C) is unaffected, exactly
   as the category boost this replaces used to work.
   ========================================================= */
function buildFilterOptions(){
  const cats = new Set(), colours = new Set(), sizes = new Set();
  for (const docid in DATA.docs){
    const d = DATA.docs[docid];
    cats.add(d.category);
    if (d.color) colours.add(d.color);
    if (d.size) sizes.add(d.size);
  }
  const sizeOrder = ['S','M','L','XL','XXL'];
  const fillSelect = (el, values) => {
    values.forEach(v=>{
      const opt = document.createElement('option');
      opt.value = v; opt.textContent = v;
      el.appendChild(opt);
    });
  };
  fillSelect(document.getElementById('filterType'), Array.from(cats).sort());
  fillSelect(document.getElementById('filterColour'), Array.from(colours).sort());
  fillSelect(document.getElementById('filterSize'), Array.from(sizes).sort((a,b)=>sizeOrder.indexOf(a)-sizeOrder.indexOf(b)));
}
buildFilterOptions();

function getActiveFilters(){
  return {
    category: document.getElementById('filterType').value,
    color: document.getElementById('filterColour').value,
    size: document.getElementById('filterSize').value,
  };
}
function docPassesFilters(docid, filters){
  const d = DATA.docs[docid];
  if (filters.category !== 'any' && d.category !== filters.category) return false;
  if (filters.color !== 'any' && d.color !== filters.color) return false;
  if (filters.size !== 'any' && d.size !== filters.size) return false;
  return true;
}

/* =========================================================
   PART B — free-text VSM search (lnc.ltc cosine similarity)
   ========================================================= */
function vsmSearch(queryText, filters){
  const qStems = preprocess(queryText);
  const qTermCounts = {};
  const unknownTerms = new Set();
  qStems.forEach(t=>{
    if (!DATA.index[t]) { unknownTerms.add(t); return; }
    qTermCounts[t] = (qTermCounts[t]||0) + 1;
  });

  // query weights: (1+log10(tf)) * log10(N/df)
  const qWeights = {};
  for (const t in qTermCounts){
    const tf = qTermCounts[t];
    const df = DATA.index[t].df;
    const idf = Math.log10(DATA.N / df);
    qWeights[t] = (1 + Math.log10(tf)) * idf;
  }
  const qNorm = Math.sqrt(Object.values(qWeights).reduce((a,w)=>a+w*w,0)) || 1;
  for (const t in qWeights) qWeights[t] = qWeights[t] / qNorm;

  // candidate docs: union of postings for query terms, restricted to
  // documents that pass the active type/colour/size filters
  const candidateDocs = new Set();
  for (const t in qWeights){
    DATA.index[t].postings.forEach(p=>{
      if (docPassesFilters(p.doc, filters)) candidateDocs.add(p.doc);
    });
  }

  const scores = [];
  candidateDocs.forEach(docid=>{
    const dvec = DATA.docVectors[docid];
    let score = 0;
    for (const t in qWeights){
      if (dvec[t]) score += qWeights[t] * dvec[t];
    }
    if (score > 0){
      const matchedTerms = Object.keys(qWeights).filter(t=>dvec[t]);
      scores.push({docid, score, finalScore: score, matchedTerms, category: DATA.docs[docid].category});
    }
  });

  scores.sort((a,b)=>{
    if (b.finalScore !== a.finalScore) return b.finalScore - a.finalScore;
    return a.docid.localeCompare(b.docid);
  });

  return {
    results: scores.slice(0,10),
    totalMatches: scores.length,
    queryStems: Object.keys(qTermCounts),
    unknownTerms: Array.from(unknownTerms),
  };
}

/* =========================================================
   PART C — positional index: phrase & ordered-proximity search
   ========================================================= */
function phraseSearch(termsRaw){
  const stems = preprocess(termsRaw.join(' '));
  if (stems.some(t=>!DATA.index[t])){
    const missing = stems.filter(t=>!DATA.index[t]);
    return {results:[], stems, missing};
  }
  // docs containing the first term
  const first = DATA.index[stems[0]];
  const candidateDocs = first.postings.map(p=>p.doc);
  const results = [];

  candidateDocs.forEach(docid=>{
    // positions of each stem in this doc
    const posLists = stems.map(t=>{
      const entry = DATA.index[t].postings.find(p=>p.doc===docid);
      return entry ? entry.pos : [];
    });
    if (posLists.some(pl=>pl.length===0)) return;

    const hits = []; // starting positions of a full phrase match
    posLists[0].forEach(startPos=>{
      let ok = true;
      for (let i=1;i<posLists.length;i++){
        if (!posLists[i].includes(startPos+i)) { ok=false; break; }
      }
      if (ok) hits.push(startPos);
    });
    if (hits.length>0) results.push({docid, occurrences:hits.length, spans:hits.map(h=>[h,h+stems.length-1])});
  });

  results.sort((a,b)=>{
    if (b.occurrences !== a.occurrences) return b.occurrences - a.occurrences;
    return a.docid.localeCompare(b.docid);
  });

  return {results: results.slice(0,10), totalMatches:results.length, stems, missing:[]};
}

function proximitySearch(termA, termB, k){
  const stems = preprocess(termA+' '+termB);
  if (stems.length < 2){
    return {results:[], stems, missing: stems.filter(t=>!DATA.index[t])};
  }
  const [tA, tB] = stems;
  if (!DATA.index[tA] || !DATA.index[tB]){
    return {results:[], stems, missing:[tA,tB].filter(t=>!DATA.index[t])};
  }
  const postA = DATA.index[tA].postings;
  const postB = DATA.index[tB].postings;
  const docsB = {};
  postB.forEach(p=>docsB[p.doc]=p.pos);

  const results = [];
  postA.forEach(pa=>{
    const posB = docsB[pa.doc];
    if (!posB) return;
    const hits = [];
    pa.pos.forEach(pA=>{
      posB.forEach(pB=>{
        const diff = pB - pA;
        if (diff > 0 && diff <= k) hits.push([pA,pB]);
      });
    });
    if (hits.length>0) results.push({docid:pa.doc, occurrences:hits.length, spans:hits});
  });

  results.sort((a,b)=>{
    if (b.occurrences !== a.occurrences) return b.occurrences - a.occurrences;
    return a.docid.localeCompare(b.docid);
  });

  return {results: results.slice(0,10), totalMatches: results.length, stems, missing:[]};
}

function parsePositionalQuery(raw){
  const trimmed = raw.trim();
  const phraseMatch = trimmed.match(/^"(.+)"$/);
  if (phraseMatch){
    const words = phraseMatch[1].trim().split(/\s+/);
    return {type:'phrase', words};
  }
  const proxMatch = trimmed.match(/^(.+?)\s+within\/(\d+)\s+(.+)$/i);
  if (proxMatch){
    return {type:'proximity', termA:proxMatch[1].trim(), k:parseInt(proxMatch[2],10), termB:proxMatch[3].trim()};
  }
  // fallback: treat bare multi-word input as a phrase query
  const words = trimmed.split(/\s+/);
  return {type:'phrase', words};
}

/* =========================================================
   Snippets & highlighting (built fresh per doc, same pipeline)
   ========================================================= */
function buildDocTokens(docid){
  const d = DATA.docs[docid];
  const fullText = d.title + '. ' + d.text;
  return {fullText, tokens: preprocessWithOffsets(fullText)};
}

function highlightSnippet(docid, matchedStems, maxLen){
  const {fullText, tokens} = buildDocTokens(docid);
  const stemSet = new Set(matchedStems);
  // find first token index that matches, to center the snippet
  let anchor = tokens.findIndex(t=>stemSet.has(t.stem));
  if (anchor === -1) anchor = 0;
  const anchorChar = tokens[anchor] ? tokens[anchor].start : 0;
  let start = Math.max(0, anchorChar - 60);
  let end = Math.min(fullText.length, start + maxLen);
  if (end - start < maxLen) start = Math.max(0, end - maxLen);

  let snippet = fullText.slice(start, end);
  const prefix = start>0 ? '…' : '';
  const suffix = end<fullText.length ? '…' : '';

  // build highlighted HTML by walking tokens within [start,end)
  let html = '';
  let cursor = start;
  tokens.forEach(t=>{
    if (t.end <= start || t.start >= end) return;
    html += escapeHtml(fullText.slice(cursor, t.start));
    const word = fullText.slice(t.start, t.end);
    if (stemSet.has(t.stem)) html += `<mark>${escapeHtml(word)}</mark>`;
    else html += escapeHtml(word);
    cursor = t.end;
  });
  html += escapeHtml(fullText.slice(cursor, end));
  return prefix + html + suffix;
}

function highlightFullText(docid, matchedStems, spanPositions){
  const {fullText, tokens} = buildDocTokens(docid);
  const stemSet = new Set(matchedStems);
  const spanPosSet = new Set(spanPositions || []);
  let html = '';
  let cursor = 0;
  tokens.forEach(t=>{
    html += escapeHtml(fullText.slice(cursor, t.start));
    const word = fullText.slice(t.start, t.end);
    if (spanPosSet.has(t.pos)) html += `<mark class="span-hit">${escapeHtml(word)}</mark>`;
    else if (stemSet.has(t.stem)) html += `<mark>${escapeHtml(word)}</mark>`;
    else html += escapeHtml(word);
    cursor = t.end;
  });
  html += escapeHtml(fullText.slice(cursor));
  return html;
}

/* =========================================================
   Drawer (click-through detail view)
   ========================================================= */
const drawer = document.getElementById('drawer');
const scrim = document.getElementById('scrim');

function openDrawer(docid, matchedStems, spanPositions, evidenceRows){
  const d = DATA.docs[docid];
  document.getElementById('drawerId').textContent = docid;
  document.getElementById('drawerCat').textContent = d.category;
  document.getElementById('drawerTitle').textContent = d.title;
  const body = document.getElementById('drawerBody');
  let html = `<div class="db-text">${highlightFullText(docid, matchedStems, spanPositions)}</div>`;
  if (evidenceRows && evidenceRows.length){
    html += `<div class="evidence-box"><h4>Matched term positions (evidence from the positional index)</h4>`;
    evidenceRows.forEach(r=>{
      html += `<div class="pos-row"><span class="term">${escapeHtml(r.term)}</span><span>${escapeHtml(r.value)}</span></div>`;
    });
    html += `</div>`;
  }
  body.innerHTML = html;
  drawer.classList.add('open');
  scrim.classList.add('open');
}
function closeDrawer(){
  drawer.classList.remove('open');
  scrim.classList.remove('open');
}
document.getElementById('drawerClose').addEventListener('click', closeDrawer);
scrim.addEventListener('click', closeDrawer);
document.addEventListener('keydown', e=>{ if (e.key==='Escape') closeDrawer(); });

/* =========================================================
   Rendering: free-text results
   ========================================================= */
let lastFreeTextQuery = '';

function renderFilterSummary(){
  const filters = getActiveFilters();
  const parts = [];
  if (filters.category !== 'any') parts.push(`type = <code>${escapeHtml(filters.category)}</code>`);
  if (filters.color !== 'any') parts.push(`colour = <code>${escapeHtml(filters.color)}</code>`);
  if (filters.size !== 'any') parts.push(`size = <code>${escapeHtml(filters.size)}</code>`);
  const el = document.getElementById('filterSummary');
  el.innerHTML = parts.length ? `Filtering to: ${parts.join(', ')}` : '';
}

function renderFreeText(queryText){
  lastFreeTextQuery = queryText;
  const filters = getActiveFilters();
  renderFilterSummary();
  const head = document.getElementById('freeTextResultsHead');
  const box = document.getElementById('freeTextResults');
  const {results, totalMatches, queryStems, unknownTerms} = vsmSearch(queryText, filters);

  let headHtml = `<div class="results-head"><h2>Results for &ldquo;${escapeHtml(queryText)}&rdquo;</h2><span class="count">${totalMatches} match${totalMatches===1?'':'es'}</span></div>`;
  if (unknownTerms.length){
    headHtml += `<p style="font-size:12.5px;color:var(--ink-faint);margin:-4px 0 14px;">Not in the dictionary, ignored in ranking: ${unknownTerms.map(t=>`<code style="font-family:var(--mono);">${escapeHtml(t)}</code>`).join(', ')}</p>`;
  }
  head.innerHTML = headHtml;

  if (results.length===0){
    box.innerHTML = `<div class="empty-state"><strong>No products matched.</strong>Try a broader query, loosen a filter, or check the Index tab to see which terms are in the dictionary.</div>`;
    return;
  }

  box.innerHTML = results.map(r=>{
    const d = DATA.docs[r.docid];
    const snippet = highlightSnippet(r.docid, r.matchedTerms, 150);
    return `<button class="result-card" data-docid="${r.docid}" data-terms='${JSON.stringify(r.matchedTerms)}'>
      <div class="rc-top"><span class="rc-id">${r.docid} · ${escapeHtml(d.category)}</span><span class="rc-score">${fmtScore(r.finalScore)}</span></div>
      <h3>${escapeHtml(d.title)}</h3>
      <p class="rc-snippet">${snippet}</p>
    </button>`;
  }).join('');

  box.querySelectorAll('.result-card').forEach(card=>{
    card.addEventListener('click', ()=>{
      const docid = card.dataset.docid;
      const terms = JSON.parse(card.dataset.terms);
      const evidence = terms.map(t=>{
        const entry = DATA.index[t].postings.find(p=>p.doc===docid);
        return {term:t, value:`tf=${entry.tf}, positions ${entry.pos.join(', ')}`};
      });
      openDrawer(docid, terms, [], evidence);
    });
  });
}

document.getElementById('freeTextForm').addEventListener('submit', e=>{
  e.preventDefault();
  const q = document.getElementById('freeTextInput').value.trim();
  if (q) renderFreeText(q);
});

['filterType','filterColour','filterSize'].forEach(id=>{
  document.getElementById(id).addEventListener('change', ()=>{
    if (lastFreeTextQuery) renderFreeText(lastFreeTextQuery);
    else renderFilterSummary();
  });
});
document.getElementById('filterReset').addEventListener('click', ()=>{
  document.getElementById('filterType').value = 'any';
  document.getElementById('filterColour').value = 'any';
  document.getElementById('filterSize').value = 'any';
  if (lastFreeTextQuery) renderFreeText(lastFreeTextQuery);
  else renderFilterSummary();
});

const FREE_TEXT_EXAMPLES = ['cotton shirt for office wear','denim jeans','festive saree','winter jacket','breathable fabric','regular fit kurta','printed dress','zip closure jacket','stretch leggings','sweatshirt hoodie'];
document.getElementById('freeTextChips').innerHTML = FREE_TEXT_EXAMPLES.map(q=>`<span class="chip">${escapeHtml(q)}</span>`).join('');
document.getElementById('freeTextChips').addEventListener('click', e=>{
  if (e.target.classList.contains('chip')){
    document.getElementById('freeTextInput').value = e.target.textContent;
    renderFreeText(e.target.textContent);
  }
});

/* =========================================================
   Rendering: phrase / proximity results
   ========================================================= */
function renderPositional(raw){
  const parsed = parsePositionalQuery(raw);
  const head = document.getElementById('posResultsHead');
  const box = document.getElementById('posResults');

  let outcome, label, evidenceBuilder;
  if (parsed.type === 'phrase'){
    outcome = phraseSearch(parsed.words);
    label = `exact phrase &ldquo;${escapeHtml(parsed.words.join(' '))}&rdquo;`;
    evidenceBuilder = (docid, r)=>{
      return [{term: parsed.words.join(' '), value:`matched at token position${r.spans.length>1?'s':''} ${r.spans.map(s=>s[0]).join(', ')} (span length ${outcome.stems.length})`}];
    };
  } else {
    outcome = proximitySearch(parsed.termA, parsed.termB, parsed.k);
    label = `&ldquo;${escapeHtml(parsed.termA)}&rdquo; within ${parsed.k} of &ldquo;${escapeHtml(parsed.termB)}&rdquo; (ordered)`;
    evidenceBuilder = (docid, r)=>{
      return [{term: outcome.stems.join(' … '), value:`position pairs ${r.spans.map(s=>`(${s[0]}, ${s[1]})`).join(', ')}`}];
    };
  }

  let headHtml = `<div class="results-head"><h2>Results for ${label}</h2><span class="count">${outcome.totalMatches||0} match${(outcome.totalMatches===1)?'':'es'}</span></div>`;
  if (outcome.missing && outcome.missing.length){
    headHtml += `<p style="font-size:12.5px;color:var(--ink-faint);margin:-4px 0 14px;">Term not found in the dictionary: ${outcome.missing.map(t=>`<code style="font-family:var(--mono);">${escapeHtml(t)}</code>`).join(', ')}</p>`;
  }
  head.innerHTML = headHtml;

  if (!outcome.results || outcome.results.length===0){
    box.innerHTML = `<div class="empty-state"><strong>No positional matches.</strong>The terms may not co-occur within the required order/window, or a term is absent from the corpus.</div>`;
    return;
  }

  box.innerHTML = outcome.results.map(r=>{
    const d = DATA.docs[r.docid];
    const flatSpans = [].concat(...r.spans);
    const snippet = highlightFullTextSnippet(r.docid, flatSpans, 160);
    return `<button class="result-card" data-docid="${r.docid}" data-spans='${JSON.stringify(flatSpans)}' data-stems='${JSON.stringify(outcome.stems)}'>
      <div class="rc-top"><span class="rc-id">${r.docid} · ${escapeHtml(d.category)}</span><span class="rc-score">${r.occurrences} occ.</span></div>
      <h3>${escapeHtml(d.title)}</h3>
      <p class="rc-snippet">${snippet}</p>
      <div class="rc-evidence">positions: ${flatSpans.join(', ')}</div>
    </button>`;
  }).join('');

  box.querySelectorAll('.result-card').forEach((card,i)=>{
    card.addEventListener('click', ()=>{
      const docid = card.dataset.docid;
      const spans = JSON.parse(card.dataset.spans);
      const stems = JSON.parse(card.dataset.stems);
      const r = outcome.results[i];
      openDrawer(docid, stems, spans, evidenceBuilder(docid, r));
    });
  });
}

function highlightFullTextSnippet(docid, spanPositions, maxLen){
  const {fullText, tokens} = buildDocTokens(docid);
  const spanSet = new Set(spanPositions);
  let anchorTok = tokens.find(t=>spanSet.has(t.pos));
  const anchorChar = anchorTok ? anchorTok.start : 0;
  let start = Math.max(0, anchorChar - 60);
  let end = Math.min(fullText.length, start + maxLen);
  if (end - start < maxLen) start = Math.max(0, end - maxLen);
  let html = '';
  let cursor = start;
  tokens.forEach(t=>{
    if (t.end <= start || t.start >= end) return;
    html += escapeHtml(fullText.slice(cursor, t.start));
    const word = fullText.slice(t.start, t.end);
    html += spanSet.has(t.pos) ? `<mark>${escapeHtml(word)}</mark>` : escapeHtml(word);
    cursor = t.end;
  });
  html += escapeHtml(fullText.slice(cursor, end));
  return (start>0?'…':'') + html + (end<fullText.length?'…':'');
}

document.getElementById('posForm').addEventListener('submit', e=>{
  e.preventDefault();
  const q = document.getElementById('posInput').value.trim();
  if (q) renderPositional(q);
});

const POS_EXAMPLES = ['"cotton shirt"','"stretch denim"','"festive wear"','"winter wear"','"regular fit"','cotton WITHIN/3 shirt','stretch WITHIN/5 denim','festive WITHIN/4 kurta'];
document.getElementById('posChips').innerHTML = POS_EXAMPLES.map(q=>`<span class="chip">${escapeHtml(q)}</span>`).join('');
document.getElementById('posChips').addEventListener('click', e=>{
  if (e.target.classList.contains('chip')){
    document.getElementById('posInput').value = e.target.textContent;
    renderPositional(e.target.textContent);
  }
});

/* =========================================================
   Index explorer
   ========================================================= */
function renderIndexLookup(rawTerm){
  const box = document.getElementById('idxResult');
  const stems = preprocess(rawTerm);
  const stem = stems[0];
  if (!stem || !DATA.index[stem]){
    box.innerHTML = `<div class="empty-state"><strong>&ldquo;${escapeHtml(rawTerm)}&rdquo; is not in the dictionary.</strong>It was stemmed to <code style="font-family:var(--mono);">${escapeHtml(stem||'(stopword)')}</code>, which has no postings. Try another term, e.g. cotton, fit, waist, zip.</div>`;
    return;
  }
  const entry = DATA.index[stem];
  let html = `<div class="idx-result-head"><span class="term-big">${escapeHtml(stem)}</span><span class="df">df = ${entry.df} of ${DATA.N} documents</span></div>`;
  if (stem !== rawTerm.toLowerCase().trim()) html += `<p class="idx-stem-note">&ldquo;${escapeHtml(rawTerm)}&rdquo; was stemmed to &ldquo;${escapeHtml(stem)}&rdquo; before lookup, matching the pipeline used to build the index.</p>`;
  html += `<table class="postings"><thead><tr><th>docID</th><th>category</th><th>tf</th><th>positions</th></tr></thead><tbody>`;
  entry.postings.forEach(p=>{
    const d = DATA.docs[p.doc];
    html += `<tr><td>${p.doc}</td><td>${escapeHtml(d.category)}</td><td>${p.tf}</td><td>[${p.pos.join(', ')}]</td></tr>`;
  });
  html += `</tbody></table>`;
  box.innerHTML = html;
}

document.getElementById('idxForm').addEventListener('submit', e=>{
  e.preventDefault();
  const q = document.getElementById('idxInput').value.trim();
  if (q) renderIndexLookup(q);
});
const IDX_EXAMPLES = ['cotton','fit','waist','zip','festive','denim'];
document.getElementById('idxChips').innerHTML = IDX_EXAMPLES.map(q=>`<span class="chip">${escapeHtml(q)}</span>`).join('');
document.getElementById('idxChips').addEventListener('click', e=>{
  if (e.target.classList.contains('chip')){
    document.getElementById('idxInput').value = e.target.textContent;
    renderIndexLookup(e.target.textContent);
  }
});

/* run an initial demo query so the page isn't empty on load */
renderFreeText('cotton shirt for office wear');
