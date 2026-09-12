const fs = require('fs');
const { preprocessWithOffsets } = require('./pipeline.js');

const data = JSON.parse(fs.readFileSync('./index_data.json', 'utf8'));

let mismatches = 0;
let checked = 0;

for (const docid of Object.keys(data.docs)) {
  const d = data.docs[docid];
  const fullText = d.title + '. ' + d.text;
  const toks = preprocessWithOffsets(fullText);
  const jsStems = toks.map(t => t.stem);

  // Reconstruct expected stems+positions for this doc from the inverted index
  const expected = []; // pos -> stem
  for (const term of Object.keys(data.index)) {
    for (const p of data.index[term].postings) {
      if (p.doc === docid) {
        for (const pos of p.pos) expected[pos] = term;
      }
    }
  }

  checked++;
  if (expected.length !== jsStems.length) {
    console.log(`LEN MISMATCH ${docid}: expected ${expected.length} vs js ${jsStems.length}`);
    mismatches++;
    continue;
  }
  for (let i = 0; i < jsStems.length; i++) {
    if (expected[i] !== jsStems[i]) {
      console.log(`MISMATCH ${docid} pos ${i}: expected=${expected[i]} js=${jsStems[i]}`);
      mismatches++;
    }
  }
}

console.log(`Checked ${checked} docs. Mismatches: ${mismatches}`);
