// (lh, la) -> 1X2 probabilities with the v1 Markov-chain match model.
//   node code/probs.mjs in.csv out.csv [hazard.json]
// in.csv: id,lh,la   out.csv: id,pH,pD,pA
import { readFileSync, writeFileSync } from 'node:fs';
import { scoreDistribution } from '/home/user/brasileiraodovitroncio/site/js/engine/match-model.js';
const [inp, out, hzPath] = process.argv.slice(2);
const hz = JSON.parse(readFileSync(hzPath || new URL('../meta/hazard.json', import.meta.url), 'utf8'));
const lines = readFileSync(inp, 'utf8').trim().split(/\r?\n/);
const head = lines[0].split(','); const iI = head.indexOf('id'), iH = head.indexOf('lh'), iA = head.indexOf('la');
const res = ['id,pH,pD,pA'];
for (const l of lines.slice(1)) {
  const r = l.split(','); const d = scoreDistribution(hz, +r[iH], +r[iA]);
  res.push([r[iI], d.pH.toFixed(6), d.pD.toFixed(6), d.pA.toFixed(6)].join(','));
}
writeFileSync(out, res.join('\n') + '\n');
