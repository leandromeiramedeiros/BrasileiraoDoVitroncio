// Maps expected goals (lh, la) to 1X2 probabilities with the v1 minute-by-minute hazard model.
// Usage: node lam2prob.mjs in.csv out.csv   (in: id,lh,la ; out: id,pH,pD,pA)
import { readFileSync, writeFileSync } from 'node:fs';
import { scoreDistribution } from '/home/user/brasileiraodovitroncio/site/js/engine/match-model.js';
const hz = JSON.parse(readFileSync('/home/user/brasileiraodovitroncio/site/data/model.json', 'utf8')).hazard;
const rows = readFileSync(process.argv[2], 'utf8').trim().split('\n').slice(1).map((l) => l.split(','));
const out = ['id,pH,pD,pA'];
for (const [id, lh, la] of rows) { const d = scoreDistribution(hz, +lh, +la); out.push([id, d.pH.toFixed(6), d.pD.toFixed(6), d.pA.toFixed(6)].join(',')); }
writeFileSync(process.argv[3], out.join('\n'));
