// Dumps the production Markov chain (scoreDistribution) on a log-lambda grid -> work/chain_grid.json
// and, with a CSV argument (id,lh,la), writes exact probabilities: node grid_chain.mjs in.csv out.csv
import { readFileSync, writeFileSync } from 'node:fs';
import { scoreDistribution } from './v1_match_model.mjs';   // frozen copy of site/js/engine/match-model.js (git HEAD)
const hz = JSON.parse(readFileSync(new URL('./v1_snapshot.json', import.meta.url), 'utf8')).hazard;
if (process.argv[2]) {
  const rows = readFileSync(process.argv[2], 'utf8').trim().split('\n').slice(1).map((l) => l.split(','));
  const out = ['id,pH,pD,pA'];
  for (const [id, lh, la] of rows) { const d = scoreDistribution(hz, +lh, +la); out.push([id, d.pH.toFixed(6), d.pD.toFixed(6), d.pA.toFixed(6)].join(',')); }
  writeFileSync(process.argv[3], out.join('\n') + '\n');
} else {
  const N = 121, lo = Math.log(0.12), hi = Math.log(6.0);
  const g = Array.from({ length: N }, (_, i) => lo + ((hi - lo) * i) / (N - 1));
  const pH = [], pD = [], pA = [], eh = [], ea = [];
  for (const a of g) { const rH = [], rD = [], rA = [], rh = [], ra = []; for (const b of g) { const d = scoreDistribution(hz, Math.exp(a), Math.exp(b)); rH.push(d.pH); rD.push(d.pD); rA.push(d.pA); rh.push(d.xgH); ra.push(d.xgA); } pH.push(rH); pD.push(rD); pA.push(rA); eh.push(rh); ea.push(ra); }
  writeFileSync(new URL('./work/chain_grid.json', import.meta.url), JSON.stringify({ g, pH, pD, pA, eh, ea }));
  console.log('grid', N);
}
