// Dumps the v1 pre-match team state (goal channel o/d/g + shots channels) for every match, walk-forward.
// Read-only use of the v1 engine; mirrors shared/export.mjs. Output: work/team_state.csv
import { readFileSync, writeFileSync } from 'node:fs';
import { loadHistory } from '/home/user/brasileiraodovitroncio/tools/lib/history.mjs';
import * as R from '/home/user/brasileiraodovitroncio/site/js/engine/ratings.js';
const { matches } = await loadHistory();
const model = JSON.parse(readFileSync('/home/user/brasileiraodovitroncio/site/data/model.json', 'utf8'));
const { params, xg } = model;
const st = R.createState(params);
const out = ['id,oh,dh,gh,Poh,Pdh,oa,da,Poa,Pda,sot_oh,sot_dh,sot_oa,sot_da,sh_oh,sh_dh,sh_oa,sh_da'];
const f = (v) => +v.toFixed(5);
for (const m of matches) {
  if (m.status === 'off') continue;
  const H = R.prepareTeam(st, params, m.h, m.lg, m.season), A = R.prepareTeam(st, params, m.a, m.lg, m.season);
  out.push([m.id, f(H.ch.g.o), f(H.ch.g.d), f(H.ch.g.g), f(H.ch.g.Po), f(H.ch.g.Pd), f(A.ch.g.o), f(A.ch.g.d), f(A.ch.g.Po), f(A.ch.g.Pd),
    f(H.ch.sot.o), f(H.ch.sot.d), f(A.ch.sot.o), f(A.ch.sot.d), f(H.ch.sh.o), f(H.ch.sh.d), f(A.ch.sh.o), f(A.ch.sh.d)].join(','));
  if (m.status === 'post') R.applyMatch(st, params, m, xg);
}
writeFileSync(new URL('./work/team_state.csv', import.meta.url), out.join('\n'));
console.log('rows', out.length - 1);
