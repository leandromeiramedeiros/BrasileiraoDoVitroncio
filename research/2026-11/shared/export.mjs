// Exporta a base e as previsões do modelo atual (v1) para a bancada de pesquisa.
import { readFileSync, writeFileSync } from 'node:fs';
import { loadHistory } from '/home/user/brasileiraodovitroncio/tools/lib/history.mjs';
import * as R from '/home/user/brasileiraodovitroncio/site/js/engine/ratings.js';
import { scoreDistribution } from '/home/user/brasileiraodovitroncio/site/js/engine/match-model.js';
const { matches, teams } = await loadHistory();
const model = JSON.parse(readFileSync('/home/user/brasileiraodovitroncio/site/data/model.json', 'utf8'));
const { params, xg, stack, hazard } = model;
writeFileSync('shared/matches.json', JSON.stringify(matches));
writeFileSync('shared/teams.json', JSON.stringify(teams));
// odds de mercado (apenas régua)
const map = { 'Flamengo RJ': 'Flamengo', 'Sport Recife': 'Sport', 'Sao Paulo': 'São Paulo', Gremio: 'Grêmio', 'Botafogo RJ': 'Botafogo', Ceara: 'Ceará', Goias: 'Goiás', 'Atletico GO': 'Atlético-GO', 'America MG': 'América-MG', Avai: 'Avaí', Cuiaba: 'Cuiabá', Vitoria: 'Vitória', 'Chapecoense-SC': 'Chapecoense', Nautico: 'Náutico', Parana: 'Paraná', Criciuma: 'Criciúma', 'Atletico-MG': 'Atlético-MG' };
const nameToId = {}; for (const [id, t] of Object.entries(teams)) { nameToId[t.name] = id; nameToId[t.full] = id; }
const odds = new Map();
for (const r of readFileSync('/home/user/brasileiraodovitroncio/model/cache/BRA.csv', 'utf8').replace(/^﻿/, '').split(/\r?\n/).slice(1).filter(Boolean).map((l) => l.split(','))) {
  const hi = nameToId[map[r[5]] || r[5]], ai = nameToId[map[r[6]] || r[6]]; if (!hi || !ai) continue;
  const o = [parseFloat(r[10]) || parseFloat(r[16]), parseFloat(r[11]) || parseFloat(r[17]), parseFloat(r[12]) || parseFloat(r[18])]; if (!(o[0] && o[1] && o[2])) continue;
  const inv = o.map((v) => 1 / v), s = inv[0] + inv[1] + inv[2]; odds.set(`${r[2]}|${hi}|${ai}`, inv.map((v) => v / s));
}
const st = R.createState(params);
const feat = ['id,lg,season,date,h,a,status,hg,ag,lh,la,g_lh,g_la,g_Sh,g_Sa,sh_lh,sh_la,sot_lh,sot_la,co_lh,co_la,yc_lh,yc_la,fo_lh,fo_la,nh,na,mu,hfa'];
const preds = ['id,pH,pD,pA'], od = ['id,mH,mD,mA'];
for (const m of matches) {
  if (m.status === 'off') continue;
  R.prepareTeam(st, params, m.h, m.lg, m.season); R.prepareTeam(st, params, m.a, m.lg, m.season);
  const pr = {}; for (const c of R.CHANNELS) pr[c] = R.predictChannel(st, params, c, m.h, m.a, m.lg);
  const { lh, la } = R.stackLambda(pr, st.lg[m.lg], stack);
  const f = (v) => (v == null ? '' : +v.toFixed(5));
  feat.push([m.id, m.lg, m.season, m.date, m.h, m.a, m.status, m.hg ?? '', m.ag ?? '', f(lh), f(la), f(pr.g.lh), f(pr.g.la), f(pr.g.Sh), f(pr.g.Sa), f(pr.sh.lh), f(pr.sh.la), f(pr.sot.lh), f(pr.sot.la), f(pr.co.lh), f(pr.co.la), f(pr.yc.lh), f(pr.yc.la), f(pr.fo.lh), f(pr.fo.la), st.teams[m.h].n, st.teams[m.a].n, f(st.lg[m.lg].g.mu), f(st.lg[m.lg].g.hfa)].join(','));
  if (m.lg === 'A' && m.season >= 2013 && m.status === 'post') {
    const d = scoreDistribution(hazard, lh, la); preds.push([m.id, d.pH.toFixed(6), d.pD.toFixed(6), d.pA.toFixed(6)].join(','));
    const o = odds.get(`${m.season}|${m.h}|${m.a}`); if (o) od.push([m.id, ...o.map((v) => v.toFixed(6))].join(','));
  }
  if (m.status === 'post') R.applyMatch(st, params, m, xg);
}
writeFileSync('shared/baseline_features.csv', feat.join('\n'));
writeFileSync('shared/preds_baseline.csv', preds.join('\n'));
writeFileSync('shared/odds_benchmark.csv', od.join('\n'));
console.log('matches', matches.length, 'feature rows', feat.length - 1, 'preds', preds.length - 1, 'odds', od.length - 1);
