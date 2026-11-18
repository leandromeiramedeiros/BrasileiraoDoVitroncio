// Validação da simulação de temporada: em temporadas passadas, a partir de
// vários pontos do campeonato, simula o restante com o que se sabia naquele
// momento e confere se as chances de título / G4 / Z4 eram bem calibradas.
// Serve para escolher o multiplicador da incerteza de força (kappa).
//   node tools/season-backtest.mjs
import { readFile } from 'node:fs/promises';
import { loadHistory } from './lib/history.mjs';
import * as R from '../site/js/engine/ratings.js';
import { simulateSeason, buildTable } from '../site/js/engine/season.js';

const model = JSON.parse(await readFile(new URL('../site/data/model.json', import.meta.url), 'utf8'));
const { params, xg, stack, hazard } = model;
const { matches } = await loadHistory();
const done = matches.filter((m) => m.status === 'post');
const KAPPAS = [0, 0.6, 1, 1.4, 1.8, 2.4];
const CHECK = [0, 95, 190, 285, 340]; // jogos disputados na temporada no momento da simulação
const seasons = [...new Set(done.filter((m) => m.lg === 'A' && m.season >= 2013 && m.season <= 2025).map((m) => m.season))];
const bySeason = Object.fromEntries(seasons.map((s) => [s, done.filter((m) => m.lg === 'A' && m.season === s)]));
const st = R.createState(params);
const played = {};
const out = {}; // kappa -> checkpoint -> acumuladores
for (const k of KAPPAS) { out[k] = {}; for (const c of CHECK) out[k][c] = { n: 0, llT: 0, llZ: 0, llG: 0, brZ: 0, cover: 0, nT: 0, bins: Array.from({ length: 5 }, () => ({ n: 0, p: 0, y: 0 })) }; }

function evaluate(season, c) {
  const all = bySeason[season]; if (all.length < 370) return;
  const ids = [...new Set(all.flatMap((m) => [m.h, m.a]))];
  const fin = all.slice(0, c), rest = all.slice(c);
  const table = buildTable(ids, fin);
  const final = buildTable(ids, all);
  const rank = Object.values(final).sort((x, y) => y.pts - x.pts || y.v - x.v || (y.gp - y.gc) - (x.gp - x.gc) || y.gp - x.gp);
  const pos = Object.fromEntries(rank.map((r, i) => [r.id, i + 1]));
  const tmp = structuredClone(st);
  const fx = rest.map((m) => { const e = R.expectedMatch({ params, stack }, tmp, m.h, m.a, 'A', season); return { id: m.id, h: m.h, a: m.a, lh: e.lh, la: e.la }; });
  for (const id of ids) R.prepareTeam(tmp, params, id, 'A', season);
  const tm = ids.map((id) => tmp.teams[id]);
  const sigma0 = Math.sqrt(tm.reduce((s, t) => s + t.ch.g.Po + t.ch.g.Pd, 0) / (2 * tm.length));
  for (const k of KAPPAS) {
    const res = simulateSeason(hazard, ids, table, fx, { n: 4000, sigma: sigma0 * k, seed: season * 100 + c });
    const A = out[k][c]; const eps = 1 / 8000;
    for (const id of ids) {
      const t = res.teams[id], p = pos[id];
      const cl = (v) => Math.min(1 - eps, Math.max(eps, v));
      const yZ = p >= 17 ? 1 : 0, yG = p <= 4 ? 1 : 0;
      A.llZ -= yZ ? Math.log(cl(t.rebaixamento)) : Math.log(1 - cl(t.rebaixamento));
      A.llG -= yG ? Math.log(cl(t.libertadores)) : Math.log(1 - cl(t.libertadores));
      A.brZ += (t.rebaixamento - yZ) ** 2;
      if (p === 1) { A.llT -= Math.log(cl(t.titulo)); A.nT++; }
      if (final[id].pts >= t.ptsLo && final[id].pts <= t.ptsHi) A.cover++;
      for (const [pp, y] of [[t.rebaixamento, yZ], [t.libertadores, yG], [t.titulo, p === 1 ? 1 : 0]]) { const b = A.bins[Math.min(4, Math.floor(pp * 5))]; b.n++; b.p += pp; b.y += y; }
      A.n++;
    }
  }
}

for (const m of done) {
  if (m.lg === 'A' && bySeason[m.season]) {
    const c = (played[m.season] ||= 0);
    if (CHECK.includes(c) && bySeason[m.season][c]?.id === m.id) evaluate(m.season, c);
    played[m.season]++;
  }
  R.applyMatch(st, params, m, xg);
}
console.log('kappa | ponto | logloss título | logloss Z4 (por time) | logloss G4 | cobertura do intervalo 10–90% dos pontos (alvo 0,80)');
const total = {};
for (const k of KAPPAS) {
  total[k] = { t: 0, z: 0, g: 0 };
  for (const c of CHECK) { const A = out[k][c]; if (!A.n) continue; total[k].t += A.llT / A.nT; total[k].z += A.llZ / A.n; total[k].g += A.llG / A.n; console.log(`${String(k).padStart(4)} | ${String(c).padStart(3)} | ${(A.llT / A.nT).toFixed(3)} | ${(A.llZ / A.n).toFixed(4)} | ${(A.llG / A.n).toFixed(4)} | ${(A.cover / A.n).toFixed(3)}`); }
}
console.log('\nresumo (média dos pontos de verificação):');
for (const k of KAPPAS) console.log(`kappa ${k}: título ${(total[k].t / CHECK.length).toFixed(3)}  Z4 ${(total[k].z / CHECK.length).toFixed(4)}  G4 ${(total[k].g / CHECK.length).toFixed(4)}`);
console.log('\ncalibração agregada (kappa=1): faixa prevista -> observado');
const agg = Array.from({ length: 5 }, () => ({ n: 0, p: 0, y: 0 }));
for (const c of CHECK) out[1][c].bins.forEach((b, i) => { agg[i].n += b.n; agg[i].p += b.p; agg[i].y += b.y; });
agg.forEach((b, i) => b.n && console.log(`  ${i * 20}-${i * 20 + 20}%: n=${b.n} previsto ${(b.p / b.n).toFixed(3)} observado ${(b.y / b.n).toFixed(3)}`));
