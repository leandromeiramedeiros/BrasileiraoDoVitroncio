// ─────────────────────────────────────────────────────────────────────────────
// Simulador de temporada — Monte Carlo sobre os jogos restantes.
//
// Cada jogo restante tem sua matriz exata de placares (match-model.js). Para
// refletir a incerteza sobre a força real de cada time, cada temporada
// simulada sorteia um "desvio de forma" por time (5 níveis, normal
// discretizada); a matriz usada em cada jogo depende da diferença de níveis
// (9 matrizes pré-calculadas por jogo). O resto é sortear placares e montar a
// tabela com os critérios de desempate da CBF (pontos, vitórias, saldo, gols
// pró; persistindo, sorteio).
// ─────────────────────────────────────────────────────────────────────────────
import { scoreDistribution } from './match-model.js?v=2a005902ef';
import { makeRng } from './rng.js?v=2a005902ef';

const LEVELS = [-2, -1, 0, 1, 2];
const LEVEL_CDF = [0.0668, 0.3085, 0.6915, 0.9332, 1];
const CAP = 8; // gols por time na amostragem (0..7 + cauda agregada)

export const ZONES = { titulo: [1, 1], libertadores: [1, 4], preLibertadores: [5, 6], sulAmericana: [7, 12], rebaixamento: [17, 20] };

/** Tabela atual a partir de partidas encerradas. */
export function buildTable(teamIds, finished) {
  const T = {};
  for (const id of teamIds) T[id] = { id, j: 0, v: 0, e: 0, d: 0, gp: 0, gc: 0, pts: 0 };
  for (const m of finished) {
    const h = T[m.h], a = T[m.a];
    if (!h || !a || m.hg == null) continue;
    h.j++; a.j++; h.gp += m.hg; h.gc += m.ag; a.gp += m.ag; a.gc += m.hg;
    if (m.hg > m.ag) { h.v++; a.d++; h.pts += 3; } else if (m.hg < m.ag) { a.v++; h.d++; a.pts += 3; } else { h.e++; a.e++; h.pts++; a.pts++; }
  }
  return T;
}

export function sortTable(rows) {
  return rows.slice().sort((x, y) => y.pts - x.pts || y.v - x.v || (y.gp - y.gc) - (x.gp - x.gc) || y.gp - x.gp || (x.name || '').localeCompare(y.name || ''));
}

/**
 * @param hz        model.hazard
 * @param teamIds   ids dos 20 times
 * @param table     tabela atual (buildTable)
 * @param fixtures  jogos restantes [{id,h,a,lh,la, live?: estado ao vivo}]
 * @param opts      {n, sigma, seed, onProgress}
 */
export function simulateSeason(hz, teamIds, table, fixtures, { n = 20000, sigma = 0.12, seed = 12345, fixed = {} } = {}) {
  const rng = makeRng(seed);
  const T = teamIds.length;
  const pos = new Map(teamIds.map((id, i) => [id, i]));
  // CDFs por jogo e por diferença de nível (−4..4)
  const fx = fixtures.filter((f) => pos.has(f.h) && pos.has(f.a)).map((f) => {
    const cdfs = [];
    const fixedScore = fixed[f.id];
    for (let k = -4; k <= 4; k++) {
      if (fixedScore) { cdfs.push(null); continue; }
      const sh = Math.exp(k * sigma);
      const d = scoreDistribution(hz, f.lh * sh, f.la / sh, f.live || null);
      const cdf = new Float64Array(CAP * CAP); let acc = 0;
      for (let i = 0; i < CAP; i++) for (let j = 0; j < CAP; j++) {
        let p = 0;
        // cauda (8+ gols) agregada na última casa
        for (let ii = i; ii <= (i === CAP - 1 ? 10 : i); ii++) for (let jj = j; jj <= (j === CAP - 1 ? 10 : j); jj++) p += d.matrix[ii][jj];
        acc += p; cdf[i * CAP + j] = acc;
      }
      for (let q = 0; q < cdf.length; q++) cdf[q] /= acc;
      cdfs.push(cdf);
    }
    return { h: pos.get(f.h), a: pos.get(f.a), cdfs, fixedScore };
  });
  const base = teamIds.map((id) => table[id] || { pts: 0, v: 0, gp: 0, gc: 0 });
  const posCount = Array.from({ length: T }, () => new Uint32Array(T));
  const ptsSum = new Float64Array(T), ptsSq = new Float64Array(T);
  const ptsHist = Array.from({ length: T }, () => new Uint32Array(115));
  const pts = new Int32Array(T), wins = new Int32Array(T), gf = new Int32Array(T), ga = new Int32Array(T), lvl = new Int8Array(T);
  const order = new Int32Array(T), tie = new Float64Array(T);
  const champPts = new Uint32Array(115), z4Pts = new Uint32Array(115); // pontuação do campeão e do 16º (corte do rebaixamento)
  for (let s = 0; s < n; s++) {
    for (let i = 0; i < T; i++) {
      pts[i] = base[i].pts; wins[i] = base[i].v; gf[i] = base[i].gp; ga[i] = base[i].gc;
      const u = rng(); lvl[i] = u < LEVEL_CDF[0] ? -2 : u < LEVEL_CDF[1] ? -1 : u < LEVEL_CDF[2] ? 0 : u < LEVEL_CDF[3] ? 1 : 2;
      order[i] = i; tie[i] = rng();
    }
    for (let f = 0; f < fx.length; f++) {
      const m = fx[f];
      let hg, ag;
      if (m.fixedScore) { hg = m.fixedScore[0]; ag = m.fixedScore[1]; }
      else {
        const cdf = m.cdfs[lvl[m.h] - lvl[m.a] + 4];
        const u = rng(); let lo = 0, hi = cdf.length - 1;
        while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < u) lo = mid + 1; else hi = mid; }
        hg = (lo / CAP) | 0; ag = lo % CAP;
      }
      gf[m.h] += hg; ga[m.h] += ag; gf[m.a] += ag; ga[m.a] += hg;
      if (hg > ag) { pts[m.h] += 3; wins[m.h]++; } else if (hg < ag) { pts[m.a] += 3; wins[m.a]++; } else { pts[m.h]++; pts[m.a]++; }
    }
    const arr = Array.from(order).sort((x, y) => pts[y] - pts[x] || wins[y] - wins[x] || (gf[y] - ga[y]) - (gf[x] - ga[x]) || gf[y] - gf[x] || tie[x] - tie[y]);
    for (let r = 0; r < T; r++) {
      const i = arr[r]; posCount[i][r]++;
      ptsSum[i] += pts[i]; ptsSq[i] += pts[i] * pts[i]; ptsHist[i][Math.min(114, pts[i])]++;
    }
    champPts[Math.min(114, pts[arr[0]])]++;
    if (T >= 17) z4Pts[Math.min(114, pts[arr[T - 5]])]++;
  }
  const zoneP = (i, [a, b]) => { let s = 0; for (let r = a - 1; r <= b - 1 && r < T; r++) s += posCount[i][r]; return s / n; };
  const quant = (hist, q) => { let acc = 0; for (let k = 0; k < hist.length; k++) { acc += hist[k]; if (acc >= q * n) return k; } return hist.length - 1; };
  const teams = {};
  teamIds.forEach((id, i) => {
    const mean = ptsSum[i] / n;
    let ep = 0; for (let r = 0; r < T; r++) ep += (r + 1) * posCount[i][r];
    teams[id] = {
      id, pos: Array.from(posCount[i], (c) => c / n), posMedia: ep / n,
      pts: mean, ptsSd: Math.sqrt(Math.max(0, ptsSq[i] / n - mean * mean)), ptsLo: quant(ptsHist[i], 0.1), ptsHi: quant(ptsHist[i], 0.9),
      titulo: zoneP(i, ZONES.titulo), libertadores: zoneP(i, ZONES.libertadores), g6: zoneP(i, [1, 6]), sulAmericana: zoneP(i, ZONES.sulAmericana), rebaixamento: zoneP(i, ZONES.rebaixamento),
    };
  });
  return { n, teams, corteTitulo: quant(champPts, 0.5), corteRebaixamento: quant(z4Pts, 0.5), jogos: fx.length };
}
