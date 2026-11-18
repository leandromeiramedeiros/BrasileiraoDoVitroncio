// ─────────────────────────────────────────────────────────────────────────────
// Modelo de partida do Vitrôncio — cadeia de Markov não homogênea, minuto a
// minuto, resolvida de forma EXATA por programação dinâmica (sem ruído de
// Monte Carlo).
//
// Estado: (gols mandante, gols visitante, vermelhos mandante, vermelhos visitante)
// A cada minuto, a taxa de gol de cada time é
//
//   h(t) = λ_time/90 · exp( β_faixa(t) + β_placar(saldo) + β_homens(vantagem) )
//
// onde λ_time vem dos ratings (pré-jogo) e os β são estimados no reajuste
// semanal a partir de ~15 mil partidas com eventos minuto a minuto: gols saem
// mais no fim, quem está perdendo ataca mais, quem tem um a mais marca mais.
// O mesmo modelo serve para o pré-jogo (parte do minuto 0, 0–0) e para o ao
// vivo (parte do minuto e do placar atuais).
// ─────────────────────────────────────────────────────────────────────────────

export const MAXG = 10; // gols por time considerados (0..10)
const G1 = MAXG + 1;

export function diffIdx(d) { return d <= -2 ? 0 : d >= 2 ? 4 : d + 2; }
export function menIdx(own, opp) { return opp > own ? 2 : own > opp ? 0 : 1; } // 2 = tem um a mais

/** Linha do tempo: lista de passos de 1 minuto com a faixa de cada um. */
export function timeline(hz) {
  const steps = [];
  const push = (bin, label, half) => steps.push({ bin, label, half });
  for (let m = 1; m <= 45; m++) push(m <= 15 ? 0 : m <= 30 ? 1 : 2, `${m}'`, 1);
  for (let x = 1; x <= hz.S1; x++) push(3, `45'+${x}'`, 1);
  for (let m = 46; m <= 90; m++) push(m <= 60 ? 4 : m <= 75 ? 5 : 6, `${m}'`, 2);
  for (let x = 1; x <= hz.S2; x++) push(7, `90'+${x}'`, 2);
  return steps;
}

/** Índice do passo para um evento (minuto, acréscimo). */
export function stepOf(hz, min, add = 0, half = null) {
  const h = half ?? (min > 45 ? 2 : 1);
  if (h === 1) {
    if (min < 45 || (min === 45 && !add)) return Math.max(0, Math.min(44, min - 1));
    return 45 + Math.max(0, Math.min(hz.S1 - 1, (add || min - 45) - 1));
  }
  if (min < 90 || (min === 90 && !add)) return hz.S1 + Math.max(45, Math.min(89, min - 1));
  return 90 + hz.S1 + Math.max(0, Math.min(hz.S2 - 1, (add || min - 90) - 1));
}

/** Tabelas de multiplicadores por faixa/estado (pré-calculadas uma vez). */
export function compile(hz) {
  if (hz._c) return hz._c;
  const steps = timeline(hz);
  const goalF = [], redH = [], redA = [];
  for (let b = 0; b < 8; b++) {
    goalF[b] = []; redH[b] = []; redA[b] = [];
    for (let d = 0; d < 5; d++) {
      goalF[b][d] = [];
      for (let m = 0; m < 3; m++) {
        let e = hz.goal.bin[b] + hz.goal.diff[d] + hz.goal.men[m];
        if (b >= 6 && d < 2) e += hz.goal.lateTrail || 0;
        goalF[b][d][m] = (Math.exp(e) / 90) * (hz.goal.norm || 1);
      }
      redH[b][d] = Math.exp(hz.red.bin[b] + hz.red.diff[d]);
      redA[b][d] = Math.exp(hz.red.bin[b] + hz.red.diff[d] + hz.red.away);
    }
  }
  const c = { steps, goalF, redH, redA, n: steps.length };
  Object.defineProperty(hz, '_c', { value: c, enumerable: false });
  return c;
}

/**
 * Distribuição exata do placar final.
 * @param hz     modelo de intensidades (model.hazard)
 * @param lh,la  gols esperados pré-jogo de mandante e visitante (90 min, 11x11)
 * @param start  estado atual {step, hg, ag, rh, ra}; omitido = pré-jogo
 * @returns {matrix, pH, pD, pA, xgH, xgA, pRed, ...}
 */
export function scoreDistribution(hz, lh, la, start = null) {
  const c = compile(hz);
  const s0 = start || { step: 0, hg: 0, ag: 0, rh: 0, ra: 0 };
  const N = G1 * G1 * 9;
  let cur = new Float64Array(N), nxt = new Float64Array(N);
  const idx = (hg, ag, rh, ra) => ((hg * G1 + ag) * 3 + rh) * 3 + ra;
  cur[idx(Math.min(MAXG, s0.hg), Math.min(MAXG, s0.ag), Math.min(2, s0.rh || 0), Math.min(2, s0.ra || 0))] = 1;
  let maxH = Math.min(MAXG, s0.hg), maxA = Math.min(MAXG, s0.ag);
  for (let t = Math.max(0, s0.step | 0); t < c.n; t++) {
    const b = c.steps[t].bin;
    nxt.fill(0);
    const gF = c.goalF[b], rHt = c.redH[b], rAt = c.redA[b];
    for (let hg = 0; hg <= maxH; hg++) for (let ag = 0; ag <= maxA; ag++) {
      const dH = diffIdx(hg - ag), dA = diffIdx(ag - hg);
      for (let rh = 0; rh < 3; rh++) for (let ra = 0; ra < 3; ra++) {
        const i = idx(hg, ag, rh, ra), p = cur[i];
        if (p < 1e-15) continue;
        const pgh = hg < MAXG ? lh * gF[dH][menIdx(rh, ra)] : 0;
        const pga = ag < MAXG ? la * gF[dA][menIdx(ra, rh)] : 0;
        const prh = rh < 2 ? rHt[dH] : 0, pra = ra < 2 ? rAt[dA] : 0;
        const stay = 1 - pgh - pga - prh - pra;
        nxt[i] += p * stay;
        if (pgh) nxt[idx(hg + 1, ag, rh, ra)] += p * pgh;
        if (pga) nxt[idx(hg, ag + 1, rh, ra)] += p * pga;
        if (prh) nxt[idx(hg, ag, rh + 1, ra)] += p * prh;
        if (pra) nxt[idx(hg, ag, rh, ra + 1)] += p * pra;
      }
    }
    if (maxH < MAXG) maxH++;
    if (maxA < MAXG) maxA++;
    const tmp = cur; cur = nxt; nxt = tmp;
  }
  const matrix = Array.from({ length: G1 }, () => new Array(G1).fill(0));
  let pH = 0, pD = 0, pA = 0, xgH = 0, xgA = 0, pRedH = 0, pRedA = 0;
  for (let hg = 0; hg <= MAXG; hg++) for (let ag = 0; ag <= MAXG; ag++) {
    let s = 0;
    for (let rh = 0; rh < 3; rh++) for (let ra = 0; ra < 3; ra++) {
      const p = cur[idx(hg, ag, rh, ra)];
      s += p;
      if (rh > (s0.rh || 0)) pRedH += p;
      if (ra > (s0.ra || 0)) pRedA += p;
    }
    matrix[hg][ag] = s;
    if (hg > ag) pH += s; else if (hg === ag) pD += s; else pA += s;
    xgH += hg * s; xgA += ag * s;
  }
  return { matrix, pH, pD, pA, xgH, xgA, pRedH, pRedA };
}

/**
 * Caminho inverso: dadas as probabilidades-alvo de vitória do mandante e do
 * visitante (por exemplo, vindas de um conjunto de modelos), encontra os gols
 * esperados (lh, la) com que a cadeia reproduz essas probabilidades. Assim
 * qualquer previsão 1X2 vira uma distribuição completa de placares, coerente
 * com o ao vivo. `total` (opcional) fixa a soma lh+la e ajusta só o equilíbrio
 * entre os lados (nesse caso casa pH/(pH+pA) em vez dos dois alvos).
 */
export function lambdasFromProbs(hz, pH, pA, { total = null, init = null } = {}) {
  const f = (x) => { const d = scoreDistribution(hz, Math.exp(x[0]), Math.exp(x[1])); return [d.pH, d.pA]; };
  if (total) {
    // busca 1-D na fração do mandante
    const target = pH / (pH + pA);
    let lo = 0.03, hi = 0.97;
    for (let i = 0; i < 40; i++) { const mid = (lo + hi) / 2; const d = scoreDistribution(hz, total * mid, total * (1 - mid)); if (d.pH / (d.pH + d.pA) < target) lo = mid; else hi = mid; }
    const s = (lo + hi) / 2;
    return { lh: total * s, la: total * (1 - s) };
  }
  let x = init ? [Math.log(init[0]), Math.log(init[1])] : [Math.log(1.2 + 1.2 * (pH - pA)), Math.log(1.2 - 1.2 * (pH - pA) + 0.05)];
  for (let it = 0; it < 30; it++) {
    const y = f(x), r = [y[0] - pH, y[1] - pA];
    if (Math.abs(r[0]) + Math.abs(r[1]) < 1e-7) break;
    const h = 1e-4, y0 = f([x[0] + h, x[1]]), y1 = f([x[0], x[1] + h]);
    const J = [[(y0[0] - y[0]) / h, (y1[0] - y[0]) / h], [(y0[1] - y[1]) / h, (y1[1] - y[1]) / h]];
    const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    if (Math.abs(det) < 1e-12) break;
    let dx = [(r[0] * J[1][1] - r[1] * J[0][1]) / det, (J[0][0] * r[1] - J[1][0] * r[0]) / det];
    const mx = Math.max(Math.abs(dx[0]), Math.abs(dx[1])); if (mx > 0.7) dx = dx.map((v) => (v * 0.7) / mx);
    x = [Math.min(2.2, Math.max(-3.5, x[0] - dx[0])), Math.min(2.2, Math.max(-3.5, x[1] - dx[1]))];
  }
  return { lh: Math.exp(x[0]), la: Math.exp(x[1]) };
}

/** Probabilidade de quem marca o PRÓXIMO gol (ou de não sair mais gol). */
export function nextGoal(hz, lh, la, start = null) {
  const c = compile(hz);
  const s0 = start || { step: 0, hg: 0, ag: 0, rh: 0, ra: 0 };
  const dH = diffIdx(s0.hg - s0.ag), dA = diffIdx(s0.ag - s0.hg);
  let cur = new Float64Array(9), nxt = new Float64Array(9);
  cur[Math.min(2, s0.rh || 0) * 3 + Math.min(2, s0.ra || 0)] = 1;
  let pH = 0, pA = 0;
  const byStep = [];
  for (let t = Math.max(0, s0.step | 0); t < c.n; t++) {
    const b = c.steps[t].bin; nxt.fill(0);
    let stepP = 0;
    for (let rh = 0; rh < 3; rh++) for (let ra = 0; ra < 3; ra++) {
      const p = cur[rh * 3 + ra]; if (p < 1e-15) continue;
      const pgh = lh * c.goalF[b][dH][menIdx(rh, ra)], pga = la * c.goalF[b][dA][menIdx(ra, rh)];
      const prh = rh < 2 ? c.redH[b][dH] : 0, pra = ra < 2 ? c.redA[b][dA] : 0;
      pH += p * pgh; pA += p * pga; stepP += p * (pgh + pga);
      nxt[rh * 3 + ra] += p * (1 - pgh - pga - prh - pra);
      if (prh) nxt[(rh + 1) * 3 + ra] += p * prh;
      if (pra) nxt[rh * 3 + ra + 1] += p * pra;
    }
    byStep.push(stepP);
    const tmp = cur; cur = nxt; nxt = tmp;
  }
  return { pH, pA, pNone: Math.max(0, 1 - pH - pA), byStep };
}

/** Resumos derivados da matriz de placar (mercados "clássicos" de torcedor). */
export function summarize(dist) {
  const m = dist.matrix;
  const totals = new Array(2 * MAXG + 1).fill(0);
  let btts = 0, csH = 0, csA = 0;
  const scores = [];
  for (let i = 0; i <= MAXG; i++) for (let j = 0; j <= MAXG; j++) {
    const p = m[i][j];
    totals[i + j] += p;
    if (i > 0 && j > 0) btts += p;
    if (j === 0) csH += p;
    if (i === 0) csA += p;
    if (p > 0.0005) scores.push({ h: i, a: j, p });
  }
  scores.sort((a, b) => b.p - a.p);
  const over = (line) => totals.reduce((s, p, k) => s + (k > line ? p : 0), 0);
  return { totals, btts, cleanSheetH: csH, cleanSheetA: csA, over05: over(0.5), over15: over(1.5), over25: over(2.5), over35: over(3.5), top: scores.slice(0, 12) };
}

/** Estado "ao vivo" a partir de minuto/placar/vermelhos. */
export function liveState(hz, { minute, added = 0, half, hg, ag, rh = 0, ra = 0, finished = false }) {
  const c = compile(hz);
  if (finished) return { step: c.n, hg, ag, rh, ra };
  // o minuto exibido é o minuto em curso; o passo já "gasto" é o anterior
  let step = stepOf(hz, Math.max(1, minute), added, half);
  // no intervalo, começa do início do 2º tempo
  return { step: Math.min(c.n - 1, step), hg, ag, rh, ra };
}
