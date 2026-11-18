// ─────────────────────────────────────────────────────────────────────────────
// Desfalques conhecidos na véspera — suspensos e titulares habituais fora do
// último jogo (indício de lesão). Módulo puro, roda no reajuste e no navegador,
// em cima de lineup.js. Desenvolvido na pesquisa de nov/2026 (research/2026-11/absent).
//
// Suspensão automática (regra da CBF): expulsão (vermelho direto ou segundo amarelo)
// → próximo jogo da mesma liga e temporada; a cada 3º amarelo (3, 6, 9…) → um jogo.
// Na ESPN o segundo amarelo aparece como 'Y' + 'R' do mesmo jogador; esse amarelo
// não conta para o acúmulo. 'Y','Y' sem 'R' quase sempre é evento duplicado: só
// vale como expulsão se os minutos do jogador terminam até 3' depois do 2º amarelo.
// Validação 2013+: 2,9% dos suspensos apareceram na escalação do jogo seguinte.
//
// Atributos do próximo jogo de um time (pesos w de lineup.js; ~0,09 por titular):
//   susp_w  Σ w dos suspensos
//   absP_w  Σ w (anterior à ausência) dos titulares habituais fora do último jogo,
//           sem contar suspensos; ausência de mais de 6 jogos = saiu do time (0)
// (susp_g, abs1_*, abs2_w: variantes testadas que não ajudaram.)
// No modelo só entram antes de a escalação sair: com ela, o ganho some (context.js).
// ─────────────────────────────────────────────────────────────────────────────

export const W_MIN = 0.03;          // a "regular": ≥ ~1/3 of a full-time starter's weight (full-time starter ≈ 0.09)
export const G_MIN = 0.05;          // …or ≥ 5% of the team's recent goals+assists
export const MAX_GAP_DAYS = 60;     // last squad must be this recent
export const MAX_STREAK = 6;        // absP: longer absence streaks count as "gone"
const K_USE = 10, D_USE = 0.85;     // same window as lineup.mjs (used by absP only)

const r4 = (x) => Math.round(x * 1e4) / 1e4;
const dayOf = (iso) => Math.floor(Date.parse(iso) / 864e5);

export function createCardState() {
  // yc[lg|season|pid] = yellows counted; pend[pid] = { t: team, lg, season, n: matches left, why };
  // served[team] = players who served a ban in the team's latest league match
  return { yc: {}, pend: {}, served: {} };
}

/** Players suspended for `team`'s next match in (lg, season). */
export function suspendedFor(cards, team, lg, season) {
  const out = [];
  for (const pid in cards.pend) { const s = cards.pend[pid]; if (s.t === String(team) && s.lg === lg && s.season === season) out.push(pid); }
  return out;
}

/** The team plays a league match: its pending bans are served (also used to roll fixtures forward). */
export function serveBans(cards, team, lg, season) {
  team = String(team);
  const served = [];
  for (const pid in cards.pend) {
    const s = cards.pend[pid];
    if (s.t !== team) continue;
    if (s.lg !== lg || s.season !== season) { delete cards.pend[pid]; continue; }   // lapsed at season end
    served.push(pid);
    if (--s.n <= 0) delete cards.pend[pid];
  }
  cards.served[team] = served;
  return served;
}

/**
 * Feed a finished league match: match = { lg, season, h, a, ev: [{t:'Y'|'R'|…, s:'h'|'a', p: athleteId, m: minute}],
 * min?: {athleteId: minutes played} } (h, a = team ids as used in the lineup state). Serves the bans of both teams for this match, then books the new ones.
 * opts.yrYellowCounts: count the yellow of a 'Y'+'R' pair towards accumulation (default false = treated as a second yellow).
 */
export function applyCardMatch(cards, match, opts = {}) {
  const { lg, season } = match;
  serveBans(cards, match.h, lg, season); serveBans(cards, match.a, lg, season);
  const by = {};
  for (const e of match.ev || []) {
    if ((e.t !== 'Y' && e.t !== 'R') || !e.p) continue;
    const k = e.s + '|' + e.p; const o = by[k] || (by[k] = { s: e.s, p: String(e.p), y: 0, r: 0, ym: 0 });
    if (e.t === 'Y') { o.y++; o.ym = Math.max(o.ym, e.m || 0); } else o.r++;
  }
  for (const k in by) {                       // 'Y','Y' without 'R': genuine 2nd yellow only if he left the pitch right then
    const o = by[k]; if (o.y < 2 || o.r) continue;
    const mn = match.min && match.min[o.p];
    if (!(mn != null && mn < o.ym + 3)) o.y = 1;
  }
  const booked = [];
  for (const k in by) {
    const { s, p, y, r } = by[k];
    const team = String(s === 'h' ? match.h : match.a);
    const second = y >= 2 || (y >= 1 && r >= 1);
    const off = r >= 1 || y >= 2;
    const counted = second ? (opts.yrYellowCounts && y === 1 ? 1 : 0) : y;
    let n = off ? 1 : 0, why = off ? (second ? '2Y' : 'R') : '';
    if (counted) {
      const key = lg + '|' + season + '|' + p; const before = cards.yc[key] || 0; const after = before + counted;
      cards.yc[key] = after;
      if (Math.floor(after / 3) > Math.floor(before / 3)) { n += 1; why = why ? why + '+3Y' : '3Y'; }
    }
    if (n > 0) {
      const old = cards.pend[p];
      cards.pend[p] = { t: team, lg, season, n: (old && old.t === team && old.season === season ? old.n : 0) + n, why };
      booked.push({ p, team, why, n });
    }
  }
  return booked;
}

const squadOf = (r) => { const s = new Set([...r.xi, ...r.bench]); for (const id in r.min || {}) s.add(id); return s; };

// usual-XI weights from the team's records BEFORE the last `skip` matches (same window as lineup.mjs)
function weightsBefore(hist, skip) {
  const u = {}; let j = 0;
  for (let i = hist.length - 1 - skip; i >= 0 && j < K_USE; i--, j++) {
    const w = D_USE ** j; for (const id in hist[i].min) u[id] = (u[id] || 0) + (w * hist[i].min[id]) / 90;
  }
  const top11 = Object.values(u).sort((x, y) => y - x).slice(0, 11).reduce((a, x) => a + x, 0);
  const out = {}; if (top11 > 0) for (const id in u) out[id] = u[id] / top11;
  return out;
}

/**
 * Pre-match missing-player features of `team` for its next match of (lg, season) on isoDate.
 * snapshot: teamSnapshot(lineupState, team, isoDate) from lineup.mjs; history: lineupState.teams[team] (may be undefined).
 */
export function absentFeatures(snapshot, history, cards, team, lg, season, isoDate) {
  team = String(team);
  const hist = history || [];
  const p = (snapshot && snapshot.p) || {};
  const f = { susp_w: null, susp_g: null, susp_n: null, abs1_w: null, abs1_g: null, abs2_w: null, absP_w: null, abs1_n: null };
  if (!hist.length) return f;
  const susp = suspendedFor(cards, team, lg, season);
  let sw = 0, sg = 0, sn = 0;
  for (const id of susp) { const q = p[id]; if (q) { sw += q[0]; sg += q[1]; if (q[0] >= W_MIN) sn++; } }
  Object.assign(f, { susp_w: r4(sw), susp_g: r4(sg), susp_n: sn });
  const last = hist[hist.length - 1], day = dayOf(isoDate);
  if (last.season !== season || day - last.day > MAX_GAP_DAYS || !last.bench || !last.bench.length) return f;
  const excl = new Set([...susp, ...(cards.served[team] || [])]);
  const sq = hist.map(squadOf);                                   // oldest → newest
  const L = sq.length;
  const wPre = {};
  let a1w = 0, a1g = 0, a1n = 0, a2w = 0, aPw = 0;
  for (const id in p) {
    const [w, g] = p[id];
    if (w < W_MIN && g < G_MIN) continue;
    if (sq[L - 1].has(id) || excl.has(id)) continue;
    a1w += w; a1g += g; if (w >= W_MIN) a1n++;
    if (L >= 2 && !sq[L - 2].has(id)) a2w += w;
    let k = 1; while (k < L && !sq[L - 1 - k].has(id)) k++;        // length of the absence streak (matches missed)
    if (k <= MAX_STREAK && k < L) { const wb = wPre[k] || (wPre[k] = weightsBefore(hist, k)); aPw += wb[id] || 0; }
  }
  Object.assign(f, { abs1_w: r4(a1w), abs1_g: r4(a1g), abs2_w: r4(a2w), absP_w: r4(aPw), abs1_n: a1n });
  return f;
}

/** Production shift: log λ_home += Σ b_j (x_j,home − x_j,away), log λ_away −= the same (null for either team → 0). */
export function absentShift(coefs, fh, fa) {
  let e = 0;
  coefs.features.forEach((k, j) => { if (fh[k] != null && fa[k] != null) e += coefs.b[j] * (fh[k] - fa[k]); });
  return e;
}
