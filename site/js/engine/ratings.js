// ─────────────────────────────────────────────────────────────────────────────
// Ratings Vitrôncio — modelo dinâmico de espaço de estados (Poisson bivariado)
// estimado por filtro de Kalman estendido (EKF), partida a partida.
//
// Cada time carrega, por "canal" (gols, finalizações, escanteios, cartões…),
// uma força ofensiva `o`, uma força defensiva `d` e a incerteza de cada uma
// (`Po`, `Pd`). No canal de gols há ainda o fator casa próprio do time (`g`).
//
//   log λ_mandante  = mu + hfa/2 + g_m + o_m − d_v
//   log λ_visitante = mu − hfa/2 − g_m + o_v − d_m
//
// O mesmo código roda no reajuste semanal (Node) e no navegador, que aplica
// as partidas encerradas depois da "foto" do modelo — é isso que mantém as
// simulações dinâmicas entre um reajuste e outro.
// ─────────────────────────────────────────────────────────────────────────────

import { applyContext } from './context.js';
import { calibrateLambdas } from './match-model.js';

// 'x' é um segundo rating de gols: mesmo filtro do 'g', mas observando o xG medido
// chute a chute quando o jogo tem esse dado (m.xg); sem ele, vê o mesmo que o 'g'.
export const CHANNELS = ['g', 'sh', 'sot', 'co', 'yc', 'fo', 'x'];
const HOME = (c) => c === 'g' || c === 'x'; // canais com fator casa próprio do time

/** Hiperparâmetros padrão por canal (os de `g` são reotimizados no reajuste). */
export const DEFAULT_PARAMS = {
  g: { q: 0.0012, P0: 0.06, rho: 0.85, sSeason: 0.01, dPromoO: 0.18, dPromoD: 0.18, dNew: 0.15, w: 0.6, phi: 1.0, kMu: 0.004, kH: 0.004, Pg0: 0.01, qg: 0.0001, rhoG: 0.9, wRed: 0.8, mu0: Math.log(1.25), hfa0: 0.35 },
  sh: { q: 0.0006, P0: 0.03, rho: 0.8, sSeason: 0.004, dPromoO: 0.05, dPromoD: 0.05, dNew: 0.05, phi: 2.0, kMu: 0.004, kH: 0.004, mu0: Math.log(12.5), hfa0: 0.2 },
  sot: { q: 0.0006, P0: 0.03, rho: 0.8, sSeason: 0.004, dPromoO: 0.08, dPromoD: 0.08, dNew: 0.08, phi: 1.5, kMu: 0.004, kH: 0.004, mu0: Math.log(4.3), hfa0: 0.22 },
  co: { q: 0.0006, P0: 0.03, rho: 0.8, sSeason: 0.004, dPromoO: 0.04, dPromoD: 0.04, dNew: 0.04, phi: 1.6, kMu: 0.004, kH: 0.004, mu0: Math.log(5.1), hfa0: 0.2 },
  yc: { q: 0.0004, P0: 0.02, rho: 0.7, sSeason: 0.003, dPromoO: 0, dPromoD: 0, dNew: 0, phi: 1.2, kMu: 0.004, kH: 0.004, mu0: Math.log(2.3), hfa0: -0.1 },
  fo: { q: 0.0004, P0: 0.02, rho: 0.7, sSeason: 0.003, dPromoO: 0, dPromoD: 0, dNew: 0, phi: 2.0, kMu: 0.004, kH: 0.004, mu0: Math.log(15), hfa0: -0.03 },
};

DEFAULT_PARAMS.x = { ...DEFAULT_PARAMS.g, w: 0.15, phi: 0.7 };

export function createState(params = DEFAULT_PARAMS) {
  const lg = {};
  for (const L of ['A', 'B']) {
    lg[L] = {};
    for (const c of CHANNELS) lg[L][c] = { mu: params[c].mu0, hfa: params[c].hfa0, n: 0 };
  }
  return { teams: {}, lg, applied: 0, lastDate: null };
}

function newTeam(params, lg, season, fresh) {
  const t = { lg, season, n: 0, ch: {} };
  for (const c of CHANNELS) {
    const p = params[c];
    const off = fresh ? 0 : -p.dNew;
    t.ch[c] = { o: off, d: off, Po: p.P0, Pd: p.P0 };
    if (HOME(c)) { t.ch[c].g = 0; t.ch[c].Pg = p.Pg0; }
  }
  return t;
}

/** Garante que o time existe e aplica virada de temporada / troca de divisão. */
export function prepareTeam(state, params, id, lg, season) {
  let t = state.teams[id];
  if (!t) {
    // "fresh" = primeiras temporadas da base (sem histórico da Série B antes de 2008)
    t = state.teams[id] = newTeam(params, lg, season, lg === 'A' && season <= 2007);
    if (lg === 'A' && season > 2007) for (const c of CHANNELS) { t.ch[c].o -= params[c].dPromoO - params[c].dNew; t.ch[c].d -= params[c].dPromoD - params[c].dNew; }
    return t;
  }
  if (season > t.season) {
    const gap = season - t.season; // >1: passou pela Série C (ou fora da base)
    for (const c of CHANNELS) {
      const p = params[c], s = t.ch[c];
      const r = Math.pow(p.rho, gap);
      s.o *= r; s.d *= r;
      s.Po = Math.min(p.P0, r * r * s.Po + p.sSeason * gap);
      s.Pd = Math.min(p.P0, r * r * s.Pd + p.sSeason * gap);
      if (HOME(c)) { const rg = Math.pow(p.rhoG, gap); s.g *= rg; s.Pg = Math.min(p.Pg0, rg * rg * s.Pg + p.qg * 20 * gap); }
      if (gap > 1) { s.o = Math.min(s.o, -p.dNew); s.d = Math.min(s.d, -p.dNew); s.Po = s.Pd = p.P0; }
      else if (lg !== t.lg) {
        const sign = lg === 'A' ? -1 : 1; // subiu: fica mais fraco em relação à nova média
        s.o += sign * p.dPromoO; s.d += sign * p.dPromoD;
        s.Po = Math.min(p.P0, s.Po + p.sSeason); s.Pd = Math.min(p.P0, s.Pd + p.sSeason);
      }
    }
    t.season = season; t.lg = lg; t.n = 0;
  }
  return t;
}

/** Previsão de um canal: taxas esperadas e variâncias (escala log). */
export function predictChannel(state, params, c, hId, aId, lg, neutral = false) {
  const H = state.teams[hId].ch[c], A = state.teams[aId].ch[c], L = state.lg[lg][c];
  const g = HOME(c) && !neutral ? H.g : 0;
  const hfa = neutral ? 0 : L.hfa;
  const etaH = L.mu + hfa / 2 + g + H.o - A.d;
  const etaA = L.mu - hfa / 2 - g + A.o - H.d;
  const Sh = H.Po + A.Pd + (HOME(c) ? H.Pg : 0);
  const Sa = A.Po + H.Pd + (HOME(c) ? H.Pg : 0);
  return { lh: Math.exp(etaH), la: Math.exp(etaA), Sh, Sa };
}

/**
 * Observações de uma partida para cada canal. `xg` = coeficientes do proxy de
 * gols esperados a partir de finalizações ({b0,b1,b2}); null = só gols.
 */
export function observe(m, params, xg) {
  const obs = {};
  // gols ajustados: gols feitos com um a mais em campo valem um pouco menos
  let gh = 0, ga = 0, rh = 0, ra = 0, yh = 0, ya = 0;
  const wRed = params.g.wRed;
  if (m.evOk && m.ev.length) {
    for (const e of m.ev) {
      if (e.t === 'R') { e.s === 'h' ? rh++ : ra++; }
      else if (e.t === 'Y') { e.s === 'h' ? yh++ : ya++; }
      else if (e.t === 'G' || e.t === 'P' || e.t === 'O') {
        if (e.s === 'h') gh += ra > rh ? wRed : 1; else ga += rh > ra ? wRed : 1;
      }
    }
  } else { gh = m.hg; ga = m.ag; for (const e of m.ev) if (e.t === 'Y') (e.s === 'h' ? yh++ : ya++); }
  let yH = gh, yA = ga;
  const s = m.statsOk ? m.stats : null;
  if (xg && s) {
    const xh = Math.max(0.05, xg.b0 + xg.b1 * s.h.sot + xg.b2 * Math.max(0, s.h.sh - s.h.sot));
    const xa = Math.max(0.05, xg.b0 + xg.b1 * s.a.sot + xg.b2 * Math.max(0, s.a.sh - s.a.sot));
    const w = params.g.w;
    yH = w * gh + (1 - w) * xh; yA = w * ga + (1 - w) * xa;
  }
  obs.g = [yH, yA];
  if (m.xg) { const w = params.x?.w ?? 0.15; obs.x = [w * gh + (1 - w) * m.xg[0], w * ga + (1 - w) * m.xg[1]]; } else obs.x = obs.g;
  if (s) {
    obs.sh = [s.h.sh, s.a.sh];
    obs.sot = [s.h.sot, s.a.sot];
    if (s.h.co != null && s.a.co != null) obs.co = [s.h.co, s.a.co];
    if (s.h.fo != null && s.a.fo != null && s.h.fo + s.a.fo > 0) obs.fo = [s.h.fo, s.a.fo];
  }
  if (m.ev.length) obs.yc = [yh, ya];
  return obs;
}

/** Aplica o resultado de uma partida encerrada ao estado (atualização EKF). */
export function applyMatch(state, params, m, xg, channels = CHANNELS) {
  const H = prepareTeam(state, params, m.h, m.lg, m.season);
  const A = prepareTeam(state, params, m.a, m.lg, m.season);
  const obs = observe(m, params, xg);
  for (const c of channels) {
    const y = obs[c];
    if (!y) continue;
    const p = params[c], h = H.ch[c], a = A.ch[c], L = state.lg[m.lg][c];
    // ruído de processo (passeio aleatório por partida)
    h.Po += p.q; h.Pd += p.q; a.Po += p.q; a.Pd += p.q;
    const isG = HOME(c);
    if (isG) h.Pg += p.qg;
    const g = isG ? h.g : 0;
    const lh = Math.exp(L.mu + L.hfa / 2 + g + h.o - a.d);
    const la = Math.exp(L.mu - L.hfa / 2 - g + a.o - h.d);
    const eh = y[0] - lh, ea = y[1] - la;
    // observação 1: gols do mandante  (o_m +, d_v −, g_m +)
    {
      const S = h.Po + a.Pd + (isG ? h.Pg : 0);
      const den = lh * S + p.phi;
      h.o += (h.Po * eh) / den; a.d -= (a.Pd * eh) / den;
      if (isG) { h.g += (h.Pg * eh) / den; h.Pg -= (h.Pg * h.Pg * lh) / den; }
      h.Po -= (h.Po * h.Po * lh) / den; a.Pd -= (a.Pd * a.Pd * lh) / den;
    }
    // observação 2: gols do visitante (o_v +, d_m −, g_m −)
    {
      const S = a.Po + h.Pd + (isG ? h.Pg : 0);
      const den = la * S + p.phi;
      a.o += (a.Po * ea) / den; h.d -= (h.Pd * ea) / den;
      if (isG) { h.g -= (h.Pg * ea) / den; h.Pg -= (h.Pg * h.Pg * la) / den; }
      a.Po -= (a.Po * a.Po * la) / den; h.Pd -= (h.Pd * h.Pd * la) / den;
    }
    // nível e mando da liga (ganho maior enquanto há poucos jogos)
    const kM = Math.max(p.kMu, 1 / (L.n + 30)), kH = Math.max(p.kH, 1 / (L.n + 30));
    L.mu += (kM * (eh / lh + ea / la)) / 2;
    L.hfa += kH * (eh / lh - ea / la) / 2;
    L.n++;
  }
  H.n++; A.n++;
  state.applied++;
  state.lastDate = m.date;
}

/** Previsão completa (todos os canais) para um jogo futuro, sem alterar o estado. */
export function predictMatch(state, params, hId, aId, lg, season, neutral = false) {
  // trabalha numa cópia rasa dos dois times para não "virar a temporada" no estado real
  const tmp = { teams: {}, lg: state.lg };
  for (const id of [hId, aId]) if (state.teams[id]) tmp.teams[id] = structuredClone(state.teams[id]);
  prepareTeam(tmp, params, hId, lg, season);
  prepareTeam(tmp, params, aId, lg, season);
  const out = {};
  for (const c of CHANNELS) out[c] = predictChannel(tmp, params, c, hId, aId, lg, neutral);
  return out;
}

/**
 * Camada de combinação ("stacking"): junta o rating de gols com os ratings de
 * finalizações para chegar aos gols esperados finais de cada lado.
 *   log λ = a0 + a1·log λ_gols + a2·rel(finalizações no alvo) + a3·rel(finalizações) + a4·log(λ_xG / λ_gols)
 * `rel` = quanto o time deve produzir acima/abaixo da média da liga (escala log).
 */
export function stackLambda(pr, L, stack, neutral = false) {
  if (!stack) return { lh: pr.g.lh, la: pr.g.la };
  const rel = (c, o, sign) => Math.log(pr[c][o]) - (L[c].mu + (neutral ? 0 : (sign * L[c].hfa) / 2));
  const f = (o, sign) => Math.exp(stack.a0 + stack.a1 * Math.log(pr.g[o]) + stack.a2 * rel('sot', o, sign) + stack.a3 * rel('sh', o, sign) + (stack.a4 && pr.x ? stack.a4 * Math.log(pr.x[o] / pr.g[o]) : 0));
  return { lh: f('lh', 1), la: f('la', -1) };
}

/**
 * Gols esperados finais (ratings + combinação + contexto do jogo) e demais canais.
 * `cx` = contexto conhecido do jogo ({dec, nc, disp, closed} — ver context.js); sem ele, só as constantes.
 */
export function expectedMatch(model, state, hId, aId, lg, season, neutral = false, cx = null) {
  const pr = predictMatch(state, model.params, hId, aId, lg, season, neutral);
  const base = stackLambda(pr, state.lg[lg], model.stack, neutral);
  const tgt = lg === 'A' ? applyContext(model.ctx, base.lh, base.la, cx, neutral) : base;
  // lh/la são as taxas que alimentam a cadeia de Markov; xh/xa, os gols esperados de fato
  const { lh, la } = model.calib && model.hazard ? calibrateLambdas(model.hazard, tgt.lh, tgt.la, model.calib) : tgt;
  return { lh, la, xh: tgt.lh, xa: tgt.la, pr };
}

/** Força "geral" em gols por jogo contra um time médio, em campo neutro. */
export function teamStrength(state, id, lg = 'A') {
  const t = state.teams[id];
  if (!t) return null;
  const mu = state.lg[lg].g.mu, s = t.ch.g;
  return { att: Math.exp(mu + s.o), def: Math.exp(mu - s.d), rating: s.o + s.d, sd: Math.sqrt(s.Po + s.Pd), home: s.g };
}
