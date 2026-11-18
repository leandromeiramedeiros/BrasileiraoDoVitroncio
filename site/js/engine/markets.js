// ─────────────────────────────────────────────────────────────────────────────
// Mercados de um jogo para as múltiplas: probabilidade de cada seleção e de várias
// seleções do MESMO jogo acontecerem juntas.
//
// Resultado, dupla chance, gols e ambos marcam saem da matriz de placares da cadeia
// de Markov, então a correlação entre eles é exata (ex.: "mandante vence" + "mais de
// 2,5 gols" soma só os placares em que as duas coisas acontecem). Cartões amarelos e
// escanteios usam os ratings desses canais (média de cada lado) com a dispersão medida
// no reajuste (binomial negativa; Poisson se não houver sobredispersão). Expulsão usa a
// cadeia. Cartões, escanteios e expulsão são tratados como independentes do placar.
//
// ctx.cal (do reajuste) recalibra esses três: média = e^(a + b·log média), dispersão
// própria do total do jogo; expulsão = σ(a + b·logit p). Sem ctx.cal, usa os valores crus.
//
// Seleções: r:H r:D r:A · dc:1X dc:12 dc:X2 · g:o2.5 g:u2.5 (qualquer linha .5) ·
// bt:s bt:n · c:o4.5 c:u4.5 (amarelos) · k:o9.5 k:u9.5 (escanteios) · v:s v:n (expulsão)
// ─────────────────────────────────────────────────────────────────────────────

const GOAL = {
  'r:H': (i, j) => i > j, 'r:D': (i, j) => i === j, 'r:A': (i, j) => i < j,
  'dc:1X': (i, j) => i >= j, 'dc:12': (i, j) => i !== j, 'dc:X2': (i, j) => i <= j,
  'bt:s': (i, j) => i > 0 && j > 0, 'bt:n': (i, j) => i === 0 || j === 0,
};

/** Interpreta uma seleção. */
export function parseSel(id) {
  if (GOAL[id]) return { kind: 'goal', f: GOAL[id] };
  const m = /^([gck]):([ou])(\d+(?:\.5)?)$/.exec(id);
  if (m) {
    const L = +m[3];
    if (m[1] === 'g') return { kind: 'goal', f: m[2] === 'o' ? (i, j) => i + j > L : (i, j) => i + j < L };
    return { kind: 'count', v: m[1], lo: m[2] === 'o' ? L : -1, hi: m[2] === 'u' ? L : Infinity };
  }
  if (id === 'v:s' || id === 'v:n') return { kind: 'red', yes: id === 'v:s' };
  return null;
}

/** Distribuição de uma contagem com média mu e dispersão phi (variância = phi·mu). */
function countPmf(mu, phi, kmax = 40) {
  const p = new Array(kmax + 1);
  if (!(phi > 1.02)) { p[0] = Math.exp(-mu); for (let k = 1; k <= kmax; k++) p[k] = (p[k - 1] * mu) / k; return p; }
  const r = mu / (phi - 1), q = 1 - 1 / phi;
  p[0] = Math.pow(1 / phi, r);
  for (let k = 1; k <= kmax; k++) p[k] = (p[k - 1] * (k - 1 + r) * q) / k;
  return p;
}
const between = (pmf, lo, hi) => { let s = 0; for (let k = 0; k < pmf.length; k++) if (k > lo && k < hi) s += pmf[k]; return s; };

/**
 * ctx = { matrix (placares), pRed (alguma expulsão), yc, co (médias totais), disp: { yc, co } }.
 * Probabilidade de todas as seleções `ids` (do mesmo jogo) acontecerem juntas.
 */
export function jointProb(ctx, ids) {
  const goal = [], cnt = {}, red = new Set();
  for (const id of ids) {
    const s = parseSel(id); if (!s) return NaN;
    if (s.kind === 'goal') goal.push(s.f);
    else if (s.kind === 'count') { const c = (cnt[s.v] ||= { lo: -1, hi: Infinity }); c.lo = Math.max(c.lo, s.lo); c.hi = Math.min(c.hi, s.hi); }
    else red.add(s.yes);
  }
  let p = 1;
  if (goal.length) {
    let g = 0;
    ctx.matrix.forEach((row, i) => row.forEach((v, j) => { if (goal.every((f) => f(i, j))) g += v; }));
    p *= g;
  }
  for (const [v, c] of Object.entries(cnt)) {
    let mu = v === 'c' ? ctx.yc : ctx.co, phi = v === 'c' ? ctx.disp?.yc : ctx.disp?.co;
    const k = ctx.cal?.[v]; if (k) { mu = Math.exp(k.a + k.b * Math.log(mu)); phi = k.phi; }
    p *= between(countPmf(mu, phi), c.lo, c.hi);
  }
  if (red.size === 2) return 0;
  if (red.size === 1) { let q = ctx.pRed; const k = ctx.cal?.v; if (k) q = 1 / (1 + Math.exp(-(k.a + k.b * Math.log(q / (1 - q))))); p *= red.has(true) ? q : 1 - q; }
  return p;
}
export const selProb = (ctx, id) => jointProb(ctx, [id]);
