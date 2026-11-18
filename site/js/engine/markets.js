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
// bt:s bt:n · th:o1.5 ta:u0.5 (gols de cada time) · cs:h cs:hn cs:a cs:an (não sofre / sofre gol) ·
// c:o4.5 c:u4.5 (amarelos) · k:o9.5 k:u9.5 (escanteios) · f:o25.5 (faltas) · s:o23.5 (finalizações) ·
// n:o8.5 (chutes no gol) · v:s v:n (expulsão) ·
// p:<id do jogador> (marca a qualquer momento; probabilidade em ctx.players, tratada como independente)
// ─────────────────────────────────────────────────────────────────────────────

// contagens do jogo: letra da seleção → canal (média total em ctx[canal], dispersão em ctx.disp[canal])
const COUNT = { c: 'yc', k: 'co', f: 'fo', s: 'sh', n: 'sot' };

const GOAL = {
  'r:H': (i, j) => i > j, 'r:D': (i, j) => i === j, 'r:A': (i, j) => i < j,
  'dc:1X': (i, j) => i >= j, 'dc:12': (i, j) => i !== j, 'dc:X2': (i, j) => i <= j,
  'bt:s': (i, j) => i > 0 && j > 0, 'bt:n': (i, j) => i === 0 || j === 0,
  'cs:h': (i, j) => j === 0, 'cs:hn': (i, j) => j > 0, 'cs:a': (i, j) => i === 0, 'cs:an': (i, j) => i > 0,
};

/** Interpreta uma seleção. */
export function parseSel(id) {
  if (GOAL[id]) return { kind: 'goal', f: GOAL[id] };
  const m = /^([gckfsn]):([ou])(\d+(?:\.5)?)$/.exec(id);
  if (m) {
    const L = +m[3];
    if (m[1] === 'g') return { kind: 'goal', f: m[2] === 'o' ? (i, j) => i + j > L : (i, j) => i + j < L };
    return { kind: 'count', v: m[1], lo: m[2] === 'o' ? L : -1, hi: m[2] === 'u' ? L : Infinity };
  }
  const t = /^t([ha]):([ou])(\d+\.5)$/.exec(id);
  if (t) { const L = +t[3], side = t[1] === 'h' ? (i) => i : (i, j) => j; return { kind: 'goal', f: t[2] === 'o' ? (i, j) => side(i, j) > L : (i, j) => side(i, j) < L }; }
  if (/^p:\d+$/.test(id)) return { kind: 'player', id: id.slice(2) };
  if (id === 'v:s' || id === 'v:n') return { kind: 'red', yes: id === 'v:s' };
  return null;
}

/** Distribuição de uma contagem com média mu e dispersão phi (variância = phi·mu). */
function countPmf(mu, phi) {
  // cobre a cauda inteira (finalizações chegam a 40+ por jogo)
  const kmax = Math.max(40, Math.ceil(mu + 12 * Math.sqrt(Math.max(1, phi || 1) * mu)));
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
  const goal = [], cnt = {}, red = new Set(), pl = [];
  for (const id of ids) {
    const s = parseSel(id); if (!s) return NaN;
    if (s.kind === 'goal') goal.push(s.f);
    else if (s.kind === 'player') { const q = ctx.players?.[s.id]; if (q == null) return NaN; pl.push(q); }
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
    const ch = COUNT[v]; let mu = ctx[ch], phi = ctx.disp?.[ch];
    if (!(mu > 0)) return NaN;
    const k = ctx.cal?.[v]; if (k) { mu = Math.exp(k.a + k.b * Math.log(mu)); phi = k.phi; }
    p *= between(countPmf(mu, phi), c.lo, c.hi);
  }
  for (const q of pl) p *= q;
  if (red.size === 2) return 0;
  if (red.size === 1) { let q = ctx.pRed; const k = ctx.cal?.v; if (k) q = 1 / (1 + Math.exp(-(k.a + k.b * Math.log(q / (1 - q))))); p *= red.has(true) ? q : 1 - q; }
  return p;
}
export const selProb = (ctx, id) => jointProb(ctx, [id]);
