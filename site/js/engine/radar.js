// ─────────────────────────────────────────────────────────────────────────────
// Radar das múltiplas: lê as odds de "prop bets" da DraftKings (via ESPN) e devolve as
// seleções que o Vitrôncio sabe precificar (engine/markets.js). Puro: roda no
// navegador e em tools/radar.mjs (registro diário dos palpites).
//
// A ESPN não diz qual lado é cada linha; a ordem é fixa: em pares, primeiro "sim"
// e depois "não"; em linhas (escanteios, gols de cada time), primeiro todas as de
// "mais" e depois as de "menos". A ordem é conferida pelas odds: as de "mais" sobem
// com a linha; se não subirem, os lados são trocados.
// Probabilidade do mercado: odds sem a margem dentro de cada par (dupla chance: as
// três somam 2). Marcador: só a odd de "sim" é publicada, então fica com a margem.
// Placar exato, faixas de gols e cartões de jogador não entram (a ESPN não diz qual é qual,
// ou o Vitrôncio não precifica).
// ─────────────────────────────────────────────────────────────────────────────

const tid = (it) => it.team?.$ref?.match(/teams\/(\d+)/)?.[1] || null;
const aid = (it) => it.athlete?.$ref?.match(/athletes\/(\d+)/)?.[1] || null;
const odd = (it) => +(it.current?.over?.decimal ?? it.current?.over?.value ?? NaN);
const lineOf = (it) => it.current?.target?.value;

/** items = resposta de .../odds/100/propBets; h, a = ids da ESPN dos times. → [{ sel, o, q, raw }] */
export function parseProps(items, h, a) {
  const by = {}; for (const it of items || []) (by[it.type?.name] ||= []).push(it);
  const out = [], groups = {};
  const add = (sel, o, g) => { if (!(o > 1)) return; const x = { sel, o, g }; out.push(x); (groups[g] ||= []).push(x); };
  const bt = by['Both Teams To Score'] || [];
  if (bt.length === 2) { add('bt:s', odd(bt[0]), 'bt'); add('bt:n', odd(bt[1]), 'bt'); }
  for (const it of by['Double Chance'] || []) { const t = tid(it); add(t === h ? 'dc:1X' : t === a ? 'dc:X2' : 'dc:12', odd(it), 'dc'); }
  for (const [t, s] of [[h, 'h'], [a, 'a']]) {
    const cs = (by['Team Clean Sheet'] || []).filter((it) => tid(it) === t);
    if (cs.length === 2) { add(`cs:${s}`, odd(cs[0]), `cs${s}`); add(`cs:${s}n`, odd(cs[1]), `cs${s}`); }
  }
  const lines = (list, key) => {
    const seen = new Set(), first = [], second = [];
    for (const it of list) { const L = lineOf(it); if (L == null || !Number.isFinite(+L)) continue; (seen.has(L) ? second : first).push([+L, odd(it)]); seen.add(L); }
    const rising = (xs) => xs.slice().sort((x, y) => x[0] - y[0]).every((x, i, s) => !i || x[1] >= s[i - 1][1]);
    const [ov, un] = rising(first) || !rising(second) ? [first, second] : [second, first];
    for (const [L, o] of ov) add(key('o', L), o, key('', L));
    for (const [L, o] of un) add(key('u', L), o, key('', L));
  };
  lines(by['Total Corners'] || [], (s, L) => `k:${s}${L}`);
  for (const [t, s] of [[h, 'h'], [a, 'a']]) lines((by['Team Total Goals'] || []).filter((it) => tid(it) === t), (x, L) => `t${s}:${x}${L}`);
  for (const it of by['Anytime Goalscorer'] || []) { const id = aid(it); if (id) add(`p:${id}`, odd(it), `p:${id}`); }
  for (const [g, xs] of Object.entries(groups)) {
    const inv = xs.map((x) => 1 / x.o), s = inv.reduce((u, v) => u + v, 0);
    xs.forEach((x, i) => { if (g === 'dc' && xs.length === 3) x.q = (inv[i] * 2) / s; else if (xs.length === 2) x.q = inv[i] / s; else { x.q = inv[i]; x.raw = true; } });
  }
  return out.map(({ sel, o, q, raw }) => ({ sel, o, q, ...(raw ? { raw } : {}) }));
}

/** Confiança no Vitrôncio para cada tipo de seleção (o que já foi verificado). */
export function trustOf(sel) {
  if (/^r:/.test(sel)) return 'historico';   // resultado: 14 temporadas de odds, Best Against corrige
  if (/^p:/.test(sel)) return 'nao';          // marcador: modelo de jogadores não validado
  return 'calibrado';                          // gols, ambos marcam, dupla chance, escanteios: calibrados
}

/** Candidatos do radar: o Vitrôncio dá bem mais chance que o mercado. offers com p (Vitrôncio) já preenchido. */
export function radarPicks(offers, { minRatio = 1.25, minGap = 0.03, minOdd = 1.5 } = {}) {
  return offers.filter((x) => x.p > 0 && x.q > 0 && x.o >= minOdd && x.p / x.q >= minRatio && x.p - x.q >= minGap)
    .map((x) => ({ ...x, ratio: x.p / x.q, ev: x.p * x.o - 1, contra: x.o >= 3 }))
    .sort((u, v) => v.ev - u.ev);
}
