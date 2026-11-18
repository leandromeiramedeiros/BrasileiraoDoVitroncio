// Best Against: os jogos em que o Vitrôncio mais discorda do mercado, ordenados pelo
// ganho esperado calculado com o histórico de discordâncias parecidas
// (engine/against.js). Pedido do dono; mostra sempre o retrospecto, que na maior
// parte das temporadas é negativo. Não é recomendação de aposta.
import { S, team, upcoming, liveMatches, forecast, market, table, loadOdds, expected, loadProps, scorers } from '../store.js';
import { esc, pct, num, fmtShort, fmtTime, store } from '../util.js';
import { logo } from '../components.js';
import { bestAgainst, GAPS } from '../../engine/against.js';
import { jointProb, selProb } from '../../engine/markets.js';
import { radarPicks, trustOf } from '../../engine/radar.js';
import { face } from '../vitroncio.js';

const KEYS = ['pH', 'pD', 'pA'];
const dec = (v) => v.toFixed(2).replace('.', ',');
const signed = (v, d = 0) => `${v > 0.00005 ? '+' : v < -0.00005 ? '−' : ''}${Math.abs(v * 100).toFixed(d).replace('.', ',')}%`;
const pickName = (m, i) => (i === 1 ? 'Empate' : `${team(i === 0 ? m.h : m.a).name} vence`);
const pickShort = (m, i) => `<span class="nm-full">${esc(pickName(m, i))}</span><span class="nm-abbr">${i === 1 ? 'Empate' : `${esc(team(i === 0 ? m.h : m.a).abbr)} vence`}</span>`;
const gameCell = (m, extra = '') => `<td class="team"><a href="#/jogo/${m.id}">${logo(team(m.h))}<span class="nm-full">${esc(team(m.h).name)} × ${esc(team(m.a).name)}</span><span class="nm-abbr">${esc(team(m.h).abbr)} × ${esc(team(m.a).abbr)}</span>${logo(team(m.a))}</a>${extra}</td>`;

function early(m) {
  const j = new Map(table().map((r) => [r.id, r.j]));
  return (j.get(m.h) ?? 99) < 5 && (j.get(m.a) ?? 99) < 5;
}
function histText(h) {
  if (!h || !h.n) return '<span class="dim">sem jogos parecidos</span>';
  return `<b>${pct(h.y / h.n)}</b> em ${h.n} <span class="dim">· mercado ${pct(h.q / h.n)}</span>`;
}

/** Linhas de jogos (próximos ou ao vivo) com odds, ordenadas pelo ganho esperado corrigido. */
function rows(list, live = false) {
  const A = S.model.against;
  const out = [];
  for (const m of list) {
    const k = market(m); if (!k) continue;
    const f = forecast(m), P = live ? f.now : f.pre;
    const p = KEYS.map((x) => P[x]), q = KEYS.map((x) => k[x]);
    const b = bestAgainst(A.cells, p, q, early(m));
    if (!b) { out.push({ m, none: true }); continue; }
    out.push({ m, b, p, q, o: k.dec[b.i], ev: b.c[b.i] * k.dec[b.i] - 1, evRaw: p[b.i] * k.dec[b.i] - 1 });
  }
  const ranked = out.filter((r) => !r.none).sort((x, y) => y.ev - x.ev);
  return { ranked, quiet: out.filter((r) => r.none).length };
}

function oppTable(ranked, live = false) {
  return `<div class="tw"><table class="cmp ba"><thead><tr><th class="l">Jogo</th><th class="hide-m">${live ? 'Agora' : 'Quando'}</th><th class="l hide-m">Resultado</th><th><span class="nm-full">Mercado</span><span class="nm-abbr">Merc.</span></th><th>${face(16)}<span class="nm-full">Vitrôncio</span><span class="nm-abbr">Vit.</span></th><th class="l hide-m">Jogos parecidos: aconteceu</th><th><span class="nm-full">Corrigida</span><span class="nm-abbr">Corr.</span></th><th class="hide-m">Odd</th><th><span class="nm-full">Ganho esperado</span><span class="nm-abbr">Ganho</span></th></tr></thead><tbody>
    ${ranked.map(({ m, b, p, q, o, ev }) => `<tr class="${ev > 0 ? 'up' : ''}">${gameCell(m, `<span class="only-m ba-pick">${pickShort(m, b.i)}</span>`)}<td class="small hide-m">${live ? `<span class="livepill">${esc(m.clock || 'ao vivo')}</span>` : `${esc(fmtShort(m.date))} ${fmtTime(m.date)}`}</td>
      <td class="l hide-m"><b>${esc(pickName(m, b.i))}</b></td><td>${pct(q[b.i])}</td><td><b>${pct(p[b.i])}</b></td>
      <td class="l small hide-m hist">${histText(b.hist)}</td><td><b>${pct(b.c[b.i])}</b></td><td class="hide-m">${dec(o)}</td><td class="ev ${ev > 0 ? 'pos' : 'neg'}">${signed(ev)}</td></tr>`).join('')}
  </tbody></table></div>`;
}

const oddsSrc = () => { const m = upcoming().find((x) => market(x)); return m ? `${market(m).src}, via ESPN` : 'ESPN'; };

export function againstView() {
  const A = S.model.against;
  if (!A) return '<div class="wrap"><p class="muted">O histórico do Best Against sai no próximo reajuste do modelo.</p></div>';
  const T = A.total, rawRet = T.rawBets ? T.rawRet / T.rawBets : 0, corRet = T.bets ? T.ret / T.bets : 0;
  loadOdds(liveMatches()); // jogo já começou: busca as odds de antes do apito se este navegador não as viu
  const up = rows(upcoming()), live = rows(liveMatches(), true);
  const rec = A.record.map((r) => ({ ...r, m: S.matches.get(r.id) })).filter((r) => r.m).sort((x, y) => (x.m.date < y.m.date ? 1 : -1));
  const recWins = rec.filter((r) => r.won).length, recRet = rec.reduce((s, r) => s + (r.won ? r.o - 1 : -1), 0);
  const seasons = Object.entries(A.temporadas).filter(([, v]) => v.n).reverse();
  return `<div class="wrap">
    <section data-mk="ba-h"><div class="sec-h"><h2>Best Against</h2><span class="sub">onde o Vitrôncio mais discorda do mercado</span></div>
      <div class="panel ba-warn"><p><b>O mercado costuma ter razão.</b> De 2013 a ${S.season}, apostar onde o Vitrôncio puro via preço bom deu <b>${signed(rawRet, 1)}</b> por aposta (${T.rawBets.toLocaleString('pt-BR')} apostas); os palpites desta aba, que corrigem o Vitrôncio pelo histórico, deram <b>${signed(corRet, 1)}</b> (${T.bets} apostas). Não é recomendação de aposta.</p>
      <p class="small muted">Para cada jogo entra o resultado em que o Vitrôncio está mais acima do mercado (pelo menos 5 pontos). <b>Corrigida</b> parte da probabilidade do mercado e soma o quanto, em jogos passados com diferença parecida (mesmo tamanho, azarão ou não, início da temporada ou não), o resultado aconteceu acima do que o mercado dizia, com desconto quando há poucos jogos. <b>Ganho esperado</b> = corrigida × odd − 1. Odds dos próximos jogos: ${esc(oddsSrc())}; histórico: ${esc(A.fonte)}.</p></div></section>

    <section data-mk="ba-up"><div class="sec-h"><h2>Próximos jogos</h2><span class="sub">ordenados pelo ganho esperado</span></div>
      ${up.ranked.length ? oppTable(up.ranked) : '<p class="muted">Nenhum jogo com odds publicadas em que o Vitrôncio discorde do mercado por 5 pontos ou mais.</p>'}
      ${up.quiet ? `<p class="small muted" style="margin-top:8px">${up.quiet} jogo${up.quiet > 1 ? 's' : ''} com odds em que o Vitrôncio e o mercado estão a menos de 5 pontos.</p>` : ''}</section>

    ${live.ranked.length ? `<section data-mk="ba-live"><div class="sec-h"><h2>Ao vivo</h2><span class="sub">probabilidade do Vitrôncio agora contra as odds de antes do jogo (a ESPN não publica odds ao vivo)</span></div>${oppTable(live.ranked, true)}</section>` : ''}

    <section data-mk="ba-rec"><div class="sec-h"><h2>Temporada ${S.season}</h2><span class="sub">palpites desta aba até o último reajuste, com a odd média de fechamento</span></div>
      ${rec.length ? `<div class="kv"><div><div class="k">Palpites</div><div class="v">${rec.length}</div></div><div><div class="k">Acertos</div><div class="v">${recWins} <small>(${pct(recWins / rec.length)})</small></div></div><div><div class="k">Retorno por aposta</div><div class="v ${recRet >= 0 ? 'gold' : ''}">${signed(recRet / rec.length, 1)}</div></div><div><div class="k">Saldo, 1 por palpite</div><div class="v">${recRet >= 0 ? '+' : '−'}${Math.abs(recRet).toFixed(2).replace('.', ',')}</div></div></div>
      ${rec.length < 60 ? '<p class="small muted" style="margin:0 0 10px">Amostra pequena: compare com as temporadas anteriores, no fim da página.</p>' : ''}
      <div class="tw"><table class="cmp"><thead><tr><th class="hide-m">Data</th><th class="l">Jogo</th><th class="l">Palpite</th><th>Placar</th><th class="hide-m">Mercado</th><th class="hide-m">Corrigida</th><th>Odd</th><th>Saldo</th></tr></thead><tbody>
      ${rec.map((r) => `<tr><td class="small dim hide-m">${esc(fmtShort(r.m.date))}</td>${gameCell(r.m)}<td class="l">${pickShort(r.m, r.k)}</td><td class="pts">${r.m.hg}–${r.m.ag}</td><td class="hide-m">${pct(r.q)}</td><td class="hide-m">${pct(r.c)}</td><td>${dec(r.o)}</td><td>${r.won ? `<span class="chip win">+${dec(r.o - 1)}</span>` : '<span class="chip">−1</span>'}</td></tr>`).join('')}
      </tbody></table></div>` : '<p class="muted">Nenhum palpite nesta temporada até o último reajuste.</p>'}</section>

    <section data-mk="ba-gap"><div class="sec-h"><h2>Quando discordamos, quem acertou?</h2><span class="sub">Série A 2013–${S.season}, resultado em que o Vitrôncio estava mais acima do mercado</span></div>
      <div class="tw"><table><thead><tr><th class="l"><span class="nm-full">Diferença</span><span class="nm-abbr">Dif.</span></th><th>Jogos</th><th><span class="nm-full">Vitrôncio dizia</span><span class="nm-abbr">Vit.</span></th><th><span class="nm-full">Mercado dizia</span><span class="nm-abbr">Merc.</span></th><th><span class="nm-full">Aconteceu</span><span class="nm-abbr">Real</span></th><th><span class="nm-full">Retorno por aposta</span><span class="nm-abbr">Retorno</span></th></tr></thead><tbody>
      ${A.porDiferenca.map((b, i) => `<tr><td class="l">${(GAPS[i] * 100).toFixed(0)}${GAPS[i + 1] ? `–${(GAPS[i + 1] * 100).toFixed(0)}` : '+'}<span class="nm-full"> pontos</span></td><td>${b.n}</td><td>${pct(b.modelo)}</td><td>${pct(b.mercado)}</td><td class="pts">${pct(b.aconteceu)}</td><td class="ev ${b.retorno > 0 ? 'pos' : 'neg'}">${signed(b.retorno, 1)}</td></tr>`).join('')}
      </tbody></table></div>
      <p class="small muted" style="margin-top:8px">Quando "aconteceu" fica perto de "mercado dizia", o mercado tinha razão; perto de "Vitrôncio dizia", o Vitrôncio. Retorno: apostando 1 em cada um desses resultados, na odd média de fechamento.</p>
      <div class="tw" style="margin-top:12px"><table><thead><tr><th>Temporada</th><th>Palpites da aba</th><th>Retorno</th><th>Vitrôncio puro</th><th>Retorno</th></tr></thead><tbody>
      ${seasons.map(([s, v]) => `<tr><td>${s}</td><td>${v.bets}</td><td class="ev ${v.ret > 0 ? 'pos' : 'neg'}">${v.bets ? signed(v.ret / v.bets, 0) : '–'}</td><td>${v.rawBets}</td><td class="ev ${v.rawRet > 0 ? 'pos' : 'neg'}">${v.rawBets ? signed(v.rawRet / v.rawBets, 0) : '–'}</td></tr>`).join('')}
      </tbody></table></div>
      <p class="small muted" style="margin-top:8px">Em cada temporada, os palpites da aba usam só o histórico das temporadas anteriores. Vitrôncio puro: o mesmo resultado, apostado sempre que a probabilidade do Vitrôncio × odd passava de 1.</p></section>
  </div>`;
}

// ── Múltiplas: sugestões da rodada e montador ──────────────────────────────
// Odds multiplicam, e a margem da casa também. Pernas do mesmo jogo: resultado, dupla
// chance, gols e ambos marcam são calculados juntos pela matriz de placares; cartões,
// escanteios e expulsão como independentes do placar (engine/markets.js).
const legsFromStore = () => { const o = store.get('mult2', null); if (o) return o; const old = store.get('mult', {}); const n = {}; for (const [id, j] of Object.entries(old)) n[`${id}|r:${'HDA'[j]}`] = { odd: null }; return n; };
export const multState = { legs: legsFromStore(), open: {} };
const saveLegs = () => store.set('mult2', multState.legs);
export function multToggle(id, sel) { const k = `${id}|${sel}`; if (multState.legs[k]) delete multState.legs[k]; else multState.legs[k] = { odd: null }; saveLegs(); }
export function multSet(keys) { multState.legs = Object.fromEntries(keys.map((k) => [k, { odd: null }])); saveLegs(); }
export function multOdd(key, v) { const x = parseFloat(String(v).replace(',', '.')); if (multState.legs[key]) { multState.legs[key].odd = x > 1 ? x : null; saveLegs(); } }
export function multOpen(id) { multState.open[id] = !multState.open[id]; }

// linhas de mais (em cima) e de menos (embaixo)
const ou = (v, lines) => [...lines.map((L) => [`${v}:o${L}`, () => `+${dec1(L)}`]), ...lines.map((L) => [`${v}:u${L}`, () => `−${dec1(L)}`])];
const GROUPS = [
  ['Dupla chance', [['dc:1X', (m) => `${team(m.h).abbr} ou X`], ['dc:12', () => '1 ou 2'], ['dc:X2', (m) => `X ou ${team(m.a).abbr}`]]],
  ['Gols', ou('g', [1.5, 2.5, 3.5])],
  ['Ambos marcam', [['bt:s', () => 'Sim'], ['bt:n', () => 'Não']]],
  ['Gols do mandante', ou('th', [0.5, 1.5, 2.5])],
  ['Gols do visitante', ou('ta', [0.5, 1.5, 2.5])],
  ['Não sofre gol', [['cs:h', (m) => team(m.h).abbr], ['cs:a', (m) => team(m.a).abbr]]],
  ['Cartões amarelos', ou('c', [3.5, 4.5, 5.5])],
  ['Escanteios', ou('k', [8.5, 9.5, 10.5])],
  ['Expulsão', [['v:s', () => 'Sim'], ['v:n', () => 'Não']]],
];
function dec1(v) { return String(v).replace('.', ','); }
function selName(m, sel) {
  const [v, x] = sel.split(':'), H = team(m.h).name, A = team(m.a).name, L = x && dec1(x.slice(1)), more = x?.[0] === 'o';
  if (v === 'r') return x === 'D' ? 'Empate' : `${x === 'H' ? H : A} vence`;
  if (v === 'dc') return x === '1X' ? `${H} ou empate` : x === 'X2' ? `Empate ou ${A}` : 'Não empata';
  if (v === 'g') return `${more ? 'Mais' : 'Menos'} de ${L} gols`;
  if (v === 'bt') return x === 's' ? 'Ambos marcam' : 'Ambos marcam: não';
  if (v === 'c') return `${more ? 'Mais' : 'Menos'} de ${L} cartões amarelos`;
  if (v === 'k') return `${more ? 'Mais' : 'Menos'} de ${L} escanteios`;
  if (v === 'th' || v === 'ta') return `${v === 'th' ? H : A}: ${more ? 'mais' : 'menos'} de ${L} gol${L === '0,5' ? '' : 's'}`;
  if (v === 'cs') return x === 'h' ? `${H} não sofre gol` : x === 'a' ? `${A} não sofre gol` : x === 'hn' ? `${H} sofre gol` : `${A} sofre gol`;
  if (v === 'p') return `${playerName(m, x)} marca`;
  return x === 's' ? 'Tem expulsão' : 'Sem expulsão';
}
function playerName(m, id) { for (const t of [m.h, m.a]) { const p = (S.model.players?.[t] || []).find((x) => x.id === id); if (p) return `${p.sn || p.n} (${team(t).abbr})`; } return `jogador ${id}`; }
/** Probabilidades e odd publicada (se houver) de uma seleção. */
function mctx(m) {
  const f = forecast(m).pre, e = expected(m).pr;
  const players = {}; for (const [t, xg] of [[m.h, f.xgH], [m.a, f.xgA]]) for (const x of scorers(t, xg)) players[x.id] = x.p;
  return { matrix: f.matrix, pRed: 1 - (1 - f.pRedH) * (1 - f.pRedA), yc: e.yc.lh + e.yc.la, co: e.co.lh + e.co.la, disp: S.model.dispersion, cal: S.model.against?.calMercados, players };
}
function published(m, sel) {
  const k = market(m); if (!k) return null;
  const r = /^r:([HDA])$/.exec(sel);
  if (r) { const j = 'HDA'.indexOf(r[1]); return { o: k.dec[j], q: k[KEYS[j]] }; }
  const g = /^g:([ou])(\d+\.5)$/.exec(sel);
  if (g && k.tot && +g[2] === k.tot.line) { const io = 1 / k.tot.o, iu = 1 / k.tot.u; return { o: g[1] === 'o' ? k.tot.o : k.tot.u, q: (g[1] === 'o' ? io : iu) / (io + iu) }; }
  const pr = S.props.get(m.id)?.find((x) => x.sel === sel);
  return pr ? { o: pr.o, q: pr.q, raw: pr.raw } : null;
}
/** Bilhete: pernas agrupadas por jogo, probabilidades conjuntas e odds. */
function ticket(keys) {
  const byMatch = new Map();
  for (const key of keys) { const [id, sel] = key.split('|'); const m = S.matches.get(id); if (!m) continue; (byMatch.get(id) || byMatch.set(id, { m, sels: [] }).get(id)).sels.push(sel); }
  const legs = [];
  let p = 1, q = 1, c = 1, odds = 1, missing = 0, sameGame = false, allMkt = true, all1x2 = true;
  for (const { m, sels } of byMatch.values()) {
    const ctx = mctx(m);
    p *= jointProb(ctx, sels);
    if (sels.length > 1) sameGame = true;
    for (const sel of sels) {
      const key = `${m.id}|${sel}`, pub = published(m, sel), user = multState.legs[key]?.odd;
      const o = user || pub?.o || null, ps = selProb(ctx, sel);
      if (o) odds *= o; else missing++;
      if (pub) q *= pub.q; else allMkt = false;
      const r = /^r:([HDA])$/.exec(sel);
      if (r && sels.length === 1) { const j = 'HDA'.indexOf(r[1]), k = market(m); const P = forecast(m).pre; const b = k && bestAgainst(S.model.against.cells, KEYS.map((x) => P[x]), KEYS.map((x) => k[x]), early(m)); c *= b ? b.c[j] : pub ? pub.q : ps; } else all1x2 = false;
      legs.push({ key, m, sel, p: ps, o, pub, user });
    }
  }
  const n = legs.length, O = missing ? null : odds;
  return { n, legs, p, q: allMkt && !sameGame ? q : null, c: all1x2 ? c : null, o: O, missing, sameGame, ev: O ? p * O - 1 : null, evC: O && all1x2 ? c * O - 1 : null, evQ: O && allMkt && !sameGame ? q * O - 1 : null };
}
const oddTxt = (v) => (v == null ? '–' : v >= 100 ? Math.round(v).toLocaleString('pt-BR') : dec(v));
const histCell = (k) => { const h = S.model.against.multiplas?.[k]; return h?.n ? `${h.hits} em ${h.n} · <b class="${h.ret > 0 ? 'gold' : ''}">${signed(h.ret / h.n, 0)}</b>` : '–'; };

export function multiView() {
  const A = S.model.against;
  if (!A?.multiplas) return '<div class="wrap"><p class="muted">O histórico de múltiplas sai no próximo reajuste do modelo.</p></div>';
  const games = upcoming().filter((m) => market(m));
  loadProps(games);
  // radar: todas as linhas com odds (resultado e total da ESPN + prop bets da DraftKings) precificadas pelo Vitrôncio
  const offers = [];
  for (const g of games) {
    const k = market(g), ctx = mctx(g), seen = new Set();
    const base = [...['H', 'D', 'A'].map((x, j) => ({ sel: `r:${x}`, o: k.dec[j], q: k[KEYS[j]] })), ...(k.tot ? [['o', k.tot.o], ['u', k.tot.u]].map(([s, o]) => ({ sel: `g:${s}${k.tot.line}`, o, q: 1 / o / (1 / k.tot.o + 1 / k.tot.u) })) : [])];
    for (const x of [...base, ...(S.props.get(g.id) || [])]) { if (seen.has(x.sel)) continue; seen.add(x.sel); const pv = selProb(ctx, x.sel); if (Number.isFinite(pv)) offers.push({ ...x, p: pv, m: g, key: `${g.id}|${x.sel}` }); }
  }
  const radar = radarPicks(offers);
  const bestPer = []; { const seen = new Set(); for (const r of [...radar].sort((u, v) => v.p * v.o - u.p * u.o)) if (!seen.has(r.m.id)) { seen.add(r.m.id); bestPer.push(r); } }
  const rcombo = (xs) => { const P = xs.reduce((a, x) => a * x.p, 1), Q = xs.reduce((a, x) => a * x.q, 1), O = xs.reduce((a, x) => a * x.o, 1); return { P, Q, O, ev: P * O - 1, evQ: Q * O - 1 }; };
  const nProps = games.filter((g) => S.props.get(g.id)?.length).length;
  for (const key of Object.keys(multState.legs)) if (!games.some((m) => key.startsWith(m.id + '|'))) delete multState.legs[key];
  // sugestões: os palpites do Best Against com maior probabilidade corrigida × odd, de 2 a 6 pernas
  const picks = rows(upcoming()).ranked.sort((x, y) => y.b.c[y.b.i] * y.o - x.b.c[x.b.i] * x.o).map((r) => `${r.m.id}|r:${'HDA'[r.b.i]}`);
  const sug = [2, 3, 4, 5, 6].filter((k) => picks.length >= k).map((k) => ({ k, keys: picks.slice(0, k), t: ticket(picks.slice(0, k)) }));
  const T = ticket(Object.keys(multState.legs));
  const mg = A.margem, loss = (k) => signed((1 / (1 + mg)) ** k - 1);
  const calRow = (x) => { const lo = x.tercos[0], hi = x.tercos[2], gain = x.brierBase - x.brier; return `<tr><td class="l">${esc(x.label)}</td><td>${x.n.toLocaleString('pt-BR')}</td><td>${pct(x.previsto)}</td><td>${pct(x.aconteceu)}</td><td class="hide-m">${pct(lo[0])} → ${pct(lo[1])}</td><td class="hide-m">${pct(hi[0])} → ${pct(hi[1])}</td><td class="l small">${gain > 0.001 ? 'sim' : gain > -0.0005 ? 'quase igual à média' : 'não'}</td></tr>`; };
  return `<div class="wrap">
    <section data-mk="mu-h"><div class="sec-h"><h2>Múltiplas</h2><span class="sub">várias apostas numa só: as odds multiplicam</span></div>
      <div class="panel ba-warn"><p><b>A margem da casa também multiplica.</b> Com a margem média da Série A (${pct(mg, 1)} por jogo), mesmo que o mercado esteja certo o retorno esperado é ${loss(2)} com 2 pernas, ${loss(4)} com 4, ${loss(6)} com 6 e ${loss(10)} com 10. As sugestões abaixo, aplicadas em todas as rodadas desde 2014, deram ${[2, 3, 4].map((k) => `${signed(A.multiplas[k].ret / A.multiplas[k].n, 0)} com ${k} pernas`).join(', ')}. Não é recomendação de aposta.</p>
      <p class="small muted">Pernas do mesmo jogo: resultado, dupla chance, gols e ambos marcam são calculados juntos, pelo placar; cartões, escanteios e expulsão são tratados como independentes do placar. As casas costumam cobrar margem maior em múltiplas do mesmo jogo.</p></div></section>

    <section data-mk="mu-radar"><div class="sec-h"><h2>Radar</h2><span class="sub">linhas em que o Vitrôncio dá bem mais chance que o mercado (pelo menos 1,25 vez e 3 pontos a mais)</span></div>
      <p class="small muted" style="margin:0 0 8px">${games.length} jogos com odds; mercados detalhados (DraftKings) de ${nProps}. <b>Odd ≥ 3</b>: o mercado vê pouca chance. Base: <b>histórico</b> = resultado, onde há 14 temporadas de odds e o mercado costuma ter razão; <b>calibrado</b> = o Vitrôncio acerta a frequência desses mercados, mas nunca foi comparado com as odds deles; <b>não validado</b> = marcadores.</p>
      ${radar.length ? `<div class="tw"><table class="cmp ba"><thead><tr><th class="l hide-m">Jogo</th><th class="l">Seleção</th><th>Odd</th><th><span class="nm-full">Mercado</span><span class="nm-abbr">Merc.</span></th><th>${face(16)}<span class="nm-full">Vitrôncio</span><span class="nm-abbr">Vit.</span></th><th class="hide-m">×</th><th><span class="nm-full">Ganho pelo Vitrôncio</span><span class="nm-abbr">Ganho</span></th><th class="l hide-m">Base</th><th></th></tr></thead><tbody>
        ${radar.slice(0, 30).map((r) => `<tr class="${r.contra ? 'up' : ''}"><td class="l small hide-m"><a href="#/jogo/${r.m.id}">${esc(team(r.m.h).abbr)} × ${esc(team(r.m.a).abbr)}</a></td><td class="l wrapc"><span class="only-m small dim">${esc(team(r.m.h).abbr)} × ${esc(team(r.m.a).abbr)}</span>${esc(selName(r.m, r.sel))}${r.contra ? ' <span class="chip">odd ≥ 3</span>' : ''}</td><td>${dec(r.o)}</td><td>${pct(r.q)}${r.raw ? '*' : ''}</td><td><b>${pct(r.p)}</b></td><td class="hide-m">${num(r.ratio, 2)}</td><td class="ev ${r.ev > 0 ? 'pos' : 'neg'}">${signed(r.ev)}</td><td class="l small hide-m">${{ historico: 'histórico', calibrado: 'calibrado', nao: 'não validado' }[trustOf(r.sel)]}</td><td><button class="btn sm ${multState.legs[r.key] ? 'on' : ''}" data-act="mleg" data-id="${r.m.id}" data-sel="${r.sel}" aria-label="Pôr no bilhete">${multState.legs[r.key] ? '✓' : '+'}</button></td></tr>`).join('')}
      </tbody></table></div>${radar.some((r) => r.raw) ? '<p class="small dim" style="margin-top:6px">* marcador: a casa só publica a odd de "sim", então a chance do mercado inclui a margem.</p>' : ''}` : `<p class="muted">${nProps ? 'Nenhuma linha em que o Vitrôncio discorde tanto do mercado agora.' : 'Carregando os mercados detalhados…'}</p>`}
      ${bestPer.length >= 2 ? `<div class="sec-h" style="margin-top:14px"><h3>Múltiplas do radar</h3><span class="sub">a melhor linha de cada jogo, sem histórico: o registro começou em novembro de 2026</span></div>
      <div class="tw"><table class="cmp"><thead><tr><th>Pernas</th><th class="l">Seleções</th><th>Odd</th><th>${face(16)}<span class="nm-full">Vitrôncio</span><span class="nm-abbr">Vit.</span></th><th class="hide-m">Mercado</th><th><span class="nm-full">Ganho pelo Vitrôncio</span><span class="nm-abbr">Ganho</span></th><th class="hide-m">Se o mercado estiver certo</th><th></th></tr></thead><tbody>
        ${[2, 3, 4, 5].filter((k) => bestPer.length >= k).map((k) => { const xs = bestPer.slice(0, k), c = rcombo(xs); return `<tr><td>${k}</td><td class="l small wrapc">${xs.map((x) => `${esc(team(x.m.h).abbr)}×${esc(team(x.m.a).abbr)} <b>${esc(selName(x.m, x.sel))}</b>`).join(' · ')}</td><td>${oddTxt(c.O)}</td><td>${pct(c.P, 1)}</td><td class="hide-m">${pct(c.Q, 1)}</td><td class="ev ${c.ev > 0 ? 'pos' : 'neg'}">${signed(c.ev)}</td><td class="hide-m dim">${signed(c.evQ)}</td><td><button class="btn sm" data-act="mload" data-legs="${xs.map((x) => x.key).join(',')}">abrir</button></td></tr>`; }).join('')}
      </tbody></table></div>` : ''}
      ${A.radar ? radarRecord(A.radar) : '<p class="small muted" style="margin-top:8px">Os palpites do radar passam a ser registrados todo dia; o retrospecto aparece aqui depois dos primeiros jogos.</p>'}
      ${A.zebras ? `<div class="sec-h" style="margin-top:14px"><h3>Odds contra no resultado: o que já aconteceu</h3><span class="sub">Série A 2013–${S.season}, odd média de fechamento</span></div>
      <div class="tw"><table><thead><tr><th class="l">Odd</th><th class="l">O Vitrôncio dava</th><th>Apostas</th><th class="hide-m">Mercado</th><th class="hide-m">Vitrôncio</th><th>Aconteceu</th><th>Retorno</th></tr></thead><tbody>${A.zebras.map((z) => `<tr><td class="l">${esc(z.faixa)}</td><td class="l">${z.ratio > 1 ? `≥ ${num(z.ratio, 1)}× o mercado` : 'qualquer'}</td><td>${z.n}</td><td class="hide-m">${pct(z.mercado)}</td><td class="hide-m">${pct(z.modelo)}</td><td class="pts">${pct(z.aconteceu)}</td><td class="ev ${z.retorno > 0 ? 'pos' : 'neg'}">${signed(z.retorno, 1)}</td></tr>`).join('')}</tbody></table></div>
      <p class="small muted" style="margin-top:6px">Azarões perdem mais que a média (as casas cobram mais neles). Quando o Vitrôncio dá bem mais chance, a perda some nas odds de 3 a 8, mas as amostras são pequenas e o resultado oscila muito.</p>` : ''}</section>

    <section data-mk="mu-sug"><div class="sec-h"><h2>Sugestões com histórico</h2><span class="sub">resultado dos jogos em que o Vitrôncio mais discorda do mercado, corrigido pelo histórico, de 2 a 6 pernas</span></div>
      ${sug.length ? `<div class="tw"><table class="cmp"><thead><tr><th>Pernas</th><th class="l">Palpites</th><th>Odd</th><th>${face(16)}<span class="nm-full">Vitrôncio</span><span class="nm-abbr">Vit.</span></th><th class="hide-m">Mercado</th><th class="hide-m">Corrigida</th><th><span class="nm-full">Ganho esperado</span><span class="nm-abbr">Ganho</span></th><th class="hide-m">Histórico: acertos · retorno</th><th></th></tr></thead><tbody>
        ${sug.map(({ k, keys, t }) => `<tr class="${t.evC > 0 ? 'up' : ''}"><td>${k}</td><td class="l small wrapc">${t.legs.map((l) => `${esc(team(l.m.h).abbr)}×${esc(team(l.m.a).abbr)} <b>${esc(l.sel === 'r:D' ? 'empate' : team(l.sel === 'r:H' ? l.m.h : l.m.a).abbr)}</b>`).join(' · ')}</td><td>${oddTxt(t.o)}</td><td>${pct(t.p, 1)}</td><td class="hide-m">${pct(t.q, 1)}</td><td class="hide-m">${pct(t.c, 1)}</td><td class="ev ${t.evC > 0 ? 'pos' : 'neg'}">${signed(t.evC)}</td><td class="small hide-m">${histCell(k)}</td><td><button class="btn sm" data-act="mload" data-legs="${keys.join(',')}">abrir</button></td></tr>`).join('')}
      </tbody></table></div>` : '<p class="muted">Menos de dois jogos com odds em que o Vitrôncio discorda do mercado por 5 pontos ou mais.</p>'}</section>

    <section data-mk="mu-calc"><div class="sec-h"><h2>Monte sua múltipla</h2><span class="sub">resultado, gols, ambos marcam, cartões, escanteios, expulsão</span>${T.n ? '<button class="btn sm more" data-act="mclear">limpar</button>' : ''}</div>
      ${T.n ? `<div class="panel mticket"><div class="tw"><table><thead><tr><th class="l hide-m">Jogo</th><th class="l">Seleção</th><th>${face(16)}<span class="nm-full">Vitrôncio</span><span class="nm-abbr">Vit.</span></th><th class="hide-m">Odd justa</th><th>Odd</th><th></th></tr></thead><tbody>
        ${T.legs.map((l) => `<tr><td class="l small hide-m"><a href="#/jogo/${l.m.id}">${esc(team(l.m.h).abbr)} × ${esc(team(l.m.a).abbr)}</a></td><td class="l msel"><span class="only-m small dim">${esc(team(l.m.h).abbr)} × ${esc(team(l.m.a).abbr)}</span>${esc(selName(l.m, l.sel))}</td><td>${pct(l.p)}</td><td class="hide-m dim">${dec(1 / l.p)}</td>
          <td><input class="modd" inputmode="decimal" data-change="modd" data-key="${esc(l.key)}" value="${l.user ? dec(l.user) : l.pub ? dec(l.pub.o) : ''}" placeholder="odd" aria-label="Odd oferecida"></td><td><button class="btn sm" data-act="mleg" data-id="${l.m.id}" data-sel="${l.sel}" aria-label="Tirar">×</button></td></tr>`).join('')}
      </tbody></table></div>
      <div class="kv mkv"><div><div class="k">${T.n} perna${T.n > 1 ? 's' : ''}: odd</div><div class="v">${oddTxt(T.o)}</div></div><div><div class="k">Todas acontecem: Vitrôncio</div><div class="v">${pct(T.p, T.p < 0.01 ? 2 : 1)}</div></div><div><div class="k">${T.c != null ? 'Corrigida' : 'Mercado'}</div><div class="v">${T.c != null ? pct(T.c, 1) : T.q != null ? pct(T.q, 1) : '–'}</div></div><div><div class="k">Ganho esperado</div><div class="v ${(T.evC ?? T.ev) > 0 ? 'gold' : ''}">${T.o ? signed(T.evC ?? T.ev) : '–'}</div></div></div>
      <p class="small muted" style="margin:8px 0 0">${T.missing ? `Falta a odd de ${T.missing} perna${T.missing > 1 ? 's' : ''}: digite a odd da sua casa. ` : ''}${T.c != null ? `Ganho calculado com a probabilidade corrigida do Best Against${T.evQ != null ? `; se o mercado estiver certo, ${signed(T.evQ)}` : ''}.` : 'Ganho calculado com a probabilidade do Vitrôncio, que nesses mercados não foi comparada com o mercado (não há odds históricas); veja a calibração no fim da página.'}${T.sameGame ? ' Há pernas do mesmo jogo.' : ''} Histórico das sugestões com ${T.n} perna${T.n > 1 ? 's' : ''}: ${histCell(T.n)}.</p></div>` : '<p class="muted small">Nenhuma seleção. Toque nos resultados abaixo ou em "mais mercados".</p>'}
      ${games.length ? `<div class="mgames">${games.map((g) => { const k = market(g), ctx = mctx(g), on = (sel) => !!multState.legs[`${g.id}|${sel}`], open = multState.open[g.id]; return `<div class="mgame"><div class="t"><a href="#/jogo/${g.id}">${logo(team(g.h))}<span class="nm-full">${esc(team(g.h).name)} × ${esc(team(g.a).name)}</span><span class="nm-abbr">${esc(team(g.h).abbr)} × ${esc(team(g.a).abbr)}</span>${logo(team(g.a))}</a><span class="dim small">${esc(fmtShort(g.date))} ${fmtTime(g.date)}</span></div>
        <div class="seg">${['H', 'D', 'A'].map((x, j) => `<button class="${on(`r:${x}`) ? 'on' : ''}" data-act="mleg" data-id="${g.id}" data-sel="r:${x}">${j === 1 ? 'Empate' : esc(team(j === 0 ? g.h : g.a).abbr)} <b>${dec(k.dec[j])}</b></button>`).join('')}</div>
        <button class="mmore" data-act="mopen" data-id="${g.id}">${open ? 'menos mercados' : 'mais mercados'}</button>
        ${open ? `<div class="mmk">${GROUPS.map(([lab, sels]) => `<div class="mgrp"><span class="small muted">${lab}</span><div class="seg">${sels.map(([sel, f]) => { const pub = published(g, sel); return `<button class="${on(sel) ? 'on' : ''}" data-act="mleg" data-id="${g.id}" data-sel="${sel}">${esc(f(g))} <small>${pct(selProb(ctx, sel))}</small>${pub ? ` <b>${dec(pub.o)}</b>` : ''}</button>`; }).join('')}</div></div>`).join('')}</div>` : ''}</div>`; }).join('')}</div>` : '<p class="muted">Nenhum jogo com odds publicadas.</p>'}</section>

    <section data-mk="mu-hist"><div class="sec-h"><h2>Histórico por número de pernas</h2><span class="sub">a sugestão do site em cada rodada, 2014–${S.season}, odd média de fechamento</span></div>
      <div class="tw"><table><thead><tr><th>Pernas</th><th>Rodadas</th><th>Acertos</th><th>Odd mediana</th><th>Retorno por aposta</th><th>Se o mercado estiver certo</th></tr></thead><tbody>
      ${Object.entries(A.multiplas).map(([k, h]) => `<tr><td>${k}</td><td>${h.n}</td><td>${h.hits}</td><td>${oddTxt(h.oddMediana)}</td><td class="ev ${h.ret > 0 ? 'pos' : 'neg'}">${signed(h.ret / h.n, 1)}</td><td class="dim">${loss(+k)}</td></tr>`).join('')}
      </tbody></table></div></section>

    ${A.mercados ? `<section data-mk="mu-cal"><div class="sec-h"><h2>Os outros mercados são confiáveis?</h2><span class="sub">Série A desde 2016, previsão de dias antes do jogo</span></div>
      <div class="tw"><table><thead><tr><th class="l">Mercado</th><th>Jogos</th><th>Vitrôncio previa</th><th>Aconteceu</th><th class="hide-m">Jogos com chance mais baixa</th><th class="hide-m">Jogos com chance mais alta</th><th class="l">Separa melhor que a média?</th></tr></thead><tbody>${A.mercados.map(calRow).join('')}</tbody></table></div>
      <p class="small muted" style="margin-top:8px">Sem odds históricas desses mercados, o teste é se a chance prevista bate com o que aconteceu. Nas colunas de terço: a chance média prevista no terço de jogos com menor e com maior chance, e quanto de fato aconteceu. Cartões, escanteios e expulsão são recalibrados a cada temporada só com o passado. Expulsão: a chance média bate, mas o modelo quase não distingue um jogo do outro.</p></section>` : ''}
  </div>`;
}

/** Retrospecto dos palpites registrados pelo radar (tools/radar.mjs → reajuste). */
function radarRecord(R) {
  const row = (lab, x) => (x?.n ? `<tr><td class="l">${lab}</td><td>${x.n}</td><td>${x.hits} <small>(${pct(x.hits / x.n)})</small></td><td>${pct(x.p)}</td><td>${pct(x.q)}</td><td class="ev ${x.ret > 0 ? 'pos' : 'neg'}">${signed(x.ret / x.n, 1)}</td></tr>` : '');
  return `<div class="sec-h" style="margin-top:14px"><h3>Retrospecto do radar</h3><span class="sub">palpites registrados desde ${esc(R.desde ? R.desde.split('-').reverse().join('/') : '')}, odd da DraftKings no registro</span></div>
    ${R.total?.n ? `<div class="tw"><table><thead><tr><th class="l">Palpites</th><th>Jogos</th><th>Acertos</th><th>Vitrôncio dava</th><th>Mercado dava</th><th>Retorno</th></tr></thead><tbody>
      ${row('Todos', R.total)}${row('Odd ≥ 3', R.contra)}${row('Resultado', R.tipos?.historico)}${row('Mercados calibrados', R.tipos?.calibrado)}${row('Marcadores', R.tipos?.nao)}</tbody></table></div>` : '<p class="small muted">Nenhum jogo com palpite registrado terminou ainda.</p>'}`;
}
