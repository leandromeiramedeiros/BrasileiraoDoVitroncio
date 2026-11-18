// Best Against: os jogos em que o Vitrôncio mais discorda do mercado, ordenados pelo
// ganho esperado calculado com o histórico de discordâncias parecidas
// (engine/against.js). Pedido do dono; mostra sempre o retrospecto, que na maior
// parte das temporadas é negativo. Não é recomendação de aposta.
import { S, team, upcoming, liveMatches, forecast, market, table, loadOdds } from '../store.js';
import { esc, pct, fmtShort, fmtTime } from '../util.js';
import { logo } from '../components.js';
import { bestAgainst, GAPS } from '../../engine/against.js';
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
