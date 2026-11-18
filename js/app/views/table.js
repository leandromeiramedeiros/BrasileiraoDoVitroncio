// Classificação + chances do Vitrôncio para cada time.
import { S, team, table, liveMatches } from '../store.js?v=2a005902ef';
import { esc, pct } from '../util.js?v=2a005902ef';
import { logo, formDots, heat, fav } from '../components.js?v=2a005902ef';

const zone = (r) => (r <= 4 ? 'z-lib' : r <= 6 ? 'z-pre' : r <= 12 ? 'z-sul' : r >= 17 ? 'z-reb' : '');
export function posDist(pos) {
  const mx = Math.max(...pos, 0.01);
  return `<span class="posdist" title="chance de terminar em cada posição (1º à esquerda)">${pos.map((p) => `<i style="height:${Math.max(1, (p / mx) * 18)}px;opacity:${p > 0.002 ? 0.95 : 0.15}"></i>`).join('')}</span>`;
}
export function tableView() {
  const rows = table(), sim = S.seasonSim, f = fav();
  const body = rows.map((r) => {
    const t = team(r.id), s = sim?.teams[r.id];
    return `<tr data-mk="t${r.id}" class="${zone(r.rank)} ${r.id === f ? 'fav' : ''}"><td class="pos">${r.rank}</td><td class="team"><a href="#/time/${r.id}">${logo(t)}<span>${esc(t.name)}</span>${r.live ? ' <span class="livepill"></span>' : ''}</a></td>
      <td class="pts">${r.pts}</td><td>${r.j}</td><td>${r.v}</td><td>${r.e}</td><td>${r.d}</td><td>${r.gp}</td><td>${r.gc}</td><td>${r.gp - r.gc > 0 ? '+' : ''}${r.gp - r.gc}</td><td>${formDots(r.form)}</td>
      ${s ? `<td class="pts" title="faixa provável: ${s.ptsLo}–${s.ptsHi}">${s.pts.toFixed(0)}</td><td>${heat(s.titulo)}</td><td>${heat(s.libertadores, '69,167,255')}</td><td>${heat(s.sulAmericana, '31,209,127')}</td><td>${heat(s.rebaixamento, '255,61,77')}</td><td>${posDist(s.pos)}</td>` : '<td colspan="6" class="muted small">simulando…</td>'}</tr>`;
  }).join('');
  return `<div class="wrap"><section><div class="sec-h"><h2>Tabela</h2><span class="sub">Série A ${S.season}${liveMatches().length ? ' · inclui os jogos em andamento' : ''}</span></div>
    <div class="tw"><table><thead><tr><th>#</th><th class="l">Time</th><th>P</th><th>J</th><th>V</th><th>E</th><th>D</th><th>GP</th><th>GC</th><th>SG</th><th>Últimos</th>
      <th title="pontos projetados ao fim do campeonato">Pts finais</th><th>Título</th><th>G4</th><th>Sul-Am.</th><th>Z4</th><th>Posição final</th></tr></thead><tbody>${body}</tbody></table></div>
    <div class="legend"><span><i style="background:var(--blue)"></i>Libertadores (fase de grupos)</span><span><i style="background:#7fc4ff"></i>Pré-Libertadores</span><span><i style="background:var(--green)"></i>Sul-Americana</span><span><i style="background:var(--red)"></i>Rebaixamento</span></div>
    <p class="small muted" style="margin-top:12px">As colunas da direita vêm de ${sim ? sim.n.toLocaleString('pt-BR') : 'milhares de'} simulações dos jogos restantes. ${sim ? `Linha de corte mais provável: campeão com <b>${sim.corteTitulo}</b> pontos; o 16º (primeiro fora do Z4) com <b>${sim.corteRebaixamento}</b>.` : ''} As vagas reais em torneios continentais ainda dependem dos campeões das copas.</p></section></div>`;
}
