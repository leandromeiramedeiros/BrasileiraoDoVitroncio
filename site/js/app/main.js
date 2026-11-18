// Ponto de entrada: roteador por hash, laço de renderização e ações da interface.
import { S, boot, subscribe, runSeasonSim, liveMatches } from './store.js';
import { store, esc, fmtFull } from './util.js';
import { setTab } from './player.js';
import { homeView, homeState } from './views/home.js';
import { matchView, startSim, simAction, closeSim, sim, pairMatch } from './views/match.js';
import { tableView } from './views/table.js';
import { simulatorView, simState, runWhatIf } from './views/simulator.js';
import { teamsView, teamView } from './views/teams.js';
import { modelView } from './views/model.js';
import { calendarView, playersView, calState } from './views/extra.js';

const view = document.getElementById('view');
const nav = document.getElementById('nav');
const NAV = [['', 'Início'], ['jogos', 'Jogos'], ['tabela', 'Classificação'], ['simulador', 'Simulador'], ['times', 'Times'], ['artilharia', 'Artilharia'], ['modelo', 'O modelo']];
const route = () => location.hash.replace(/^#\/?/, '').split('/').filter(Boolean);

let last = '', lastNav = '', queued = false;
function render() {
  if (queued) return; queued = true;
  requestAnimationFrame(() => {
    queued = false;
    if (!S.model) return;
    const [p, a] = route();
    let html;
    try {
      html = p === 'jogo' ? matchView(a) : p === 'tabela' ? tableView() : p === 'simulador' ? simulatorView() : p === 'times' ? teamsView() : p === 'time' ? teamView(a) : p === 'modelo' ? modelView() : p === 'jogos' ? calendarView() : p === 'artilharia' ? playersView() : homeView();
    } catch (e) { console.error(e); html = `<div class="wrap"><div class="banner">Algo deu errado ao montar esta página: ${esc(e.message)}</div></div>`; }
    if (html !== last) {
      // preserva a rolagem da lista de lances ao redesenhar
      const feed = view.querySelector('.feed'); const st = feed ? feed.scrollTop : 0;
      view.innerHTML = html; last = html;
      const nf = view.querySelector('.feed'); if (nf && st) nf.scrollTop = st;
    }
    const cur = p === 'jogo' ? '' : p === 'time' ? 'times' : p || '';
    const live = liveMatches().length;
    const navHtml = NAV.map(([k, l]) => `<a href="#/${k}" class="${k === cur ? 'on' : ''}">${l}${k === '' && live ? ' <span class="livepill" style="font-size:9px;padding:1px 6px;margin-left:4px">' + live + '</span>' : ''}</a>`).join('');
    if (navHtml !== lastNav) { nav.innerHTML = navHtml; lastNav = navHtml; }
    document.title = live ? `🔴 Ao vivo · Brasileirão do Vitrôncio` : 'Brasileirão do Vitrôncio';
    const stamp = document.getElementById('stamp');
    if (stamp && !stamp.textContent) stamp.textContent = `${S.model.nome} · modelo reajustado em ${fmtFull(S.model.geradoEm)} · dados de jogo até ${fmtFull(S.model.dadosAte)} na foto semanal (o resto é atualizado ao vivo).`;
    runSeasonSim();
  });
}

window.addEventListener('hashchange', () => { closeSim(); homeState.simHere = false; window.scrollTo(0, 0); render(); });
subscribe(render);
setInterval(render, 30000); // contagens regressivas

document.addEventListener('click', (ev) => {
  const el = ev.target.closest('[data-act]'); if (!el) return;
  const act = el.dataset.act, id = el.dataset.id;
  if (el.tagName === 'A') ev.preventDefault();
  if (act === 'tab') setTab(el.dataset.key, el.dataset.tab);
  else if (act === 'selectlive') { homeState.selected = id; window.scrollTo({ top: 0, behavior: 'smooth' }); }
  else if (act === 'simstart') { const m = S.matches.get(id); if (m) { homeState.simHere = !!el.dataset.home; startSim(m, render); window.scrollTo({ top: 0, behavior: 'smooth' }); } }
  else if (act === 'simpair') { if (simState.h !== simState.a) startSim(pairMatch(simState.h, simState.a), render); }
  else if (act === 'sim') { simAction(el.dataset.do, el.dataset.val); if (el.dataset.do === 'close') homeState.simHere = false; }
  else if (act === 'swap') { [simState.h, simState.a] = [simState.a, simState.h]; closeSim(); }
  else if (act === 'fav') { store.set('fav', store.get('fav') === id ? null : id); }
  else if (act === 'whatif') { if (simState.fixed[id] === el.dataset.r) delete simState.fixed[id]; else simState.fixed[id] = el.dataset.r; }
  else if (act === 'clearwhatif') { simState.fixed = {}; simState.result = null; }
  else if (act === 'calmonth') calState.month = id;
  else if (act === 'runwhatif') { if (!simState.running) runWhatIf(render); return; }
  last = ''; render();
});
document.addEventListener('change', (ev) => {
  const el = ev.target.closest('[data-change]'); if (!el) return;
  if (el.dataset.change === 'pair') { simState[el.dataset.side] = el.value; closeSim(); last = ''; render(); }
  else if (el.dataset.change === 'calteam') { calState.team = el.value; last = ''; render(); }
});

boot().then(render).catch((e) => { console.error(e); view.innerHTML = `<div class="wrap"><div class="banner">Não consegui carregar o modelo (${esc(e.message)}). Recarregue a página.</div></div>`; });
