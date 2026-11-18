// Ponto de entrada: roteador por hash, laço de renderização e ações da interface.
import { S, boot, subscribe, runSeasonSim, liveMatches, team } from './store.js';
import { store, esc, fmtFull } from './util.js';
import { failedLogos } from './components.js';
import { setTab, setFeed } from './player.js';
import { morph } from './morph.js';
import { homeView, homeState } from './views/home.js';
import { matchView, startSim, simAction, closeSim, sim, pairMatch } from './views/match.js';
import { tableView } from './views/table.js';
import { simulatorView, simState, runWhatIf } from './views/simulator.js';
import { teamsView, teamView } from './views/teams.js';
import { modelView } from './views/model.js';
import { calendarView, playersView, calState } from './views/extra.js';
import { compareView, cmpState } from './views/compare.js';
import { againstView } from './views/against.js';
import { initVitroncio, reactFav, reactSelf } from './vitroncio.js';

const view = document.getElementById('view');
const nav = document.getElementById('nav');
const NAV = [['', 'Início'], ['jogos', 'Jogos'], ['campeonato', 'Campeonato'], ['times', 'Times'], ['contra', 'Best Against'], ['modelo', 'Modelo']];
// páginas com abas internas; os endereços antigos continuam valendo
const SUBS = { campeonato: [['', 'Tabela'], ['simulador', 'Simulador']], times: [['', 'Times'], ['jogadores', 'Jogadores']], contra: [['', 'Oportunidades'], ['comparar', 'Comparação completa']] };
const ALIAS = { tabela: ['campeonato'], simulador: ['campeonato', 'simulador'], comparar: ['contra', 'comparar'], artilharia: ['times', 'jogadores'] };
const route = () => { const r = location.hash.replace(/^#\/?/, '').split('/').filter(Boolean); return ALIAS[r[0]] ? [...ALIAS[r[0]], ...r.slice(1)] : r; };
const VIEWS = {
  jogo: (a) => matchView(a), jogos: () => calendarView(), modelo: () => modelView(), time: (a) => teamView(a),
  campeonato: (a) => (a === 'simulador' ? simulatorView() : tableView()),
  times: (a) => (a === 'jogadores' ? playersView() : teamsView()),
  contra: (a) => (a === 'comparar' ? compareView() : againstView()),
};
const subnav = (p, a) => (SUBS[p] ? `<div class="wrap subnav" data-mk="subnav">${SUBS[p].map(([k, l]) => `<a href="#/${p}${k ? '/' + k : ''}" class="${(a || '') === k ? 'on' : ''}">${l}</a>`).join('')}</div>` : '');

let last = '', lastNav = '', lastRoute = null, queued = false;
function render() {
  if (queued) return; queued = true;
  requestAnimationFrame(() => {
    queued = false;
    if (!S.model) return;
    const [p, a] = route();
    const routeKey = `${p || ''}/${a || ''}`;
    let html;
    try {
      html = subnav(p, a) + (VIEWS[p] ? VIEWS[p](a) : homeView());
    } catch (e) { console.error(e); html = `<div class="wrap"><div class="banner">Algo deu errado ao montar esta página: ${esc(e.message)}</div></div>`; }
    if (html !== last) { morph(view, html, routeKey === lastRoute); last = html; lastRoute = routeKey; }
    const cur = p === 'jogo' ? '' : p === 'time' ? 'times' : p || '';
    const live = liveMatches().length;
    const navHtml = NAV.map(([k, l]) => `<a href="#/${k}" class="${k === cur ? 'on' : ''}">${l}${k === '' && live ? `<span class="navlive">${live}</span>` : ''}</a>`).join('');
    if (navHtml !== lastNav) { nav.innerHTML = navHtml; lastNav = navHtml; }
    document.title = live ? 'Ao vivo · Brasileirão do Vitrôncio' : 'Brasileirão do Vitrôncio';
    const stamp = document.getElementById('stamp');
    if (stamp && !stamp.textContent) stamp.textContent = `${S.model.nome} · reajustado em ${fmtFull(S.model.geradoEm)}`;
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
  else if (act === 'feed') setFeed(el.dataset.key, el.dataset.mode);
  else if (act === 'selectlive') { homeState.selected = id; window.scrollTo({ top: 0, behavior: 'smooth' }); }
  else if (act === 'simstart') { const m = S.matches.get(id); if (m) { homeState.simHere = !!el.dataset.home; startSim(m, render); window.scrollTo({ top: 0, behavior: 'smooth' }); } }
  else if (act === 'simpair') { if (simState.h !== simState.a) startSim(pairMatch(simState.h, simState.a), render); else reactSelf(team(simState.h).name); }
  else if (act === 'sim') { simAction(el.dataset.do, el.dataset.val); if (el.dataset.do === 'close') homeState.simHere = false; }
  else if (act === 'swap') { [simState.h, simState.a] = [simState.a, simState.h]; closeSim(); }
  else if (act === 'fav') { const on = store.get('fav') !== id; store.set('fav', on ? id : null); if (on) reactFav(team(id).name, S.seasonSim?.teams[id]); }
  else if (act === 'whatif') { if (simState.fixed[id] === el.dataset.r) delete simState.fixed[id]; else simState.fixed[id] = el.dataset.r; }
  else if (act === 'clearwhatif') { simState.fixed = {}; simState.result = null; }
  else if (act === 'whatifall') simState.all = !simState.all;
  else if (act === 'calmonth') calState.month = id;
  else if (act === 'cmpmore') cmpState.n += 60;
  else if (act === 'runwhatif') { if (!simState.running) runWhatIf(render); return; }
  last = ''; render();
});
document.addEventListener('change', (ev) => {
  const el = ev.target.closest('[data-change]'); if (!el) return;
  if (el.dataset.change === 'pair') { simState[el.dataset.side] = el.value; closeSim(); last = ''; render(); }
  else if (el.dataset.change === 'calteam') { calState.team = el.value; last = ''; render(); }
  else if (el.dataset.change === 'cmpteam') { cmpState.team = el.value; last = ''; render(); }
});

window.bdvLogoFail = (img) => { failedLogos.add(img.getAttribute('src')); last = ''; render(); };
// Versão publicada: se o navegador estiver rodando arquivos antigos guardados
// em cache, recarrega uma vez para pegar a versão nova.
const BUILD = '__BUILD__';
async function checkVersion() {
  if (BUILD.startsWith('__')) return; // rodando local, sem carimbo
  try {
    const live = (await (await fetch(`version.txt?t=${Date.now()}`, { cache: 'no-store' })).text()).trim();
    if (live && live !== BUILD && sessionStorage.getItem('bdv:reloaded') !== live) { sessionStorage.setItem('bdv:reloaded', live); location.reload(); }
  } catch { /* sem rede: segue com o que tem */ }
}
checkVersion();
setInterval(() => { if (!document.hidden && !sim.playing) checkVersion(); }, 5 * 60000);

initVitroncio({ busy: () => sim.playing });
boot().then(render).catch((e) => { console.error(e); view.innerHTML = `<div class="wrap"><div class="banner">Não consegui carregar o modelo (${esc(e.message)}). Recarregue a página.</div></div>`; });
