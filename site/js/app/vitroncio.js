// O Vitrôncio em pessoa (o Victor): o rosto usado pelo site, as aparições e
// os easter eggs. As aparições ficam direto no <body>, fora de #view, para não
// passarem pelo redesenho das páginas.
//
// Easter eggs (não anunciar na interface):
//   · digitar "vitroncio" ou "victor"            → aparece com uma frase
//   · tocar no Victor do rodapé                   → idem
//   · cinco toques seguidos na marca / código Konami → todos os escudos viram o Victor por 20 s
//   · ficar um tempo parado                       → ele espia pelo canto (uma vez por visita)
//   · fim de um jogo simulado                     → reage a zebra, 7 a 1, 0 a 0 e placar mais provável
//   · escolher o time do coração                  → anota e diz as chances
//   · confronto livre de um time contra ele mesmo → reclama
import { pct } from './util.js';

const IMG = { cara: 'img/vitroncio-cara.webp', joia: 'img/vitroncio-joia.webp', olho: 'img/vitroncio-olho.webp' };

/** Rosto do Vitrôncio, para acompanhar tudo que é opinião do modelo. */
export const face = (size = 18) => `<img class="vit-face" src="${IMG.cara}" width="${size}" height="${size}" alt="">`;

const pick = (a) => a[Math.floor(Math.random() * a.length)];
const FALAS = [
  'Confia no Vitrôncio.',
  'Os números não mentem. Às vezes erram.',
  'Eu só calculo. Quem chuta é o seu time.',
  'Probabilidade não é promessa.',
  'Zebra também é estatística.',
  'Simulei o campeonato 20 mil vezes hoje. E você?',
  'Me achou. Agora volta para o jogo.',
];
const calm = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

let pop = null, popTimer = null;
/** Faz o Vitrôncio subir pelo canto da tela com uma fala. pose: 'joia' | 'olho'. */
export function appear(pose, text, ms = 5600) {
  if (!pop) {
    pop = document.createElement('div');
    pop.className = 'vit-pop'; pop.setAttribute('role', 'status'); pop.setAttribute('aria-live', 'polite');
    pop.innerHTML = '<img alt="Vitrôncio"><p class="vit-say"></p>';
    pop.addEventListener('click', hide);
    document.body.appendChild(pop);
  }
  hidePeek();
  const img = pop.firstChild, src = IMG[pose] || IMG.joia;
  const show = () => { pop.classList.remove('on'); void pop.offsetWidth; pop.classList.add('on'); clearTimeout(popTimer); popTimer = setTimeout(hide, ms); };
  pop.dataset.pose = pose;
  pop.lastChild.textContent = text;
  if (img.getAttribute('src') === src && img.complete) show();
  else { img.onload = show; img.src = src; }
}
export function hide() { clearTimeout(popTimer); pop?.classList.remove('on'); }

// ── todos os escudos viram o Victor ──────────────────────────────────────────
let modeTimer = null;
function vitMode() {
  document.body.classList.add('vitmode');
  clearTimeout(modeTimer); modeTimer = setTimeout(() => document.body.classList.remove('vitmode'), 20000);
  appear('joia', 'Hoje todo mundo é Vitrôncio.');
}

// ── espiada pelo canto ───────────────────────────────────────────────────────
let peek = null, peekTimer = null;
function hidePeek() { clearTimeout(peekTimer); peek?.classList.remove('on'); }
function doPeek() {
  if (!peek) {
    peek = document.createElement('button');
    peek.className = 'vit-peek'; peek.type = 'button'; peek.setAttribute('aria-label', 'Vitrôncio espiando');
    peek.innerHTML = `<img src="${IMG.olho}" alt="">`;
    peek.addEventListener('click', () => appear('joia', 'Me achou. Agora volta para o jogo.'));
    document.body.appendChild(peek);
  }
  void peek.offsetWidth; peek.classList.add('on');
  clearTimeout(peekTimer); peekTimer = setTimeout(hidePeek, 7000);
}

/** Reação ao fim de um jogo simulado. r = { hg, ag, pre:{pH,pD,pA}, top:{h,a} } */
export function reactSim({ hg, ag, pre, top }) {
  const p = hg > ag ? pre.pH : hg === ag ? pre.pD : pre.pA;
  if ((hg === 7 && ag === 1) || (hg === 1 && ag === 7)) appear('olho', 'Sete a um. De novo não.');
  else if (p < 0.15) appear('olho', `Zebra. Eu dava ${pct(p)} para isso.`);
  else if (hg === top.h && ag === top.a && Math.random() < 0.5) appear('joia', `${hg} a ${ag}: o placar mais provável. Avisei.`);
  else if (hg + ag === 0 && Math.random() < 0.5) appear('olho', 'Zero a zero. Simula de novo.');
}
/** Reação ao escolher o time do coração. */
export function reactFav(name, s) {
  if (!s) return appear('joia', `${name}, anotado.`);
  appear(s.rebaixamento > 0.3 ? 'olho' : 'joia', `${name}, anotado. Título: ${pct(s.titulo)}. Rebaixamento: ${pct(s.rebaixamento)}.`);
}
export const reactSelf = (name) => appear('olho', `${name} contra ${name}? Dá empate.`);

/** Liga os gatilhos. `busy()` diz se há algo rodando que não deve ser interrompido (jogo simulado). */
export function initVitroncio({ busy = () => false } = {}) {
  // digitar o nome dele; código Konami
  const KON = ['ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 'b', 'a'];
  let buf = '', kon = 0;
  document.addEventListener('keydown', (ev) => {
    if (ev.ctrlKey || ev.metaKey || ev.altKey || ev.target.closest?.('input, select, textarea')) return;
    kon = ev.key === KON[kon] ? kon + 1 : ev.key === KON[0] ? 1 : 0;
    if (kon === KON.length) { kon = 0; vitMode(); return; }
    if (ev.key === 'Escape') { hide(); hidePeek(); return; }
    if (ev.key.length !== 1) return;
    buf = (buf + ev.key.toLowerCase()).normalize('NFD').replace(/[̀-ͯ]/g, '').slice(-12);
    if (buf.endsWith('vitroncio') || buf.endsWith('victor')) { buf = ''; appear('joia', pick(FALAS)); }
  });
  // cinco toques seguidos na marca
  let taps = [];
  document.querySelector('.brand')?.addEventListener('click', () => {
    const now = Date.now(); taps = [...taps.filter((t) => now - t < 2500), now];
    if (taps.length >= 5) { taps = []; vitMode(); }
  });
  // o Victor do rodapé
  document.querySelector('.foot-vit')?.addEventListener('click', () => appear('joia', pick(FALAS)));
  // espiada depois de um tempo parado, uma vez por visita
  let idle = null;
  const seen = () => { try { return sessionStorage.getItem('bdv:peek'); } catch { return '1'; } };
  const arm = () => {
    clearTimeout(idle);
    if (calm() || seen()) return;
    idle = setTimeout(() => {
      if (document.hidden || busy() || pop?.classList.contains('on')) return arm();
      try { sessionStorage.setItem('bdv:peek', '1'); } catch { /* sem armazenamento */ }
      doPeek();
    }, 40000);
  };
  for (const t of ['pointerdown', 'keydown', 'scroll']) addEventListener(t, () => { hidePeek(); arm(); }, { passive: true });
  arm();
  // deixa as duas poses prontas para a primeira aparição não demorar
  setTimeout(() => { for (const k of ['joia', 'olho']) { const i = new Image(); i.src = IMG[k]; } }, 4000);
  console.log('%cVitrôncio de olho.%c Digite o nome dele em qualquer página.', 'font-weight:700;color:#f2c230', 'color:inherit');
}
