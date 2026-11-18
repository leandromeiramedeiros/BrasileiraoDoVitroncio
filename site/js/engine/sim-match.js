// ─────────────────────────────────────────────────────────────────────────────
// Jogo simulado "assistível": sorteia UMA trajetória da mesma cadeia de Markov
// usada nas probabilidades (match-model.js) e a veste com eventos de jogo —
// autores dos gols, assistências, cartões, finalizações, escanteios,
// substituições — e narração em português.
// ─────────────────────────────────────────────────────────────────────────────
import { compile, diffIdx, menIdx } from './match-model.js';
import { playerRates } from './players.js';

const pick = (rng, arr) => arr[Math.floor(rng() * arr.length)];

const TXT = {
  kickoff: ['Bola rolando! Começa o jogo.', 'Apita o árbitro: começou!', 'Autoriza o juiz, rola a bola.'],
  goal: [
    'GOOOL do {team}! {p} recebe na área e não perdoa.',
    'É GOL! {p} finaliza com categoria e balança a rede para o {team}.',
    'GOL do {team}! {p} aparece livre e empurra para o fundo do gol.',
    'Tá lá dentro! {p} acerta um chute cruzado, sem chance para o goleiro. Gol do {team}!',
    'GOOOL! {p} aproveita a sobra na área e marca para o {team}.',
    'Golaço do {team}! {p} bate de fora da área, no cantinho.',
  ],
  header: ['GOL de cabeça! {p} sobe mais que todo mundo e testa firme para o {team}.', 'No alto! {p} cabeceia no canto e marca para o {team}.'],
  freekick: ['GOLAÇO de falta! {p} coloca por cima da barreira. Gol do {team}!'],
  assist: [' Assistência de {a}.', ' Passe açucarado de {a}.', ' Cruzamento na medida de {a}.', ' {a} serviu com perfeição.'],
  pen: ['PÊNALTI convertido! {p} desloca o goleiro e marca para o {team}.', 'Gol de pênalti! {p} bate firme no canto. {team} comemora.'],
  own: ['GOL CONTRA! {p} tenta cortar e manda contra o próprio patrimônio. Gol para o {team}.', 'Que azar! {p} desvia contra a própria meta. Gol do {team}.'],
  save: ['Defendeu! {p} finaliza e o goleiro {gk} espalma.', '{p} arrisca de longe, {gk} segura firme.', 'Grande defesa de {gk}! {p} chutou colocado.', '{p} cabeceia no alvo, {gk} faz a defesa.'],
  miss: ['{p} finaliza para fora. Passou perto!', 'Pra fora! {p} isola a chance do {team}.', '{p} bate cruzado e a bola sai raspando a trave.', 'Travado! A defesa bloqueia o chute de {p}.', 'Na trave! {p} quase marca para o {team}.'],
  corner: ['Escanteio para o {team}.', 'A defesa afasta e cede o escanteio ao {team}.'],
  yellow: ['Cartão amarelo para {p} ({team}), por falta dura.', 'Amarelo! {p} ({team}) para o contra-ataque com falta.', '{p} ({team}) reclama demais e leva o amarelo.'],
  red: ['CARTÃO VERMELHO! {p} ({team}) está expulso por entrada violenta.', 'Vermelho direto! {p} deixa o {team} com um a menos.'],
  red2: ['Segundo amarelo! {p} ({team}) é expulso e deixa o time com um a menos.'],
  sub: ['Substituição no {team}: entra {p}, sai {o}.', 'Mexe o {team}: {p} no lugar de {o}.'],
  half: ['Fim do primeiro tempo.', 'Intervalo de jogo.'],
  second: ['Começa o segundo tempo!', 'Bola rolando para a etapa final.'],
  end: ['Fim de jogo!', 'Apita o árbitro: termina a partida!'],
};
const fill = (s, o) => s.replace(/\{(\w+)\}/g, (_, k) => o[k] ?? '');

/**
 * @param model  model.json (usa hazard)
 * @param cfg    {home:{name,lineup}, away:{name,lineup}, lh, la, rates:{sh,sot,co,yc,fo: [mandante, visitante]}}
 * @param rng    gerador (makeRng)
 */
export function simulateMatch(model, cfg, rng) {
  const hz = model.hazard, c = compile(hz);
  const sides = ['h', 'a'].map((s) => {
    const t = s === 'h' ? cfg.home : cfg.away;
    return {
      s, name: t.name, lam: s === 'h' ? cfg.lh : cfg.la,
      on: t.lineup.xi.map((p) => ({ ...p, r: playerRates(p), yc: 0 })),
      bench: t.lineup.bench.map((p) => ({ ...p, r: playerRates(p), yc: 0 })),
      off: [], goals: 0, reds: 0, subs: 0,
      st: { sh: 0, sot: 0, co: 0, yc: 0, rc: 0, fo: 0 },
    };
  });
  const [H, A] = sides;
  const rate = (k, i, d) => (cfg.rates?.[k]?.[i] ?? d);
  // taxas por minuto de eventos "de enfeite" (não mexem no placar)
  const extra = sides.map((T, i) => {
    const sh = rate('sh', i, 12.5), sot = rate('sot', i, 4.3);
    return {
      saved: Math.max(0.3, sot - T.lam * 0.95) / c.n,
      miss: Math.max(1, sh - sot) / c.n,
      corner: rate('co', i, 5) / c.n,
      foul: rate('fo', i, 13.5) / c.n,
      yellow: rate('yc', i, 2.4),
    };
  });
  // minutos de substituição pré-sorteados (até 5 por time, no 2º tempo)
  const subPlan = sides.map(() => {
    const n = 3 + Math.floor(rng() * 3), out = [];
    for (let k = 0; k < n; k++) out.push(hz.S1 + 45 + 8 + Math.floor(rng() * 34));
    if (rng() < 0.12) out.push(20 + Math.floor(rng() * 25));
    return out.sort((a, b) => a - b).slice(0, 5);
  });
  const events = [];
  const emit = (t, type, side, text, extraFields = {}) => events.push({ t, label: t < 0 ? '' : c.steps[Math.min(t, c.n - 1)].label, half: t < 0 ? 1 : c.steps[Math.min(t, c.n - 1)].half, type, side, text, score: [H.goals, A.goals], ...extraFields });
  const outfield = (T) => T.on.filter((p) => p.pos !== 'G');
  const gk = (T) => T.on.find((p) => p.pos === 'G') || T.on[0];
  const choose = (list, wf) => list[rng.weighted(list.map(wf))];

  emit(0, 'kickoff', null, pick(rng, TXT.kickoff));
  for (let t = 0; t < c.n; t++) {
    const b = c.steps[t].bin;
    if (t === 45 + hz.S1) { emit(t - 1, 'halftime', null, `${pick(rng, TXT.half)} ${H.name} ${H.goals} x ${A.goals} ${A.name}.`); emit(t, 'second', null, pick(rng, TXT.second)); }
    for (let i = 0; i < 2; i++) {
      const T = sides[i], O = sides[1 - i];
      const d = diffIdx(T.goals - O.goals);
      // substituições
      while (subPlan[i].length && subPlan[i][0] <= t && T.subs < 5) {
        subPlan[i].shift();
        const cands = outfield(T);
        if (!T.bench.length || !cands.length) break;
        // sai quem tem cartão/menos minutos na temporada; entra alguém da mesma posição se houver
        const outP = choose(cands, (p) => 1 + p.yc * 1.5 + (p.pos === 'F' ? 0.6 : p.pos === 'M' ? 0.5 : 0.15));
        const same = T.bench.filter((p) => p.pos === outP.pos && p.pos !== 'G');
        const pool = same.length ? same : T.bench.filter((p) => p.pos !== 'G');
        if (!pool.length) break;
        const inP = choose(pool, (p) => 1 + (p.min || 0) / 300);
        T.on.splice(T.on.indexOf(outP), 1, inP); T.bench.splice(T.bench.indexOf(inP), 1); T.off.push(outP); T.subs++;
        emit(t, 'sub', T.s, fill(pick(rng, TXT.sub), { team: T.name, p: inP.n, o: outP.n }), { player: inP.n, player2: outP.n });
      }
      // gol
      const pGoal = T.lam * c.goalF[b][d][menIdx(T.reds, O.reds)];
      if (rng() < pGoal) {
        T.goals++; T.st.sh++; T.st.sot++;
        const u = rng();
        if (u < 0.022 && outfield(O).length) {
          const p = choose(outfield(O), (x) => (x.pos === 'D' ? 3 : 1));
          T.st.sh--; T.st.sot--;
          emit(t, 'goal', T.s, fill(pick(rng, TXT.own), { team: T.name, p: p.n }), { player: p.n, kind: 'own', playerId: p.id });
        } else if (u < 0.107) {
          const takers = outfield(T);
          const p = choose(takers, (x) => 0.3 + (x.pk || 0) * 3 + x.r.goal);
          emit(t, 'goal', T.s, fill(pick(rng, TXT.pen), { team: T.name, p: p.n }), { player: p.n, kind: 'pen', playerId: p.id });
        } else {
          const p = choose(outfield(T), (x) => x.r.goal);
          const others = outfield(T).filter((x) => x !== p);
          const a = rng() < 0.72 && others.length ? choose(others, (x) => x.r.assist) : null;
          const v = rng();
          const base = v < 0.17 ? pick(rng, TXT.header) : v < 0.2 ? pick(rng, TXT.freekick) : pick(rng, TXT.goal);
          emit(t, 'goal', T.s, fill(base, { team: T.name, p: p.n }) + (a && v >= 0.2 || (a && v < 0.17) ? fill(pick(rng, TXT.assist), { a: a.n }) : ''), { player: p.n, player2: a?.n, kind: v < 0.17 ? 'header' : 'goal', playerId: p.id });
        }
        continue;
      }
      // vermelho
      const pRed = T.s === 'h' ? c.redH[b][d] : c.redA[b][d];
      if (T.reds < 2 && rng() < pRed && outfield(T).length > 6) {
        const booked = outfield(T).filter((p) => p.yc > 0);
        const second = booked.length && rng() < 0.55;
        const p = second ? pick(rng, booked) : choose(outfield(T), (x) => x.r.card);
        T.on.splice(T.on.indexOf(p), 1); T.off.push(p); T.reds++; T.st.rc++; T.st.fo++;
        emit(t, 'red', T.s, fill(pick(rng, second ? TXT.red2 : TXT.red), { team: T.name, p: p.n }), { player: p.n, playerId: p.id });
        continue;
      }
      // amarelo
      const pY = (extra[i].yellow / 90) * Math.exp(hz.yel.bin[b] + hz.yel.diff[d] + (T.s === 'a' ? hz.yel.away : 0));
      if (rng() < pY) {
        const cands = T.on.filter((p) => p.yc === 0);
        if (cands.length) {
          const p = choose(cands, (x) => x.r.card);
          p.yc = 1; T.st.yc++; T.st.fo++;
          emit(t, 'yellow', T.s, fill(pick(rng, TXT.yellow), { team: T.name, p: p.n }), { player: p.n, playerId: p.id });
        }
      }
      // finalizações que não viram gol, escanteios, faltas
      const push = d < 2 ? 1.15 : d > 2 ? 0.9 : 1;
      if (rng() < extra[i].saved * push) { const p = choose(outfield(T), (x) => x.r.shot); T.st.sh++; T.st.sot++; emit(t, 'save', T.s, fill(pick(rng, TXT.save), { team: T.name, p: p.n, gk: gk(O)?.n || 'o goleiro' }), { player: p.n }); if (rng() < 0.3) { T.st.co++; emit(t, 'corner', T.s, fill(pick(rng, TXT.corner), { team: T.name })); } }
      if (rng() < extra[i].miss * push) { const p = choose(outfield(T), (x) => x.r.shot); T.st.sh++; emit(t, 'miss', T.s, fill(pick(rng, TXT.miss), { team: T.name, p: p.n }), { player: p.n }); }
      if (rng() < extra[i].corner * push * 0.75) { T.st.co++; emit(t, 'corner', T.s, fill(pick(rng, TXT.corner), { team: T.name })); }
      if (rng() < extra[i].foul) T.st.fo++;
    }
  }
  emit(c.n - 1, 'end', null, `${pick(rng, TXT.end)} ${H.name} ${H.goals} x ${A.goals} ${A.name}.`);
  // posse: proporcional à produção ofensiva esperada, com ruído
  const share = Math.min(0.68, Math.max(0.32, 0.5 + 0.22 * Math.log((rate('sh', 0, 12.5) + 1) / (rate('sh', 1, 12.5) + 1)) + (rng() - 0.5) * 0.12));
  H.st.po = Math.round(share * 100); A.st.po = 100 - H.st.po;
  // a marcação de placar em cada evento precisa refletir o placar NAQUELE momento
  let hg = 0, ag = 0;
  for (const e of events) { if (e.type === 'goal') (e.side === 'h' ? hg++ : ag++); e.score = [hg, ag]; }
  for (const e of events) if (e.type === 'halftime' || e.type === 'end') e.text = e.text.replace(/\d+ x \d+/, `${e.score[0]} x ${e.score[1]}`);
  return { events, score: [H.goals, A.goals], stats: { h: H.st, a: A.st }, steps: c.n, S1: hz.S1, S2: hz.S2 };
}
