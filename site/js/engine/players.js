// Modelo de jogadores: provável escalação e participação de cada um nos gols,
// cartões e assistências do time — usado para "quem marca" e para dar nome aos
// eventos do jogo simulado.

const GOAL_PRIOR = { F: 0.34, M: 0.11, D: 0.04, G: 0.0005 }; // gols por 90 min, por posição
const AST_PRIOR = { F: 0.14, M: 0.13, D: 0.06, G: 0.003 };
const CARD_PRIOR = { F: 0.17, M: 0.27, D: 0.26, G: 0.08 };
const K = 7; // "jogos" de peso da priori

export function playerRates(p) {
  const n90 = p.min / 90;
  const pos = p.pos || 'M';
  // gols de bola rolando: pênaltis entram à parte (batedor)
  const goals = Math.max(0, (p.g || 0) - (p.pk || 0));
  return {
    goal: (goals + GOAL_PRIOR[pos] * K) / (n90 + K),
    assist: ((p.a || 0) + AST_PRIOR[pos] * K) / (n90 + K),
    card: ((p.y || 0) + CARD_PRIOR[pos] * K) / (n90 + K),
    shot: ((p.sh || 0) + (pos === 'F' ? 2.2 : pos === 'M' ? 1.2 : pos === 'D' ? 0.5 : 0.01) * K) / (n90 + K),
  };
}

/** Escalação provável: goleiro mais usado + 10 de linha com mais minutos recentes. */
export function probableLineup(squad) {
  if (!squad || !squad.length) return { xi: [], bench: [] };
  const score = (p) => (p.rm || 0) + 0.35 * (p.min / Math.max(1, p.apps + p.last)) + (p.rs || 0) * 20 - (p.last > 3 ? 200 : 0);
  const gks = squad.filter((p) => p.pos === 'G').sort((a, b) => score(b) - score(a));
  const out = squad.filter((p) => p.pos !== 'G').sort((a, b) => score(b) - score(a));
  const xi = [];
  if (gks[0]) xi.push(gks[0]);
  // garante um mínimo por setor para a escalação não ficar torta
  const take = (pos, k) => { for (const p of out) { if (xi.length >= 11) break; if (p.pos === pos && !xi.includes(p) && xi.filter((x) => x.pos === pos).length < k) xi.push(p); } };
  take('D', 3); take('M', 2); take('F', 1);
  for (const p of out) { if (xi.length >= 11) break; if (!xi.includes(p)) xi.push(p); }
  const bench = [...gks.slice(1, 2), ...out.filter((p) => !xi.includes(p)).slice(0, 9)];
  const ord = { G: 0, D: 1, M: 2, F: 3 };
  xi.sort((a, b) => ord[a.pos] - ord[b.pos]);
  return { xi, bench };
}

/**
 * Probabilidade de cada jogador marcar ao menos um gol, dado o total esperado
 * de gols do time (lambda). `lineup` = {xi, bench}; reservas pesam menos.
 */
export function scorerProbabilities(lineup, lambda, { ownGoalShare = 0.022, penShare = 0.085 } = {}) {
  const entries = [];
  const pkTaker = [...lineup.xi].sort((a, b) => (b.pk || 0) - (a.pk || 0) || (b.g || 0) - (a.g || 0))[0];
  for (const p of lineup.xi) entries.push({ p, w: playerRates(p).goal * 0.88 });
  for (const p of lineup.bench.slice(0, 7)) entries.push({ p, w: playerRates(p).goal * 0.22 });
  const sum = entries.reduce((s, e) => s + e.w, 0) || 1;
  const open = lambda * (1 - ownGoalShare - penShare);
  return entries.map((e) => {
    const exp = (open * e.w) / sum + (e.p === pkTaker ? lambda * penShare * 0.8 : 0);
    return { id: e.p.id, n: e.p.n, sn: e.p.sn, pos: e.p.pos, j: e.p.j, xg: exp, p: 1 - Math.exp(-exp), titular: lineup.xi.includes(e.p), gols: e.p.g };
  }).sort((a, b) => b.p - a.p);
}

/** Escalação genérica quando não há dados de elenco. */
export function genericLineup(teamName) {
  const mk = (pos, i) => ({ id: `${pos}${i}`, n: `${pos === 'G' ? 'Goleiro' : pos === 'D' ? 'Defensor' : pos === 'M' ? 'Meia' : 'Atacante'} ${i} (${teamName})`, sn: `${pos}${i}`, pos, j: '', min: 0, g: 0, a: 0, y: 0, pk: 0, apps: 0, last: 0 });
  const xi = [mk('G', 1), ...[1, 2, 3, 4].map((i) => mk('D', i)), ...[1, 2, 3].map((i) => mk('M', i)), ...[1, 2, 3].map((i) => mk('F', i))];
  const bench = [mk('D', 5), mk('M', 4), mk('M', 5), mk('F', 4), mk('F', 5)];
  return { xi, bench };
}
