// ─────────────────────────────────────────────────────────────────────────────
// Best Against — onde o Vitrôncio mais discorda do mercado e o que o histórico de
// discordâncias parecidas diz sobre quem costuma ter razão.
//
// Para cada jogo, o resultado em que o Vitrôncio está mais acima do mercado (p − q,
// pelo menos 5 pontos) cai numa "célula" de jogos passados parecidos:
//   tamanho da diferença (5–10, 10–15, 15–20, 20+ pontos) × azarão ou não no mercado
//   (q < 35%) × início da temporada (as duas equipes com menos de 5 jogos) ou não.
// Probabilidade corrigida = q + Σ(aconteceu − q) / (n + N0): parte do mercado e soma o
// quanto, nesses jogos, o resultado aconteceu acima do que o mercado dizia, encolhido
// para zero quando há poucos jogos (N0 = 100). Os outros dois resultados encolhem na
// mesma proporção. As células vêm do reajuste (tools/refit.mjs), com odds médias de
// fechamento. As odds só entram AQUI; a previsão do Vitrôncio continua independente.
// ─────────────────────────────────────────────────────────────────────────────

export const GAPS = [0.05, 0.10, 0.15, 0.20];
export const UNDERDOG = 0.35, N0 = 100;

export function cellKey(gap, q, early) {
  let gi = 0; for (let j = 0; j < GAPS.length; j++) if (gap >= GAPS[j]) gi = j;
  return `${gi}|${q < UNDERDOG ? 1 : 0}|${early ? 1 : 0}`;
}

/**
 * p, q = [mandante, empate, visitante] (Vitrôncio, mercado sem margem); cells = { chave: { n, ex, … } }.
 * Devolve null se a maior diferença a favor do Vitrôncio for menor que 5 pontos.
 */
export function bestAgainst(cells, p, q, early = false) {
  let i = -1, gap = 0;
  for (let k = 0; k < 3; k++) if (p[k] - q[k] > gap) { gap = p[k] - q[k]; i = k; }
  if (i < 0 || gap < GAPS[0]) return null;
  const key = cellKey(gap, q[i], early), h = cells?.[key] || { n: 0, ex: 0 };
  const ci = Math.min(0.97, Math.max(0.02, q[i] + h.ex / (h.n + N0)));
  const c = q.map((v, k) => (k === i ? ci : (v * (1 - ci)) / (1 - q[i])));
  return { i, gap, c, key, hist: h };
}
