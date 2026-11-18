// ─────────────────────────────────────────────────────────────────────────────
// Contexto do jogo — ajustes nos gols esperados que não vêm da força dos times:
//   dec    time com a situação já definida (matematicamente rebaixado ou campeão)
//   nc     o próximo jogo do time, em até ~4,5 dias, é de taça continental
//          (Libertadores, Sul-Americana, Recopa, Mundial…): tende a poupar
//   disp   mando jogado a mais de 100 km da sede do mandante
//   closed período de portões fechados (2020–21)
// Mais duas constantes de calibração: nível de gols (tot) e mando (hfa).
//
//   log λ_m += tot + D     log λ_v += tot − D
//   D = hfa + disp·x + closed·f + dec·(d_m − d_v) + nc·(c_m − c_v)
//
// Os coeficientes são estimados no reajuste semanal (tools/refit.mjs).
// ─────────────────────────────────────────────────────────────────────────────

/** f = { dec:[m,v], nc:[m,v], disp:0|1, closed:0|1, lu:[{…},{…}] } (tudo opcional). */
export function applyContext(ctx, lh, la, f, neutral = false) {
  if (!ctx) return { lh, la };
  const diff = (v) => (v ? (v[0] || 0) - (v[1] || 0) : 0);
  let D = (neutral ? 0 : ctx.hfa + (f?.disp ? ctx.disp : 0) + (f?.closed ? ctx.closed : 0)) + ctx.dec * diff(f?.dec) + ctx.nc * diff(f?.nc);
  // escalação anunciada: f.lu = [atributos do mandante, atributos do visitante] (ver lineup.js)
  if (ctx.lineup && f?.lu) for (const k in ctx.lineup) { const x = f.lu[0]?.[k], y = f.lu[1]?.[k]; if (x != null && y != null) D += ctx.lineup[k] * (x - y); }
  return { lh: lh * Math.exp(ctx.tot + D), la: la * Math.exp(ctx.tot - D) };
}

/**
 * Times com a situação matematicamente definida. rows = [{ id, pts, played }]
 * (todos os times da liga), total = jogos de cada time na temporada.
 * Devolve Map id -> 'rebaixado' | 'campeao'. Empates em pontos não contam
 * (critério conservador: só o que nenhum desempate muda).
 */
export function decided(rows, total, nRel = 4) {
  const N = rows.length, out = new Map();
  for (const r of rows) {
    const mx = r.pts + 3 * (total - r.played);
    let above = 0, bestOther = -1;
    for (const o of rows) {
      if (o === r) continue;
      if (o.pts > mx) above++;
      bestOther = Math.max(bestOther, o.pts + 3 * (total - o.played));
    }
    if (above >= N - nRel) out.set(r.id, 'rebaixado');
    else if (r.pts > bestOther) out.set(r.id, 'campeao');
  }
  return out;
}
