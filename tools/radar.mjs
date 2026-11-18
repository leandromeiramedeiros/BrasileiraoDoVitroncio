// Registro do radar das múltiplas: para os jogos dos próximos 3 dias, guarda as odds
// publicadas (resultado e total da ESPN + prop bets da DraftKings) e a probabilidade do
// Vitrôncio de cada linha, em model/data/radar/log.json. Cada execução substitui o
// registro dos jogos que ainda não começaram, então fica a última foto antes do apito.
// O reajuste (tools/refit.mjs) confere os palpites com os resultados.
//   node tools/radar.mjs     (roda no `npm run weekly`; vale rodar também antes de cada rodada)
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { parseScoreboard } from '../site/js/data/espn-parse.js';
import * as R from '../site/js/engine/ratings.js';
import { scoreDistribution } from '../site/js/engine/match-model.js';
import { decided } from '../site/js/engine/context.js';
import { buildTable } from '../site/js/engine/season.js';
import { probableLineup, scorerProbabilities } from '../site/js/engine/players.js';
import { selProb } from '../site/js/engine/markets.js';
import { parseProps } from '../site/js/engine/radar.js';

const run = promisify(execFile);
// curl respeita o proxy/CA do ambiente (o fetch do Node não)
const getJSON = async (url) => JSON.parse((await run('curl', ['-sSL', '--fail', '--max-time', '60', '--retry', '2', url], { maxBuffer: 64 << 20 })).stdout);

const DIR = new URL('../model/data/radar/', import.meta.url), LOG = new URL('log.json', DIR);
await mkdir(DIR, { recursive: true });
const model = JSON.parse(await readFile(new URL('../site/data/model.json', import.meta.url), 'utf8'));
let log = {}; try { log = JSON.parse(await readFile(LOG, 'utf8')); } catch { /* primeiro registro */ }

const sb = await getJSON('https://site.api.espn.com/apis/site/v2/sports/soccer/bra.1/scoreboard');
const { matches } = parseScoreboard(sb, 'A');
const fx = new Map(model.fixtures.map((f) => [f.id, f]));
const ids = [...new Set(model.fixtures.flatMap((f) => [f.h, f.a]))];
const T = buildTable(ids, model.fixtures.filter((f) => f.status === 'post'));
const dec = decided(Object.values(T).map((r) => ({ id: r.id, pts: r.pts, played: r.j })), 2 * (ids.length - 1));
let n = 0;
for (const m of matches) {
  if (m.status !== 'pre' || !m.odds || Date.parse(m.date) - Date.now() > 3 * 864e5) continue;
  const f = fx.get(m.id) || {}, c = f.cx || [0, 0, 0];
  const cx = { dec: [dec.has(m.h) ? 1 : 0, dec.has(m.a) ? 1 : 0], nc: [c[0], c[1]], disp: c[2], ab: f.ab || null };
  const e = R.expectedMatch(model, model.state, m.h, m.a, 'A', model.temporada, false, cx);
  const d = scoreDistribution(model.hazard, e.lh, e.la);
  const players = {};
  for (const [t, xg] of [[m.h, d.xgH], [m.a, d.xgA]]) if (model.players?.[t]?.length) for (const x of scorerProbabilities(probableLineup(model.players[t]), xg)) players[x.id] = x.p;
  const ctx = { matrix: d.matrix, pRed: 1 - (1 - d.pRedH) * (1 - d.pRedA), yc: e.pr.yc.lh + e.pr.yc.la, co: e.pr.co.lh + e.pr.co.la, disp: model.dispersion, cal: model.against?.calMercados, players };
  const k = m.odds, inv = k.dec.map((v) => 1 / v), s = inv[0] + inv[1] + inv[2];
  const offers = ['H', 'D', 'A'].map((x, j) => ({ sel: `r:${x}`, o: k.dec[j], q: inv[j] / s }));
  if (k.tot) { const io = 1 / k.tot.o, iu = 1 / k.tot.u; offers.push({ sel: `g:o${k.tot.line}`, o: k.tot.o, q: io / (io + iu) }, { sel: `g:u${k.tot.line}`, o: k.tot.u, q: iu / (io + iu) }); }
  try {
    const pb = await getJSON(`https://sports.core.api.espn.com/v2/sports/soccer/leagues/bra.1/events/${m.id}/competitions/${m.id}/odds/100/propBets?lang=en&region=us&limit=1000`);
    offers.push(...parseProps(pb.items, m.h, m.a));
  } catch { /* sem prop bets para este jogo */ }
  const seen = new Set(), rows = [];
  for (const x of offers) { if (seen.has(x.sel)) continue; seen.add(x.sel); const p = selProb(ctx, x.sel); if (Number.isFinite(p)) rows.push([x.sel, +x.o.toFixed(3), +x.q.toFixed(4), +p.toFixed(4), x.raw ? 1 : 0]); }
  log[m.id] = { at: new Date().toISOString(), date: m.date, h: m.h, a: m.a, modelo: model.geradoEm, offers: rows };
  n++;
}
await writeFile(LOG, JSON.stringify(log));
console.log(`radar: ${n} jogos registrados agora; ${Object.keys(log).length} no total`);
