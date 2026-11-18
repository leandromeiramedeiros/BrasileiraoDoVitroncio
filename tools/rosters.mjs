// Atualiza model/data/rosters.json.gz (escalações de cada jogo encerrado, no formato
// de site/js/engine/lineup.js) com os "summaries" da Série A já baixados em
// model/cache/sum por tools/fetch-summaries.mjs.
import { readFile, writeFile, readdir } from 'node:fs/promises';
import { gzipSync, gunzipSync } from 'node:zlib';
import { loadHistory } from './lib/history.mjs';
import { canon } from './lib/context.mjs';

const FILE = new URL('../model/data/rosters.json.gz', import.meta.url);
const SUM = new URL('../model/cache/sum/', import.meta.url);
const list = JSON.parse(gunzipSync(await readFile(FILE)).toString('utf8'));
const have = new Set(list.map((m) => m.id));
const { matches } = await loadHistory();
const byId = new Map(matches.map((m) => [m.id, m]));
let files = [];
try { files = (await readdir(SUM)).filter((f) => f.endsWith('.json')); } catch { /* sem cache */ }

/** Um lado do jogo a partir do roster do summary: titulares, banco, minutos, gols+assistências. */
export function sideOf(r) {
  const xi = [], bench = [], min = {}, ga = {};
  for (const pe of r?.roster || []) {
    const id = pe.athlete?.id && String(pe.athlete.id); if (!id) continue;
    (pe.starter ? xi : bench).push(id);
    const sub = (pe.plays || []).find((x) => x.substitution);
    const at = sub ? parseInt(sub.clock?.displayValue) || 70 : null;
    const m = pe.starter ? (pe.subbedOut ? Math.min(90, at ?? 70) : 90) : pe.subbedIn ? Math.max(1, 90 - Math.min(89, at ?? 70)) : 0;
    if (m > 0) min[id] = m;
    const st = (n) => pe.stats?.find((s) => s.name === n)?.value || 0;
    const g = st('totalGoals') + st('goalAssists'); if (g > 0) ga[id] = g;
  }
  return { xi, bench, min, ga };
}

let added = 0;
for (const f of files) {
  const id = f.replace('.json', ''), m = byId.get(id);
  if (have.has(id) || !m || m.status !== 'post') continue;
  const d = JSON.parse(await readFile(new URL(f, SUM), 'utf8'));
  const side = (tid) => sideOf((d.rosters || []).find((r) => canon(String(r.team?.id), m.lg) === tid));
  const home = side(m.h), away = side(m.a);
  if (home.xi.length < 10 || away.xi.length < 10) continue;
  list.push({ id, date: m.date, lg: m.lg, season: m.season, h: m.h, a: m.a, home, away });
  have.add(id); added++;
}
list.sort((x, y) => (x.date < y.date ? -1 : x.date > y.date ? 1 : x.id.localeCompare(y.id)));
if (added) await writeFile(FILE, gzipSync(Buffer.from(JSON.stringify(list)), { level: 9 }));
console.log(`escalações: ${list.length} jogos (${added} novos)`);
