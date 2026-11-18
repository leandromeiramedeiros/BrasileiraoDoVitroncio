// Constrói o conjunto de dados compacto (model/data/history.json.gz) a partir
// do cache bruto da ESPN. Um registro por partida, Série A e Série B.
import { readdir, readFile, writeFile, mkdir } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';
import { parseScoreboard } from '../site/js/data/espn-parse.js';

const CACHE = new URL('../model/cache/', import.meta.url);
const OUT = new URL('../model/data/', import.meta.url);

const files = (await readdir(CACHE)).filter((f) => /^bra\.[12]_\d{4}\.json$/.test(f)).sort();
const byId = new Map();
const teams = {};
const teamSeen = {};
for (const f of files) {
  const lg = f.startsWith('bra.1') ? 'A' : 'B';
  const json = JSON.parse(await readFile(new URL(f, CACHE), 'utf8'));
  const { matches, teams: ts } = parseScoreboard(json, lg);
  for (const m of matches) {
    const prev = byId.get(m.id);
    // remarcações aparecem duplicadas: fica a versão mais "avançada"
    const rank = { off: 0, pre: 1, in: 2, post: 3 };
    if (!prev || rank[m.status] >= rank[prev.status]) byId.set(m.id, m);
    for (const id of [m.h, m.a]) {
      if (!teamSeen[id] || m.date > teamSeen[id]) { teamSeen[id] = m.date; teams[id] = ts[id]; }
    }
  }
}
let matches = [...byId.values()].sort((a, b) => (a.date < b.date ? -1 : a.date > b.date ? 1 : a.id.localeCompare(b.id)));

// Jogos adiados que depois ganharam novo id: descarta o "off" se existir outro
// jogo com o mesmo mandante/visitante na mesma temporada.
const key = (m) => `${m.lg}|${m.season}|${m.h}|${m.a}`;
const live = new Set(matches.filter((m) => m.status !== 'off').map(key));
matches = matches.filter((m) => m.status !== 'off' || !live.has(key(m)));

// Validação: gols dos eventos x placar
let bad = 0, played = 0;
for (const m of matches) {
  if (m.status !== 'post') continue;
  played++;
  let hg = 0, ag = 0;
  for (const e of m.ev) if (e.t === 'G' || e.t === 'P' || e.t === 'O') (e.s === 'h' ? hg++ : ag++);
  m.evOk = hg === m.hg && ag === m.ag;
  if (!m.evOk) bad++;
}

const compact = {
  generated: new Date().toISOString(),
  teams,
  // colunas: id, liga, temporada, data, status, mandante, visitante, gm, gv, evOk, eventos, stats
  matches: matches.map((m) => [
    m.id, m.lg, m.season, m.date, m.status, m.h, m.a, m.hg, m.ag, (m.evOk ? 1 : 0) + (m.statsOk ? 2 : 0),
    m.ev.map((e) => [e.m, e.x, e.t, e.s, e.p, e.n, e.pos]),
    m.stats ? [m.stats.h.sh, m.stats.h.sot, m.stats.h.co, m.stats.h.fo, m.stats.h.po, m.stats.a.sh, m.stats.a.sot, m.stats.a.co, m.stats.a.fo, m.stats.a.po] : null,
    m.venue, m.city,
  ]),
};
await mkdir(OUT, { recursive: true });
const buf = gzipSync(Buffer.from(JSON.stringify(compact)), { level: 9 });
await writeFile(new URL('history.json.gz', OUT), buf);

const per = {};
for (const m of matches) { const k = `${m.lg}${m.season}`; (per[k] ||= { n: 0, post: 0, pre: 0, off: 0, in: 0 })[m.status]++; per[k].n++; }
console.log(`partidas: ${matches.length} (encerradas ${played}, eventos inconsistentes ${bad})  times: ${Object.keys(teams).length}  gz: ${(buf.length / 1e6).toFixed(2)} MB`);
console.log(Object.entries(per).map(([k, v]) => `${k}:${v.post}${v.pre ? '+' + v.pre + 'p' : ''}${v.off ? '+' + v.off + 'x' : ''}`).join('  '));
