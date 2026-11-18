import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';

/** Carrega model/data/history.json.gz e devolve partidas como objetos. */
export async function loadHistory() {
  const raw = JSON.parse(gunzipSync(await readFile(new URL('../../model/data/history.json.gz', import.meta.url))).toString('utf8'));
  // xG por finalização (posição do chute), já na escala de gols; só existe para jogos com dados de chute
  let xg = {}; try { xg = JSON.parse(await readFile(new URL('../../model/data/xg.json', import.meta.url), 'utf8')); } catch { /* opcional */ }
  const matches = raw.matches.map((r) => ({
    ...(xg[r[0]] ? { xg: xg[r[0]] } : {}),
    id: r[0], lg: r[1], season: r[2], date: r[3], status: r[4], h: r[5], a: r[6], hg: r[7], ag: r[8], evOk: !!(r[9] & 1), statsOk: !!(r[9] & 2),
    ev: r[10].map((e) => ({ m: e[0], x: e[1], t: e[2], s: e[3], p: e[4], n: e[5], pos: e[6] })),
    stats: r[11] ? { h: { sh: r[11][0], sot: r[11][1], co: r[11][2], fo: r[11][3], po: r[11][4] }, a: { sh: r[11][5], sot: r[11][6], co: r[11][7], fo: r[11][8], po: r[11][9] } } : null,
    venue: r[12], city: r[13],
  }));
  return { teams: raw.teams, matches, generated: raw.generated, extra: (raw.extra || []).map((e) => ({ date: e[0], h: e[1], a: e[2] })) };
}
