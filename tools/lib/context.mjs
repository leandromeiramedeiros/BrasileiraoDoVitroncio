// Contexto de cada partida da base (ver site/js/engine/context.js): situação
// definida na tabela, jogo continental em seguida, mando fora da sede e
// portões fechados. Usado pelo reajuste; para jogos futuros o navegador
// recalcula a parte da tabela com os resultados mais recentes.
import { readFile } from 'node:fs/promises';
import { decided } from '../../site/js/engine/context.js';

// A ESPN usa mais de um id para alguns clubes. (id, liga|null) -> id canônico
const ALIAS = { '5266': '3395', '6274|A': '4929', '6080': '3448', '7385|B': '3448', '15577': '4875', '17326': '18126', '17328': '7375', '2256': '900' };
export const canon = (id, lg) => ALIAS[`${id}|${lg}`] || ALIAS[id] || String(id);

const DAY = 86400000;
const norm = (s) => (s == null ? '' : String(s).split(',')[0].toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '').replace(/\s+/g, ' ').trim());
const VENUE_CITY = [['brinco de ouro', 'campinas'], ['moises lucarelli', 'campinas'], ['serra dourada', 'goiania'], ['mineirao', 'belo horizonte'], ['estadio independencia', 'belo horizonte'], ['claudio moac', 'macae'], ['prudentao', 'presidente prudente'], ['eduardo jose farah', 'presidente prudente'], ['fumeirao', 'arapiraca'], ['romeirao', 'juazeiro do norte'], ['vail chaves', 'mogi mirim'], ['ibere ferreira', 'santa cruz rn'], ['luiz jose de lacerda', 'caruaru'], ['lamengo', 'ipatinga'], ['nazarenao', 'goianinha'], ['kleber andrade', 'cariacica']];
const BAD_CITY = new Set(['rio grande do sul', 'goias', 'genova', 'santa cruz', 'campo bom']);

export async function loadCities() { return JSON.parse(await readFile(new URL('../../model/data/cities.json', import.meta.url), 'utf8')); }

/** Cidade (chave de cities.json) onde o jogo foi/será disputado, ou null se desconhecida/implausível. */
export function resolveCity(cities, venue, city) {
  const v = norm(venue), c = norm(city);
  for (const [sub, key] of VENUE_CITY) if (v.includes(sub)) return key;
  if (v === 'jk') return 'itumbiara';
  if (v === 'arena da fonte') return null; // a ESPN confunde a Fonte Nova (Salvador) com a Arena da Fonte (Araraquara)
  if (c === 'santa maria' && v.includes('presidente vargas')) return 'fortaleza';
  if (!c || BAD_CITY.has(c) || !cities[c] || cities[c][2] !== 'BR') return null;
  return c;
}
function km(a, b) {
  const r = Math.PI / 180, p1 = a[0] * r, p2 = b[0] * r, dl = (b[1] - a[1]) * r;
  const h = Math.sin((p2 - p1) / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 6371 * 2 * Math.asin(Math.sqrt(h));
}

/**
 * matches: Séries A e B em ordem cronológica (ids canônicos); extra: jogos de taças continentais [{date,h,a}].
 * Devolve Map id -> { dec:[m,v], nc:[m,v], disp, closed }.
 */
export function buildContext(matches, extra, cities) {
  const out = new Map();
  const t = (m) => Date.parse(m.date);
  // sede de cada time: cidade mais frequente como mandante; por temporada, só se ≥75% dos mandos (mudança de verdade)
  const all = {}, bySeason = {};
  const ckey = new Map();
  for (const m of matches) {
    const c = resolveCity(cities, m.venue, m.city); ckey.set(m.id, c);
    if (!c) continue;
    ((all[m.h] ||= {})[c] = (all[m.h][c] || 0) + 1);
    const k = `${m.h}|${m.season}`; ((bySeason[k] ||= {})[c] = (bySeason[k][c] || 0) + 1);
  }
  const top = (o) => Object.entries(o).sort((a, b) => b[1] - a[1])[0];
  const base = (id, season) => {
    const s = bySeason[`${id}|${season}`];
    if (s) { const [c, n] = top(s), tot = Object.values(s).reduce((a, b) => a + b, 0); if (tot >= 10 && n >= 0.75 * tot) return c; }
    return all[id] ? top(all[id])[0] : null;
  };
  // agenda continental por time
  const cont = {};
  for (const e of extra) { const d = Date.parse(e.date); for (const id of [e.h, e.a]) (cont[id] ||= []).push(d); }
  for (const id in cont) cont[id].sort((a, b) => a - b);
  // próximo jogo de liga de cada time (para saber se o continental é mesmo o jogo seguinte)
  const league = {};
  for (const m of matches) for (const id of [m.h, m.a]) (league[id] ||= []).push(t(m));
  const nextCont = (id, now) => {
    const c = (cont[id] || []).find((d) => d > now + 1);
    if (!c || c - now > 4.6 * DAY) return 0;
    const l = league[id].find((d) => d > now + 1);
    return l && l < c ? 0 : 1;
  };
  const C0 = Date.parse('2020-03-15T00:00:00Z'), C1 = Date.parse('2021-10-02T00:00:00Z');
  for (const m of matches) {
    const now = t(m), b = base(m.h, m.season), c = ckey.get(m.id);
    const disp = b && c && c !== b && cities[b] && km(cities[b], cities[c]) > 100 ? 1 : 0;
    out.set(m.id, { dec: [0, 0], nc: [nextCont(m.h, now), nextCont(m.a, now)], disp, closed: now >= C0 && now < C1 ? 1 : 0 });
  }
  // situação definida: tabela de cada liga/temporada no momento do pontapé inicial
  const groups = new Map();
  for (const m of matches) { const k = `${m.lg}|${m.season}`; (groups.get(k) || groups.set(k, []).get(k)).push(m); }
  for (const list of groups.values()) {
    const ids = [...new Set(list.flatMap((m) => [m.h, m.a]))], total = 2 * (ids.length - 1);
    const row = Object.fromEntries(ids.map((id) => [id, { id, pts: 0, played: 0 }]));
    let pending = [];
    for (const m of list) {
      const now = t(m);
      pending = pending.filter((p) => {
        if (t(p) + 0.08 * DAY > now) return true; // resultado só entra na tabela ~2 h depois do início
        row[p.h].played++; row[p.a].played++;
        if (p.hg > p.ag) row[p.h].pts += 3; else if (p.hg < p.ag) row[p.a].pts += 3; else { row[p.h].pts++; row[p.a].pts++; }
        return false;
      });
      const d = decided(Object.values(row), total);
      out.get(m.id).dec = [d.has(m.h) ? 1 : 0, d.has(m.a) ? 1 : 0];
      if (m.status === 'post' && m.hg != null) pending.push(m);
    }
  }
  return out;
}
