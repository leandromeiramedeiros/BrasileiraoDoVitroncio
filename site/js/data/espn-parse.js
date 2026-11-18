// Parser puro (sem DOM, sem rede) do formato de "scoreboard" da ESPN.
// Usado tanto pelo navegador (dados dinâmicos) quanto pelas ferramentas de
// reajuste semanal (tools/*.mjs), para que os dois lados enxerguem os jogos
// exatamente do mesmo jeito.

const GOAL_TYPES = new Set(['Goal', 'Goal - Header', 'Goal - Free-kick', 'Goal - Volley', 'Penalty - Scored', 'Own Goal']);

/** "45'+3'" -> {min:45, add:3};  "67'" -> {min:67, add:0} */
export function parseClock(displayValue, valueSeconds) {
  const m = /^(\d+)'?(?:\+(\d+))?/.exec(displayValue || '');
  if (m) return { min: Math.max(1, +m[1]), add: m[2] ? +m[2] : 0 };
  const min = Math.max(1, Math.ceil((valueSeconds || 0) / 60));
  return { min: Math.min(min, 90), add: 0 };
}

/** Odd americana ("+245", "-150", "EVEN") -> decimal. */
export function americanToDecimal(a) {
  if (a == null) return null;
  if (typeof a === 'string') { if (/even/i.test(a)) return 2; a = parseFloat(a.replace('+', '')); }
  if (!Number.isFinite(a) || a === 0) return null;
  return a > 0 ? 1 + a / 100 : 1 + 100 / -a;
}
/** Odds 1X2 de um item de odds da ESPN (scoreboard ou core API) -> {dec:[casa,empate,fora], src}. */
export function parseOdds(o) {
  if (!o) return null;
  const ml = o.moneyline || {};
  const pick = (side, legacy) => americanToDecimal(ml[side]?.close?.odds ?? ml[side]?.open?.odds ?? legacy?.moneyLine);
  const h = pick('home', o.homeTeamOdds), d = pick('draw', o.drawOdds), a = pick('away', o.awayTeamOdds);
  if (!(h > 1 && d > 1 && a > 1)) return null;
  // total de gols (mais/menos), quando a casa publica
  const ov = americanToDecimal(o.total?.over?.close?.odds ?? o.total?.over?.open?.odds), un = americanToDecimal(o.total?.under?.close?.odds ?? o.total?.under?.open?.odds);
  const line = parseFloat(o.overUnder ?? String(o.total?.over?.close?.line || '').replace(/^o/, ''));
  const tot = ov > 1 && un > 1 && Number.isFinite(line) ? { line, o: +ov.toFixed(3), u: +un.toFixed(3) } : null;
  return { dec: [+h.toFixed(3), +d.toFixed(3), +a.toFixed(3)], src: o.provider?.name || 'ESPN', ...(tot ? { tot } : {}) };
}
/** Probabilidades implícitas (sem a margem da casa), a partir das odds decimais. */
export function impliedProbs(dec) {
  const inv = dec.map((v) => 1 / v), s = inv[0] + inv[1] + inv[2];
  return { pH: inv[0] / s, pD: inv[1] / s, pA: inv[2] / s, margem: s - 1 };
}

export function statusOf(st) {
  const name = st?.type?.name || '';
  const state = st?.type?.state || 'pre';
  if (name === 'STATUS_POSTPONED' || name === 'STATUS_CANCELED' || name === 'STATUS_ABANDONED' || name === 'STATUS_SUSPENDED') return 'off';
  if (state === 'post') return 'post';
  if (state === 'in') return 'in';
  return 'pre';
}

function stat(list, name) {
  const s = (list || []).find((x) => x.name === name);
  if (!s) return null;
  const v = parseFloat(s.displayValue);
  return Number.isFinite(v) ? v : null;
}

/** Converte um evento do scoreboard em um registro compacto de partida. */
export function parseEvent(e, league = 'A') {
  const c = e.competitions?.[0];
  if (!c) return null;
  const home = c.competitors.find((x) => x.homeAway === 'home');
  const away = c.competitors.find((x) => x.homeAway === 'away');
  if (!home || !away) return null;
  const status = statusOf(c.status || e.status);
  const ev = [];
  for (const d of c.details || []) {
    const t = d.type?.text || '';
    let type = null;
    if (GOAL_TYPES.has(t) || d.scoringPlay) type = d.ownGoal || t === 'Own Goal' ? 'O' : d.penaltyKick || t === 'Penalty - Scored' ? 'P' : 'G';
    else if (d.redCard || t === 'Red Card') type = 'R';
    else if (d.yellowCard || t === 'Yellow Card') type = 'Y';
    if (!type || d.shootout) continue;
    const { min, add } = parseClock(d.clock?.displayValue, d.clock?.value);
    const side = d.team?.id === home.team.id ? 'h' : d.team?.id === away.team.id ? 'a' : null;
    if (!side) continue;
    const p = d.athletesInvolved?.[0];
    ev.push({ m: min, x: add, t: type, s: side, p: p?.id || null, n: p?.displayName || null, pos: p?.position || null, k: t });
  }
  ev.sort((a, b) => (a.m > 45 ? 1 : 0) - (b.m > 45 ? 1 : 0) || a.m - b.m || a.x - b.x);
  const hs = home.statistics, as = away.statistics;
  const stats = stat(hs, 'totalShots') == null ? null : {
    h: { sh: stat(hs, 'totalShots'), sot: stat(hs, 'shotsOnTarget'), co: stat(hs, 'wonCorners'), fo: stat(hs, 'foulsCommitted'), po: stat(hs, 'possessionPct') },
    a: { sh: stat(as, 'totalShots'), sot: stat(as, 'shotsOnTarget'), co: stat(as, 'wonCorners'), fo: stat(as, 'foulsCommitted'), po: stat(as, 'possessionPct') },
  };
  const st = c.status || e.status || {};
  return {
    id: String(e.id),
    lg: league,
    season: e.season?.year ?? new Date(e.date).getUTCFullYear(),
    date: e.date,
    status,
    statusName: st.type?.name || '',
    clock: st.displayClock || '',
    period: st.period || 0,
    h: String(home.team.id),
    a: String(away.team.id),
    hg: status === 'pre' || status === 'off' ? null : parseInt(home.score ?? '0', 10),
    ag: status === 'pre' || status === 'off' ? null : parseInt(away.score ?? '0', 10),
    venue: c.venue?.fullName || e.venue?.displayName || null,
    city: c.venue?.address?.city || null,
    att: c.attendance || null,
    ev,
    stats,
    odds: parseOdds(c.odds?.[0]),
    // estatísticas confiáveis: posse somando ~100 e volume plausível de finalizações
    statsOk: !!stats && stats.h.po != null && stats.a.po != null && Math.abs(stats.h.po + stats.a.po - 100) <= 1.5 && stats.h.sh + stats.a.sh >= 6 && stats.h.sh >= stats.h.sot && stats.a.sh >= stats.a.sot,
  };
}

export function parseTeam(comp) {
  const t = comp.team;
  return {
    id: String(t.id),
    name: t.shortDisplayName || t.displayName || t.name,
    full: t.displayName || t.name,
    abbr: t.abbreviation || (t.displayName || '').slice(0, 3).toUpperCase(),
    color: t.color || null,
    alt: t.alternateColor || null,
    logo: t.logo || `https://a.espncdn.com/i/teamlogos/soccer/500/${t.id}.png`,
  };
}

/** Scoreboard inteiro -> {matches, teams} */
export function parseScoreboard(json, league = 'A') {
  const matches = [];
  const teams = {};
  for (const e of json?.events || []) {
    const m = parseEvent(e, league);
    if (!m) continue;
    matches.push(m);
    for (const comp of e.competitions[0].competitors) {
      const t = parseTeam(comp);
      teams[t.id] = t;
    }
  }
  return { matches, teams, seasonYear: json?.leagues?.[0]?.season?.year ?? null };
}
