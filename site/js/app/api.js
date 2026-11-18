// Acesso aos dados dinâmicos (ESPN, público, sem chave). Tudo em pt-BR.
const SITE = 'https://site.api.espn.com/apis/site/v2/sports/soccer/bra.1';
const Q = 'lang=pt&region=br';

async function getJSON(url, timeout = 20000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeout);
  try {
    const r = await fetch(url, { signal: ctl.signal, cache: 'no-store' });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  } finally { clearTimeout(t); }
}

/** Jogos do dia (ou de uma data/mês/ano: '20261118', '202611', '2026'). */
export const scoreboard = (dates = null) => getJSON(`${SITE}/scoreboard?${Q}${dates ? `&dates=${dates}&limit=500` : ''}`, dates && dates.length === 4 ? 45000 : 20000);
/** Detalhes de um jogo: lances, estatísticas, escalações. */
export const summary = (id) => getJSON(`${SITE}/summary?event=${id}&${Q}`);
