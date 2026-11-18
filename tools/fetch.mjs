// Baixa os "scoreboards" anuais da ESPN (Série A, Série B e taças continentais)
// para model/cache. As taças entram só como calendário (quem joga quando).
// Temporadas encerradas são reaproveitadas do cache; a temporada atual (e a
// anterior, por causa de jogos remarcados) é sempre baixada de novo.
//   node tools/fetch.mjs            -> só o que falta + ano atual
//   node tools/fetch.mjs --all      -> rebaixa tudo
import { mkdir, writeFile, access } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const run = promisify(execFile);

const CACHE = new URL('../model/cache/', import.meta.url);
const FIRST = { 'bra.1': 2006, 'bra.2': 2008, 'conmebol.libertadores': 2006, 'conmebol.sudamericana': 2006, 'conmebol.recopa': 2006, 'fifa.cwc': 2006, 'fifa.intercontinental_cup': 2006, 'bra.supercopa_do_brazil': 2006 };
const now = new Date();
const thisYear = now.getUTCFullYear();
const all = process.argv.includes('--all');

async function exists(u) { try { await access(u); return true; } catch { return false; } }

// curl respeita o proxy/CA do ambiente (o fetch do Node não), por isso é usado aqui.
async function download(url, dest) {
  for (let i = 0; i < 4; i++) {
    try {
      await run('curl', ['-sSL', '--fail', '--max-time', '240', '-o', dest, url]);
      return;
    } catch (e) {
      if (i === 3) throw e;
      await new Promise((r) => setTimeout(r, 2000 * 2 ** i));
    }
  }
}

await mkdir(CACHE, { recursive: true });
const jobs = [];
for (const lg of Object.keys(FIRST)) {
  for (let y = FIRST[lg]; y <= thisYear + (now.getUTCMonth() === 11 ? 1 : 0); y++) {
    const dest = new URL(`${lg}_${y}.json`, CACHE);
    if (!all && y < thisYear - 1 && (await exists(dest))) continue;
    jobs.push({ lg, y, dest });
  }
}
console.log(`baixando ${jobs.length} arquivo(s)…`);
let i = 0;
async function worker() {
  while (i < jobs.length) {
    const j = jobs[i++];
    const url = `https://site.web.api.espn.com/apis/site/v2/sports/soccer/${j.lg}/scoreboard?dates=${j.y}&limit=700`;
    try { await download(url, j.dest.pathname); console.log(`  ok ${j.lg} ${j.y}`); }
    catch (e) { if (j.lg.startsWith('bra.1') || j.lg.startsWith('bra.2')) throw e; console.log(`  (sem ${j.lg} ${j.y})`); }
  }
}
await Promise.all([worker(), worker(), worker()]);
// referência de mercado para o backtest (opcional; nunca aparece no site)
try { await download('https://www.football-data.co.uk/new/BRA.csv', new URL('BRA.csv', CACHE).pathname); console.log('  ok referência de mercado'); } catch { console.log('  (referência de mercado indisponível — o backtest segue sem ela)'); }
console.log('pronto.');
