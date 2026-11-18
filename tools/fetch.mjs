// Baixa os "scoreboards" anuais da ESPN (Série A e Série B) para model/cache.
// Temporadas encerradas são reaproveitadas do cache; a temporada atual (e a
// anterior, por causa de jogos remarcados) é sempre baixada de novo.
//   node tools/fetch.mjs            -> só o que falta + ano atual
//   node tools/fetch.mjs --all      -> rebaixa tudo
import { mkdir, writeFile, access } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
const run = promisify(execFile);

const CACHE = new URL('../model/cache/', import.meta.url);
const FIRST = { 'bra.1': 2006, 'bra.2': 2008 };
const now = new Date();
const thisYear = now.getUTCFullYear();
const all = process.argv.includes('--all');

async function exists(u) { try { await access(u); return true; } catch { return false; } }

// curl respeita o proxy/CA do ambiente (o fetch do Node não), por isso é usado aqui.
async function download(url, dest) {
  for (let i = 0; i < 4; i++) {
    try {
      await run('curl', ['-sS', '--fail', '--max-time', '240', '-o', dest, url]);
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
    await download(url, j.dest.pathname);
    console.log(`  ok ${j.lg} ${j.y}`);
  }
}
await Promise.all([worker(), worker(), worker()]);
console.log('pronto.');
