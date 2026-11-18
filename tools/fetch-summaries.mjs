// Baixa o "summary" (escalações, substituições, estatísticas por jogador) de
// cada partida encerrada da Série A na temporada atual. Cache em model/cache/sum.
import { mkdir, access, readFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { parseScoreboard } from '../site/js/data/espn-parse.js';
const run = promisify(execFile);
const CACHE = new URL('../model/cache/', import.meta.url);
const SUM = new URL('sum/', CACHE);
await mkdir(SUM, { recursive: true });
const year = +(process.argv[2] || new Date().getUTCFullYear());
const sb = JSON.parse(await readFile(new URL(`bra.1_${year}.json`, CACHE), 'utf8'));
const { matches } = parseScoreboard(sb, 'A');
const todo = [];
for (const m of matches) {
  if (m.status !== 'post' || m.season !== year) continue;
  const dest = new URL(`${m.id}.json`, SUM);
  try { await access(dest); } catch { todo.push({ id: m.id, dest }); }
}
console.log(`summaries a baixar: ${todo.length}`);
let i = 0, fail = 0;
async function worker() {
  while (i < todo.length) {
    const j = todo[i++];
    try { await run('curl', ['-sS', '--fail', '--max-time', '90', '--retry', '3', '-o', j.dest.pathname, `https://site.api.espn.com/apis/site/v2/sports/soccer/bra.1/summary?event=${j.id}`]); }
    catch { fail++; }
  }
}
await Promise.all([worker(), worker(), worker(), worker()]);
console.log(`pronto (falhas: ${fail}).`);
