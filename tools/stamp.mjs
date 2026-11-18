// Carimba a versão em todos os endereços de arquivos do site (só na cópia
// publicada, dentro do workflow). O GitHub Pages deixa o navegador guardar
// cada arquivo por 10 minutos; sem o carimbo, quem estava com o site aberto
// continuava rodando scripts antigos depois de uma publicação.
//   node tools/stamp.mjs site <versão>
import { readdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const [dir, version] = process.argv.slice(2);
if (!dir || !version) { console.error('uso: node tools/stamp.mjs <pasta> <versão>'); process.exit(1); }
const v = version.slice(0, 10);

async function walk(d) {
  const out = [];
  for (const e of await readdir(d, { withFileTypes: true })) {
    const p = join(d, e.name);
    if (e.isDirectory()) out.push(...(await walk(p))); else out.push(p);
  }
  return out;
}

let n = 0;
for (const f of await walk(dir)) {
  if (f.endsWith('.js')) {
    const src = await readFile(f, 'utf8');
    const out = src
      .replace(/(\bfrom\s*['"])(\.{1,2}\/[^'"?]+\.js)(['"])/g, `$1$2?v=${v}$3`)
      .replace(/(\bimport\s*['"])(\.{1,2}\/[^'"?]+\.js)(['"])/g, `$1$2?v=${v}$3`)
      .replace(/(new URL\(\s*['"])(\.{1,2}\/[^'"?]+\.js)(['"])/g, `$1$2?v=${v}$3`)
      .replaceAll('__BUILD__', v);
    if (out !== src) { await writeFile(f, out); n++; }
  } else if (f.endsWith('index.html')) {
    const src = await readFile(f, 'utf8');
    const out = src.replace(/(href="css\/app\.css)(")/, `$1?v=${v}$2`).replace(/(src="js\/app\/main\.js)(")/, `$1?v=${v}$2`);
    if (out !== src) { await writeFile(f, out); n++; }
  }
}
await writeFile(join(dir, 'version.txt'), v);
console.log(`versão ${v} carimbada em ${n} arquivo(s)`);
