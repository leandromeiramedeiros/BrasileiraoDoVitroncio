// Atualização incremental do DOM: em vez de trocar o HTML inteiro a cada
// mudança (o que recarrega imagens e reinicia animações — "pisca"), compara a
// árvore nova com a atual e mexe só no que mudou.
//  - filhos com data-mk são casados pela chave (listas que crescem no topo);
//  - elementos com data-enter ganham uma entrada suave quando são inseridos;
//  - elementos com data-flash piscam quando o texto muda (placar).

let animate = false;

function same(a, b) { return a.nodeType === b.nodeType && (a.nodeType !== 1 || a.tagName === b.tagName); }

function syncAttrs(o, n) {
  for (const a of [...o.attributes]) if (!n.hasAttribute(a.name)) o.removeAttribute(a.name);
  for (const a of n.attributes) if (o.getAttribute(a.name) !== a.value) o.setAttribute(a.name, a.value);
}

function syncNode(o, n) {
  if (o.nodeType !== 1) {
    if (o.nodeValue !== n.nodeValue) {
      o.nodeValue = n.nodeValue;
      const p = o.parentElement;
      if (animate && p?.hasAttribute('data-flash') && p.animate) p.animate([{ transform: 'scale(1.35)', color: '#ffd21f' }, { transform: 'scale(1)' }], { duration: 700, easing: 'ease-out' });
    }
    return;
  }
  syncAttrs(o, n);
  syncChildren(o, n);
  if (o.tagName === 'SELECT') { const sel = [...o.options].find((x) => x.hasAttribute('selected')); if (sel && o.value !== sel.value) o.value = sel.value; }
}

function syncChildren(oldP, newP) {
  const keyed = new Map();
  for (const c of oldP.children) { const k = c.getAttribute('data-mk'); if (k) keyed.set(k, c); }
  let cur = oldP.firstChild;
  for (const n of [...newP.childNodes]) {
    const key = n.nodeType === 1 ? n.getAttribute('data-mk') : null;
    let match = null;
    if (key) { if (keyed.has(key) && same(keyed.get(key), n)) { match = keyed.get(key); keyed.delete(key); } }
    else if (cur && same(cur, n) && !(cur.nodeType === 1 && cur.hasAttribute('data-mk'))) match = cur;
    if (match) {
      if (match === cur) cur = cur.nextSibling; else oldP.insertBefore(match, cur);
      syncNode(match, n);
    } else {
      oldP.insertBefore(n, cur);
      if (animate && n.nodeType === 1 && n.hasAttribute('data-enter') && n.animate) n.animate([{ opacity: 0, transform: 'translateY(-6px)' }, { opacity: 1, transform: 'none' }], { duration: 280, easing: 'ease-out' });
    }
  }
  while (cur) { const nx = cur.nextSibling; oldP.removeChild(cur); cur = nx; }
}

/** Atualiza `root` para refletir `html`. `withAnimations=false` na troca de página. */
export function morph(root, html, withAnimations = true) {
  const tpl = document.createElement('template');
  tpl.innerHTML = html;
  animate = withAnimations && root.childNodes.length > 0;
  syncChildren(root, tpl.content);
  animate = false;
}
