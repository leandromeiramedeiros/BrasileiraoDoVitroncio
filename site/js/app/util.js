// Utilitários de interface: formatação pt-BR, escape de HTML, pequenos helpers.
export const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
export const pct = (p, d = 0) => (p == null || Number.isNaN(p) ? '–' : p > 0 && p < 0.005 && d === 0 ? '<1%' : p < 1 && p > 0.995 && d === 0 ? '>99%' : `${(p * 100).toFixed(d)}%`);
export const num = (v, d = 1) => (v == null || Number.isNaN(v) ? '–' : v.toFixed(d).replace('.', ','));
const fDay = new Intl.DateTimeFormat('pt-BR', { weekday: 'long', day: '2-digit', month: 'long' });
const fShort = new Intl.DateTimeFormat('pt-BR', { weekday: 'short', day: '2-digit', month: '2-digit' });
const fTime = new Intl.DateTimeFormat('pt-BR', { hour: '2-digit', minute: '2-digit' });
const fFull = new Intl.DateTimeFormat('pt-BR', { day: '2-digit', month: '2-digit', year: 'numeric', hour: '2-digit', minute: '2-digit' });
export const dayKey = (iso) => { const d = new Date(iso); return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`; };
export const fmtDay = (iso) => { const s = fDay.format(new Date(iso)); return s.charAt(0).toUpperCase() + s.slice(1); };
export const fmtShort = (iso) => fShort.format(new Date(iso)).replace('.', '');
export const fmtTime = (iso) => fTime.format(new Date(iso));
export const fmtFull = (iso) => fFull.format(new Date(iso));
export function relDay(iso) {
  const today = dayKey(new Date().toISOString()), k = dayKey(iso);
  const t = new Date(); t.setDate(t.getDate() + 1); const tom = dayKey(t.toISOString());
  const y = new Date(); y.setDate(y.getDate() - 1); const yes = dayKey(y.toISOString());
  return k === today ? 'Hoje' : k === tom ? 'Amanhã' : k === yes ? 'Ontem' : fmtDay(iso);
}
export function countdown(iso) {
  const ms = new Date(iso) - Date.now();
  if (ms <= 0) return 'a qualquer momento';
  const m = Math.floor(ms / 60000), d = Math.floor(m / 1440), h = Math.floor((m % 1440) / 60), mm = m % 60;
  return d > 0 ? `em ${d}d ${h}h` : h > 0 ? `em ${h}h ${String(mm).padStart(2, '0')}min` : `em ${mm} min`;
}
export const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
export const store = {
  get(k, d = null) { try { const v = localStorage.getItem(`bdv:${k}`); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(`bdv:${k}`, JSON.stringify(v)); } catch { /* sem armazenamento */ } },
};
/** Cor legível do time sobre fundo escuro. */
export function teamColor(t) {
  const pickc = (hex) => { if (!hex) return null; const n = parseInt(hex, 16); const r = (n >> 16) & 255, g = (n >> 8) & 255, b = n & 255; const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b; return { hex: `#${hex}`, lum }; };
  const a = pickc(t?.color), b = pickc(t?.alt);
  if (a && a.lum > 60) return a.hex;
  if (b && b.lum > 60) return b.hex;
  return a && a.lum > 25 ? a.hex : '#9fb3c8';
}
