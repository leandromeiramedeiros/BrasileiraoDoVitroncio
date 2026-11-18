/** Gerador pseudoaleatório com semente (mulberry32) + utilitários. */
export function makeRng(seed = Date.now() >>> 0) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.int = (n) => Math.floor(next() * n);
  next.pick = (arr) => arr[Math.floor(next() * arr.length)];
  /** índice sorteado proporcionalmente aos pesos */
  next.weighted = (weights) => {
    let s = 0; for (const w of weights) s += w;
    if (!(s > 0)) return Math.floor(next() * weights.length);
    let u = next() * s;
    for (let i = 0; i < weights.length; i++) { u -= weights[i]; if (u <= 0) return i; }
    return weights.length - 1;
  };
  next.poisson = (lambda) => {
    if (lambda <= 0) return 0;
    const L = Math.exp(-lambda); let k = 0, p = 1;
    do { k++; p *= next(); } while (p > L);
    return k - 1;
  };
  return next;
}

export function hashSeed(str) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
  return h >>> 0;
}
