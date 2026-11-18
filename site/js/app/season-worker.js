// Web Worker: roda a simulação da temporada fora da thread da interface.
import { simulateSeason } from '../engine/season.js';
self.onmessage = (ev) => {
  const { hz, ids, table, fixtures, sigma, fixed, n, seed } = ev.data;
  const t0 = performance.now();
  const res = simulateSeason(hz, ids, table, fixtures, { n, sigma, seed, fixed });
  res.ms = Math.round(performance.now() - t0);
  res.at = Date.now();
  res.fixed = fixed;
  self.postMessage(res);
};
