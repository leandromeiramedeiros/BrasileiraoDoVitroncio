/** Nelder–Mead simples (minimização), com reinícios opcionais. */
export function nelderMead(f, x0, { step = 0.1, maxIter = 400, tol = 1e-7, steps = null } = {}) {
  const n = x0.length;
  let simplex = [x0.slice()];
  for (let i = 0; i < n; i++) { const x = x0.slice(); x[i] += steps ? steps[i] : (Math.abs(x[i]) > 1e-8 ? step * Math.abs(x[i]) : step * 0.01); simplex.push(x); }
  let vals = simplex.map(f);
  let evals = n + 1;
  for (let it = 0; it < maxIter; it++) {
    const idx = vals.map((v, i) => i).sort((a, b) => vals[a] - vals[b]);
    simplex = idx.map((i) => simplex[i]); vals = idx.map((i) => vals[i]);
    if (Math.abs(vals[n] - vals[0]) < tol) break;
    const c = new Array(n).fill(0);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) c[j] += simplex[i][j] / n;
    const at = (t) => c.map((cj, j) => cj + t * (simplex[n][j] - cj));
    const xr = at(-1), fr = f(xr); evals++;
    if (fr < vals[0]) {
      const xe = at(-2), fe = f(xe); evals++;
      if (fe < fr) { simplex[n] = xe; vals[n] = fe; } else { simplex[n] = xr; vals[n] = fr; }
    } else if (fr < vals[n - 1]) { simplex[n] = xr; vals[n] = fr; }
    else {
      const xc = at(fr < vals[n] ? -0.5 : 0.5), fc = f(xc); evals++;
      if (fc < Math.min(fr, vals[n])) { simplex[n] = xc; vals[n] = fc; }
      else { for (let i = 1; i <= n; i++) { simplex[i] = simplex[i].map((v, j) => simplex[0][j] + 0.5 * (v - simplex[0][j])); vals[i] = f(simplex[i]); evals++; } }
    }
  }
  let b = 0; for (let i = 1; i <= n; i++) if (vals[i] < vals[b]) b = i;
  return { x: simplex[b], f: vals[b], evals };
}

/** Resolve A x = b (eliminação de Gauss com pivoteamento parcial). */
export function solve(A, b) {
  const n = b.length; const M = A.map((r, i) => [...r, b[i]]);
  for (let i = 0; i < n; i++) {
    let p = i; for (let r = i + 1; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
    [M[i], M[p]] = [M[p], M[i]];
    const d = M[i][i] || 1e-12;
    for (let j = i; j <= n; j++) M[i][j] /= d;
    for (let r = 0; r < n; r++) if (r !== i) { const f = M[r][i]; if (f) for (let j = i; j <= n; j++) M[r][j] -= f * M[i][j]; }
  }
  return M.map((r) => r[n]);
}

/** Mínimos quadrados ordinários: X (n×k), y (n) -> beta (k). */
export function ols(X, y) {
  const k = X[0].length; const A = Array.from({ length: k }, () => new Array(k).fill(0)); const b = new Array(k).fill(0);
  for (let i = 0; i < X.length; i++) { const x = X[i]; for (let a = 0; a < k; a++) { b[a] += x[a] * y[i]; for (let c = 0; c < k; c++) A[a][c] += x[a] * x[c]; } }
  return solve(A, b);
}

/** GLM de Poisson (IRLS) com offset e ridge leve: X (n×k), y, offset -> beta. */
export function poissonGLM(X, y, offset, { iters = 40, ridge = 1e-6, weights = null } = {}) {
  const n = X.length, k = X[0].length; let beta = new Array(k).fill(0);
  for (let it = 0; it < iters; it++) {
    const A = Array.from({ length: k }, (_, i) => { const r = new Array(k).fill(0); r[i] = ridge; return r; }); const g = new Array(k).fill(0);
    for (let i = 0; i < n; i++) {
      const x = X[i]; let eta = offset ? offset[i] : 0; for (let a = 0; a < k; a++) eta += x[a] * beta[a];
      const w = weights ? weights[i] : 1; const mu = Math.exp(eta) ;
      for (let a = 0; a < k; a++) { if (!x[a]) continue; g[a] += w * x[a] * (y[i] - mu); for (let c = 0; c < k; c++) if (x[c]) A[a][c] += w * mu * x[a] * x[c]; }
    }
    for (let a = 0; a < k; a++) g[a] -= ridge * beta[a];
    const d = solve(A, g); let mx = 0;
    for (let a = 0; a < k; a++) { beta[a] += d[a]; mx = Math.max(mx, Math.abs(d[a])); }
    if (mx < 1e-9) break;
  }
  return beta;
}
