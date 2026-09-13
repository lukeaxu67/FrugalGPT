/* =====================================================================
 * FrugalGPT τ 求解演示 —— 核心算法（忠实移植 src/FrugalGPT/optimizer.py）
 *
 * 移植对照（行号为仓库代码位置）：
 *   construct_data          → constructMatrices        (optimizer.py:28)
 *   compute_dist            → D = 1 - score            (optimizer.py:9)
 *   optimize/f (delta)      → simulate + objective     (optimizer.py:58-84)
 *   quatile2thres(_batch)   → tauFromQual（含 alive 掩码）(optimizer.py:95-112)
 *   g(qual) 单调约束罚      → objective 中的 u2<u1 判定  (optimizer.py:87)
 *   scipy.optimize.brute    → gridSearch (Ns=40)       (optimizer.py:124)
 *   scipy.optimize.fmin     → nelderMead（scipy 同款默认参数）
 *
 * 同时可在 Node 与浏览器中运行（用于自动化验证与页面渲染）。
 * ===================================================================== */
(function (root, factory) {
  if (typeof module !== "undefined" && module.exports) module.exports = factory();
  else root.FrugalCore = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  /* ---------------- 随机数（可复现） ---------------- */
  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a |= 0; a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function makeGauss(rand) {
    let spare = null;
    return function () {
      if (spare !== null) { const v = spare; spare = null; return v; }
      let u = 0, v = 0;
      while (u === 0) u = rand();
      while (v === 0) v = rand();
      const r = Math.sqrt(-2 * Math.log(u)), th = 2 * Math.PI * v;
      spare = r * Math.sin(th);
      return r * Math.cos(th);
    };
  }

  /* ---------------- API 档案（成本量级对齐论文 Table 1） ---------------- */
  const APIS = [
    { key: "A", name: "gpt-4o-mini", costMean: 0.001 },
    { key: "B", name: "gemma-2-9b",  costMean: 0.005 },
    { key: "C", name: "gpt-4-turbo", costMean: 0.030 },
  ];

  /* 难度分带（代表性结构：级联的每一级主要服务一个难度带）
   *  简单(68%)：A 就能答对，理想情况在第一级早停
   *  中等(15%)：A 答错、B 能救，第二级接管
   *  困难(12%)：只有 C 靠谱，落到末级
   *  极难( 5%)：C 也常常错（数据里的不可约误差）
   */
  const BANDS = [
    { name: "简单", prob: 0.68, p: [0.93, 0.95, 0.97] },
    { name: "中等", prob: 0.15, p: [0.25, 0.85, 0.96] },
    { name: "困难", prob: 0.12, p: [0.05, 0.30, 0.92] },
    { name: "极难", prob: 0.05, p: [0.03, 0.08, 0.45] },
  ];
  const SCORE_NOISE = 0.08; // 打分器不完美：score = p + 噪声（对标 DistilBERT 打分器的真实水准）

  /* ---------------- 数据生成 ----------------
   * 每条查询：
   *   band_i     难度带（按概率抽取）
   *   p_ij       第 j 个 API 答对的真实概率 = 带内基线 ± 抖动
   *   correct_ij 真实是否答对 ~ Bernoulli(p_ij)        → 矩阵 R
   *   score_ij   打分器输出 = p_ij + 噪声 ∈ (0,1)      → 矩阵 D = 1 - score
   *   cost_ij    单次调用成本（按答案长度微扰）          → 累计成 C
   * 关键代表性：score 与 correct 相关但不完美（存在"答错却高分"的查询）。
   */
  function generateDataset(n, seed) {
    n = n || 200;
    const seedUsed = (seed === undefined) ? 20260913 : seed;
    const rand = mulberry32(seedUsed);
    const gauss = makeGauss(rand);
    const rows = [];
    for (let i = 0; i < n; i++) {
      const u = rand();
      let acc = 0, band = BANDS.length - 1;
      for (let k = 0; k < BANDS.length; k++) {
        acc += BANDS[k].prob;
        if (u < acc) { band = k; break; }
      }
      const per = APIS.map(function (api, j) {
        const p = clamp(BANDS[band].p[j] + 0.05 * gauss(), 0.01, 0.99);
        const correct = rand() < p ? 1 : 0;
        const score = clamp(p + SCORE_NOISE * gauss(), 0.01, 0.99);
        const cost = api.costMean * (0.7 + 0.6 * rand());
        return { p: p, correct: correct, score: score, cost: cost };
      });
      rows.push({ id: i, band: band, per: per });
    }
    return { n: n, apis: APIS.map(function (a) { return a; }),
             bands: BANDS, rows: rows, seed: seedUsed };
  }

  function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }

  /* ---------------- construct_data 移植 ----------------
   * R[i][j] = 正确性; C[i][j] = 前缀累计成本; D[i][j] = 1 - 前缀末位打分
   * （compute_dist 取 scores[-1]，即前缀最后一个 API 的打分）
   */
  function constructMatrices(ds) {
    const n = ds.n, m = ds.apis.length;
    const R = [], C = [], D = [];
    for (let i = 0; i < n; i++) {
      const row = ds.rows[i].per;
      R.push(row.map(function (x) { return x.correct; }));
      const c = [], d = [];
      let cum = 0;
      for (let j = 0; j < m; j++) {
        cum += row[j].cost;
        c.push(cum);
        d.push(1 - row[j].score);   // 前缀末位 = 第 j 个 API（顺序链）
      }
      C.push(c); D.push(d);
    }
    return { R: R, C: C, D: D, n: n, m: m };
  }

  /* ---------------- numpy 'linear' 分位数 ---------------- */
  function quantileLinear(values, q) {
    const a = values.slice().sort(function (x, y) { return x - y; });
    const nn = a.length;
    if (nn === 0) throw new Error("quantile of empty array");
    q = clamp(q, 0, 1);
    const h = (nn - 1) * q;
    const lo = Math.floor(h), hi = Math.min(lo + 1, nn - 1);
    return a[lo] + (h - lo) * (a[hi] - a[lo]);
  }

  /* ---------------- quatile2thres_batch 移植（含 alive 掩码） ----------------
   * 输入 u = (u1, u2)，末级阈值恒为 1；返回 tau 与各级 alive 掩码。
   * tau[j] = Quantile(D[alive_j, j], 1 - u_j)（q 先按代码 clamp 到 [0,1]）
   */
  function tauFromQual(u1, u2, D, m) {
    m = m || 3;
    const qual = [u1, u2];
    const tau = new Array(m).fill(0);
    tau[m - 1] = 1;                                  // 末级硬编码（Q1）
    const alive = [];
    let mask = [];
    for (let i = 0; i < D.length; i++) mask.push(true);
    alive.push(mask.slice());
    for (let j = 0; j < m - 1; j++) {
      const qj = clamp(qual[j], 0, 1);               // optimizer.py:98 的 clamp
      const data = [];
      for (let i = 0; i < D.length; i++) if (mask[i]) data.push(D[i][j]);
      const t = quantileLinear(data, 1 - qj);
      tau[j] = t;
      const nxt = [];
      for (let i = 0; i < D.length; i++) {
        mask[i] = mask[i] && (D[i][j] >= t);
        nxt.push(mask[i]);
      }
      alive.push(nxt.slice());
    }
    return { tau: tau, alive: alive };
  }

  /* ---------------- f(delta) 移植：给定 tau 模拟级联 ----------------
   * 接受 ⟺ D < tau；每行只保留第一个接受（首站停）；末级 tau=1 兜底。
   * 返回接受矩阵、停机级、acc、cost、是否可行（cost <= budget*n）。
   */
  function simulate(R, C, D, tau, budget) {
    const n = R.length, m = R[0].length;
    const A = [], z = [];
    let acc = 0, cost = 0;
    for (let i = 0; i < n; i++) {
      const row = [0, 0, 0].slice(0, m);
      let zi = m - 1, hit = false;
      for (let j = 0; j < m; j++) {
        if (!hit && D[i][j] < tau[j]) { row[j] = 1; zi = j; hit = true; }
      }
      // 末级兜底：tau[m-1]=1 且 D<1 几乎必真；若整行无接受（D=1 理论角点），
      // 与仓库 f() 一致：该行计 0 正确、0 成本（静默丢弃）。
      A.push(row); z.push(hit ? zi : -1);
      acc += R[i][zi] * (hit ? 1 : 0);
      cost += C[i][zi] * (hit ? 1 : 0);
    }
    const budgetTotal = budget * n;
    const feasible = cost <= budgetTotal;
    return { A: A, z: z, acc: acc, cost: cost, feasible: feasible,
             accRate: acc / n, costAvg: cost / n };
  }

  /* ---------------- g(qual) 移植：优化目标 ----------------
   * 优化变量 qual=(u1,u2)；u2<u1 返罚 10000（optimizer.py:87）；
   * cost 超预算返罚 10000；否则返 -acc（转最小化）。
   * 这里额外返回细节供可视化使用。
   */
  function objective(R, C, D, budget, u1, u2) {
    if (u2 < u1) return { f: 10000, penalty: "mono" };
    const tt = tauFromQual(u1, u2, D);
    const sim = simulate(R, C, D, tt.tau, budget);
    if (!sim.feasible) return { f: 10000, penalty: "budget",
                                tau: tt.tau, sim: sim };
    return { f: -sim.acc, penalty: null, tau: tt.tau, sim: sim };
  }

  /* ---------------- scipy.optimize.brute 移植 ----------------
   * 网格 = linspace(1e-5, 1-1e-5, Ns)，逐点评估并记录全部信息。
   */
  function gridSearch(R, C, D, budget, Ns) {
    Ns = Ns || 40;
    const lo = 1e-5, hi = 1 - 1e-5;
    const axis = [];
    for (let k = 0; k < Ns; k++) axis.push(lo + k * (hi - lo) / (Ns - 1));
    const points = [];
    let best = null;
    for (let a = 0; a < Ns; a++) {
      for (let b = 0; b < Ns; b++) {
        const u1 = axis[a], u2 = axis[b];
        const o = objective(R, C, D, budget, u1, u2);
        const pt = { u1: u1, u2: u2, f: o.f, penalty: o.penalty,
                     acc: o.sim ? o.sim.accRate : null,
                     cost: o.sim ? o.sim.costAvg : null,
                     tau: o.tau || null };
        points.push(pt);
        if (o.penalty === null && (best === null || o.f < best.f)) best = pt;
      }
    }
    return { axis: axis, points: points, best: best };
  }

  /* ---------------- scipy.optimize.fmin (Nelder-Mead) 移植 ----------------
   * scipy 默认：xatol=fatol=1e-4，nonzdelt=0.05，zdelt=0.00025，maxiter=maxfev=N*200。
   * 初始单纯形：sim[0]=x0；sim[k+1]=x0 且第 k 维 *1.05（为 0 时 +0.00025）。
   * 返回迭代日志供动画使用。
   */
  function nelderMead(fun, x0, opts) {
    opts = opts || {};
    const xatol = opts.xatol !== undefined ? opts.xatol : 1e-4;
    const fatol = opts.fatol !== undefined ? opts.fatol : 1e-4;
    const maxIter = opts.maxIter !== undefined ? opts.maxIter : 400;
    const maxFun = opts.maxFun !== undefined ? opts.maxFun : 400;
    const nonzdelt = 0.05, zdelt = 0.00025;

    const n = x0.length;
    let sim = [];
    sim.push(x0.slice());
    for (let k = 0; k < n; k++) {
      const y = x0.slice();
      y[k] = y[k] !== 0 ? (1 + nonzdelt) * y[k] : zdelt;
      sim.push(y);
    }
    let fv = sim.map(function (v) { return fun(v); });
    let nfev = n + 1;
    const iterations = [];
    iterations.push({ action: "init", vertices: sim.map(v => v.slice()),
                      fvals: fv.slice(), nfev: nfev });

    for (let it = 0; it < maxIter; it++) {
      // 排序（升序）
      const order = fv.map(function (v, i) { return i; })
                      .sort(function (a, b) { return fv[a] - fv[b]; });
      sim = order.map(function (i) { return sim[i]; });
      fv = order.map(function (i) { return fv[i]; });

      if (maxAbsDiff(sim.slice(1)) <= xatol && maxAbsDiffArr(fv.slice(1)) <= fatol) break;
      if (nfev >= maxFun) break;

      const centroid = new Array(n).fill(0);
      for (let k = 0; k < n; k++) {
        for (let i = 0; i < n; i++) centroid[k] += sim[i][k] / n;
      }
      // 反射
      let xr = centroid.map(function (c, k) { return c + (c - sim[n][k]); });
      let fr = fun(xr); nfev++;
      let action = "reflect", xrAccepted = null;
      if (fr < fv[0]) {
        // 扩张
        const xe = centroid.map(function (c, k) { return c + 2 * (c - sim[n][k]); });
        const fe = fun(xe); nfev++;
        if (fe < fr) { sim[n] = xe; fv[n] = fe; action = "expand"; }
        else { sim[n] = xr; fv[n] = fr; action = "reflect"; }
      } else if (fr < fv[n - 1]) {
        sim[n] = xr; fv[n] = fr;
      } else {
        // 收缩（外收缩/内收缩）
        let xc = centroid.map(function (c, k) {
          return c + 0.5 * ((fr < fv[n] ? c - sim[n][k] : sim[n][k] - c));
        });
        const fc = fun(xc); nfev++;
        if ((fr < fv[n] && fc < fr) || (fr >= fv[n] && fc < fv[n])) {
          sim[n] = xc; fv[n] = fc; action = "contract";
        } else {
          // 缩边
          for (let i = 1; i <= n; i++) {
            sim[i] = sim[0].map(function (c, k) { return c + 0.5 * (sim[i][k] - c); });
            fv[i] = fun(sim[i]); nfev++;
          }
          action = "shrink";
        }
      }
      iterations.push({ action: action, vertices: sim.map(v => v.slice()),
                        fvals: fv.slice(), nfev: nfev });
    }
    const order = fv.map(function (v, i) { return i; })
                    .sort(function (a, b) { return fv[a] - fv[b]; });
    return { x: sim[order[0]].slice(), f: fv[order[0]], iterations: iterations,
             nfev: nfev };
  }
  function maxAbsDiff(vs) {
    let mn = Infinity, mx = -Infinity;
    vs.forEach(function (v) { v.forEach(function (x) {
      if (x < mn) mn = x; if (x > mx) mx = x; }); });
    return mx - mn;
  }
  function maxAbsDiffArr(a) {
    let mn = Infinity, mx = -Infinity;
    a.forEach(function (x) { if (x < mn) mn = x; if (x > mx) mx = x; });
    return mx - mn;
  }

  /* ---------------- 完整流程（网格 → 局部细化） ---------------- */
  function fullSolve(ds, budget, Ns) {
    const M = constructMatrices(ds);
    const gs = gridSearch(M.R, M.C, M.D, budget, Ns);
    const x0 = [gs.best.u1, gs.best.u2];
    const nm = nelderMead(function (x) {
      return objective(M.R, M.C, M.D, budget, x[0], x[1]).f;
    }, x0);
    const oBest = objective(M.R, M.C, M.D, budget, nm.x[0], nm.x[1]);
    return { M: M, grid: gs, nm: nm, best: { u1: nm.x[0], u2: nm.x[1],
             f: oBest.f, acc: oBest.sim.accRate, cost: oBest.sim.costAvg,
             tau: oBest.tau, sim: oBest.sim } };
  }

  /* ---------------- 基线（单 API） ---------------- */
  function baselines(M) {
    return M.R[0].map(function (_, j) {
      let acc = 0, cost = 0;
      for (let i = 0; i < M.n; i++) { acc += M.R[i][j]; cost += M.C[i][j]; }
      return { api: j, acc: acc / M.n, cost: cost / M.n };
    });
  }

  return { mulberry32: mulberry32, makeGauss: makeGauss, clamp: clamp,
           APIS: APIS, generateDataset: generateDataset,
           constructMatrices: constructMatrices,
           quantileLinear: quantileLinear, tauFromQual: tauFromQual,
           simulate: simulate, objective: objective,
           gridSearch: gridSearch, nelderMead: nelderMead,
           fullSolve: fullSolve, baselines: baselines };
});
