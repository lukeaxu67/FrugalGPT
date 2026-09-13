/* core.js 的自动化验证（node tests/core.test.js） */
const F = require("../core.js");

let pass = 0, fail = 0;
function check(name, cond, detail) {
  if (cond) { pass++; console.log("  PASS  " + name); }
  else { fail++; console.log("  FAIL  " + name + (detail ? "  -> " + detail : "")); }
}

console.log("== 1. 数据生成 ==" );
const ds = F.generateDataset(200, 20260913);
check("n=200", ds.rows.length === 200);
const meanAcc = [0, 1, 2].map(j => ds.rows.reduce((s, r) => s + r.per[j].correct, 0) / 200);
const meanCost = [0, 1, 2].map(j => ds.rows.reduce((s, r) => s + r.per[j].cost, 0) / 200);
console.log("  单API正确率:", meanAcc.map(x => x.toFixed(3)).join(" / "),
            " 单次成本:", meanCost.map(x => x.toFixed(4)).join(" / "));
check("正确率单调 A<B<C", meanAcc[0] < meanAcc[1] && meanAcc[1] < meanAcc[2],
      meanAcc.join(","));
check("成本单调 A<B<C", meanCost[0] < meanCost[1] && meanCost[1] < meanCost[2]);
const scoreRange = ds.rows.every(r => r.per.every(x => x.score > 0 && x.score < 1));
check("score ∈ (0,1)", scoreRange);
// 打分器不完美：正确但低分 / 错误但高分 的比例
let mismatch = 0, total = 0;
ds.rows.forEach(r => r.per.forEach(x => {
  total++;
  if ((x.correct === 1 && x.score < 0.5) || (x.correct === 0 && x.score >= 0.5)) mismatch++;
}));
console.log("  打分器错位比例:", (100 * mismatch / total).toFixed(1) + "%");
check("打分器不完美（错位 5%~40%）", mismatch / total > 0.05 && mismatch / total < 0.40);

console.log("== 2. 矩阵与不变量 ==" );
const M = F.constructMatrices(ds);
check("C 为前缀和", M.C.every((c, i) =>
  Math.abs(c[2] - (c[1] + ds.rows[i].per[2].cost)) < 1e-12 &&
  Math.abs(c[1] - (c[0] + ds.rows[i].per[1].cost)) < 1e-12));
check("D = 1 - score", M.D.every((d, i) =>
  [0,1,2].every(j => Math.abs(d[j] - (1 - ds.rows[i].per[j].score)) < 1e-12)));

// τ→0⁺：全部落到末级 C
const s0 = F.simulate(M.R, M.C, M.D, [1e-9, 1e-9, 1], 1e9);
check("τ≈0 → acc = C单独正确率", Math.abs(s0.accRate - meanAcc[2]) < 1e-12,
      s0.accRate + " vs " + meanAcc[2]);
check("τ≈0 → cost = 累计到C的平均", Math.abs(s0.costAvg - meanCost[0] - meanCost[1] - meanCost[2]) < 1e-12);
// τ=1：全部在 A 首级停
const s1 = F.simulate(M.R, M.C, M.D, [1, 1, 1], 1e9);
check("τ=1 → acc = A单独正确率", Math.abs(s1.accRate - meanAcc[0]) < 1e-12);
check("τ=1 → cost = A单独成本", Math.abs(s1.costAvg - meanCost[0]) < 1e-12);

// first-accept 独立复核（从后往前的独立实现）
function firstAcceptIndependent(D, tau, i) {
  for (let j = 0; j < 3; j++) if (D[i][j] < tau[j]) return j;
  return -1;
}
const tauRnd = [0.42, 0.61, 1];
const sR = F.simulate(M.R, M.C, M.D, tauRnd, 1e9);
let zOK = true;
for (let i = 0; i < M.n; i++) {
  if (sR.z[i] !== firstAcceptIndependent(M.D, tauRnd, i)) { zOK = false; break; }
}
check("首站停语义（独立复核 200 行）", zOK);
let accOK = true;
for (let i = 0; i < M.n; i++) {
  const zi = sR.z[i];
  if (sR.A[i].reduce((a, b) => a + b, 0) !== (zi >= 0 ? 1 : 0)) accOK = false;
}
check("接受矩阵每行 ≤1 个 1", accOK);

console.log("== 3. 分位数与 quantile/alive ==" );
const qv = F.quantileLinear([0.1, 0.2, 0.3, 0.7, 0.8], 0.6);
check("numpy linear 分位数对拍 (0.6 → 0.46)", Math.abs(qv - 0.46) < 1e-12, qv);
const tq = F.tauFromQual(0.4, 0.7, M.D);
check("tau3 ≡ 1", tq.tau[2] === 1);
check("alive 单调收缩 |alive1|>=|alive2|",
  tq.alive[0].filter(Boolean).length >= tq.alive[1].filter(Boolean).length,
  `|a1|=${tq.alive[0].filter(Boolean).length} |a2|=${tq.alive[1].filter(Boolean).length}`);
check("alive2 = D[:,0] >= tau1 的行",
  tq.alive[1].every((m, i) => m === (M.D[i][0] >= tq.tau[0])));
// u clamp：u 越界不崩溃（optimizer.py:98 的 clamp）
const tClamp = F.tauFromQual(1.4, 2.0, M.D);
check("u 越界被 clamp（不抛异常）", isFinite(tClamp.tau[0]) && isFinite(tClamp.tau[1]));

console.log("== 4. 单调约束罚 ==" );
const oMono = F.objective(M.R, M.C, M.D, 0.006, 0.7, 0.4);
check("u2<u1 → 罚 10000", oMono.f === 10000 && oMono.penalty === "mono");

console.log("== 5. Nelder-Mead 在光滑函数上收敛 ==" );
const nmT = F.nelderMead(x => (x[0]-0.3)*(x[0]-0.3) + (x[1]-0.7)*(x[1]-0.7), [0, 0]);
check("二次函数收敛到 (0.3, 0.7)",
      Math.abs(nmT.x[0]-0.3) < 1e-3 && Math.abs(nmT.x[1]-0.7) < 1e-3,
      nmT.x.join(","));
check("迭代日志非空且含动作标记", nmT.iterations.length > 3 &&
      nmT.iterations.some(it => it.action === "reflect"));

console.log("== 6. 不同预算下的完整求解（选演示预算） ==" );
[0.004, 0.005, 0.006, 0.008, 0.010].forEach(function (b) {
  const sol = F.fullSolve(ds, b, 40);
  const gb = sol.grid.best;
  const a2 = sol.best.sim ? sol.best.sim.A : null;
  const alive2 = sol.best.tau ? F.tauFromQual(sol.best.u1, sol.best.u2, M.D).alive[1].filter(Boolean).length : 0;
  console.log(
    `b=${b.toFixed(3)}  网格最优 acc=${(gb ? gb.acc : -1).toFixed(3)}` +
    `  NM后 u*=(${sol.best.u1.toFixed(3)}, ${sol.best.u2.toFixed(3)})` +
    `  acc=${sol.best.acc.toFixed(3)}  cost=${sol.best.cost.toFixed(4)}` +
    `  tau=(${sol.best.tau[0].toFixed(3)}, ${sol.best.tau[1].toFixed(3)})` +
    `  |alive2|=${alive2}/200  C单独acc=${meanAcc[2].toFixed(3)}  C成本=${(meanCost[0]+meanCost[1]+meanCost[2]).toFixed(3)}` +
    `  NM优于网格: ${sol.best.f <= gb.f}`);
  check(`b=${b}: NM 不劣于网格最优`, sol.best.f <= gb.f + 1e-12);
});

console.log(`\n结果: ${pass} 通过, ${fail} 失败`);
process.exit(fail ? 1 : 0);
