# -*- coding: utf-8 -*-
"""Q4 插图：目标/约束对 tau_1 的阶梯函数性质（单门限切片 = 笔记 §6 数值案例）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np

D  = np.array([0.1, 0.7, 0.2, 0.8, 0.3])   # D_i1（q1..q5）
R1 = np.array([1, 0, 1, 0, 0])             # A 是否答对
R3 = np.array([1, 1, 1, 1, 0])             # 兜底 C 是否答对
CA, CC = 0.001, 0.036
BUDGET = 0.02
n = len(D)
QLABELS = {0.1: "q1", 0.7: "q2", 0.2: "q3", 0.8: "q4", 0.3: "q5"}

def acc_cost(tau):
    m = D < tau
    return (R1[m].sum() + R3[~m].sum()) / n, (m.sum() * CA + (~m).sum() * CC) / n

bps = np.unique(D)
edges = np.concatenate([[0.0], bps, [1.02]])
y_acc, y_cost = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    a, c = acc_cost((lo + hi) / 2)
    y_acc.append(a); y_cost.append(c)
y_acc, y_cost = np.array(y_acc), np.array(y_cost)

SURFACE, GRID = "#fcfcfb", "#e1e0d9"
INK, INK2, MUT = "#0b0b0b", "#52514e", "#898781"
AXIS, BLUE, ORG = "#c3c2b7", "#2a78d6", "#eb6834"
RED, WASH_B = "#d03b3b", "#cde2fb"

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUT, "ytick.color": MUT,
})

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 6.9), dpi=200, sharex=True)
fig.subplots_adjust(left=0.085, right=0.975, top=0.90, bottom=0.09, hspace=0.42)

for ax in (ax1, ax2):
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for bp in bps:
        ax.axvline(bp, color=MUT, linewidth=0.8, linestyle=(0, (3, 3)), alpha=0.75)
    ax.set_xlim(0, 1.02)

# 查询标签：面板 1 放在图内底部，面板 2 放在轴上方
for bp in bps:
    ax1.text(bp, 0.04, QLABELS[bp], ha="center", va="bottom", fontsize=8.5, color=MUT)
trans2 = blended_transform_factory(ax2.transData, ax2.transAxes)
for bp in bps:
    ax2.text(bp, 1.03, QLABELS[bp], transform=trans2, ha="center", va="bottom",
             fontsize=8.5, color=MUT)

# ---- 面板 1：准确率 ------------------------------------------------
ax1.axvspan(0.3, 0.7, color=WASH_B, alpha=0.55, lw=0)
ax1.step(edges, np.append(y_acc, y_acc[-1]), where="post", color=BLUE,
         linewidth=2.2, solid_joinstyle="miter")
ax1.set_ylim(0, 1.0)
ax1.set_ylabel(r"$\widehat{\mathrm{acc}}(\tau_1)$", fontsize=11)
ax1.set_title("目标：只在断点处跳变，其余区间是平台（梯度 = 0）",
              fontsize=11.5, color=INK, loc="left", pad=26)
ax1.tick_params(labelbottom=False)

ax1.annotate("平台区 (0.3, 0.7]：任何 $\\tau_1$ 取值\n→ 同一接受集合 → 同一目标值",
             xy=(0.5, 0.8), xytext=(0.52, 0.92), fontsize=9, color=INK2,
             arrowprops=dict(arrowstyle="-", color=MUT, lw=0.8))
ax1.annotate("", xy=(0.62, 0.60), xytext=(0.38, 0.60),
             arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
ax1.text(0.50, 0.535, "$\\tau_1$ 在平台内任意移动，目标纹丝不动",
         ha="center", fontsize=9, color=INK2)
ax1.annotate("断点 = 每条查询的观测距离 $D_{i1}$：\n$\\tau_1$ 跨过它，该查询的接受状态才翻转",
             xy=(0.7, 0.72), xytext=(0.52, 0.22), fontsize=9, color=INK2, ha="left",
             arrowprops=dict(arrowstyle="->", color=MUT, lw=0.9,
                             connectionstyle="arc3,rad=0.28"))

# ---- 面板 2：成本 --------------------------------------------------
ax2.axvspan(0, 0.3, color=RED, alpha=0.055, lw=0)
ax2.step(edges, np.append(y_cost, y_cost[-1]), where="post", color=ORG,
         linewidth=2.2, solid_joinstyle="miter")
ax2.axhline(BUDGET, color=RED, linewidth=1.2, linestyle=(0, (5, 3)))
ax2.text(0.995, BUDGET + 0.0013, "预算 b = 0.02", ha="right", fontsize=9, color=RED)
ax2.set_ylim(0, 0.042)
ax2.set_ylabel(r"$\widehat{\mathrm{cost}}(\tau_1)$", fontsize=11)
ax2.set_xlabel(r"第一级门限  $\tau_1$（失败概率上限，越大越宽松）", fontsize=10.5)
ax2.set_title("约束：同为阶梯，且随 $\\tau_1$ 单调下降（早停越多，落到末级的贵查询越少）",
              fontsize=11.5, color=INK, loc="left", pad=18)
ax2.text(0.15, 0.0305, "不可行区\n$\\widehat{\\mathrm{cost}} > b$",
         ha="center", fontsize=9, color=RED)
ax2.annotate("可行且最优的平台：$\\tau_1$ 在 (0.3, 0.7] 内任取\n（经验分位数给出规范代表）",
             xy=(0.5, 0.015), xytext=(0.36, 0.0052), fontsize=9, color=INK2,
             arrowprops=dict(arrowstyle="->", color=MUT, lw=0.9))

out = r"D:\github\FrugalGPT\analysis\qa\figures\Q4_step_function.png"
fig.savefig(out, dpi=200)
print("saved:", out)
