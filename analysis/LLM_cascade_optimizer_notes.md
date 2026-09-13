# FrugalGPT LLM Cascade 优化问题求解方法笔记

> **论文**：Strategy 3: LLM cascade（`arXiv-2305.05176v1/main.tex`，§ How to Use LLMs Affordably and Accurately，公式在 152–159 行）
> **代码**：`src/FrugalGPT/{optimizer, llmchain, llmcascade, scoring}.py`
> **问答档案**：见 §8 索引（Q1–Q6）
> **交互式演示**：[`visualization/tau-solver/index.html`](../visualization/tau-solver/index.html)——200 条数据 × 固定 L，分步演示 τ 求解全过程（矩阵 → 模拟 → 阶梯 → quantile/alive → 网格 → Nelder-Mead → 结果；算法经 node 测试验证，说明见其 README）

---

## 0. 记号表（先读这张表，后文全部用它）

| 论文符号 | 本笔记记号 | 代码变量 | 含义 | 定义处 |
|---|---|---|---|---|
| $q,\ a$ | — | — | 一条查询与真实答案 | — |
| $r(a,\cdot)\in\{0,1\}$ | $R_{ij}$ | `L_mat[i,j]` | 第 $i$ 条查询用第 $j$ 级 API 的答案是否正确（exact match） | `optimizer.py:43` |
| $\tilde c_{L,0/1/2}$ | $C_{ij}$ | `C_mat[i,j]` | **累计**成本：调用前 $j$ 级 API 的总费用 | `optimizer.py:46` |
| $g(q,a)\in(0,1)$ | $D_{ij}=1-g$ | `d_mat[i,j]` | "距离" = 打分器认为该答案错误的概率 | `optimizer.py:9` |
| $\mathbf{L}=(L_1,L_2,L_3)$ | 同 | `selected_id` / `model_ids` | 选中的 API 及顺序（**决策变量，离散**） | `llmchain.py:122` |
| $\boldsymbol\tau=(\tau_1,\tau_2,\tau_3)$ | 同 | `thres` | 各级接受门限（**决策变量，连续**；$\tau_3\equiv 1$，见 Q1）。⚠️ 代码语义 = **失败概率上限**（越大越宽松），与论文（$g\ge\tau$，越大越严格）相反，见 [Q5](qa/Q5_tau_semantics.md) | — |
| — | $\mathbf{u}=(u_1,u_2)$ | `qual` / `quantile` | quantile 搜索坐标：第 $j$ 级的**升级比例**（quantile 水平；放行 $1-u_j$、存活 $u_j$） | `optimizer.py:118` |
| — | $A_{ij}(\boldsymbol\tau)$ | `e_full` | 接受指示矩阵（含"首站停"修正，见 §4.1） | `optimizer.py:67` |
| — | $\mathbf{1}\{\cdot\}$ | — | 指示函数：条件成立取 1、否则取 0 | — |
| $z$ | $z_i$ | （循环变量） | 第 $i$ 条查询的停机级 | — |

打分器 $g$：每个候选 API 训练一个 DistilBERT 二分类器，输入 "query + answer" 文本，输出正确类概率 `prob[1]`（`scoring.py:214`）。

**矩阵尺寸**：$R, C, D, A \in \mathbb{R}^{n\times m}$，$n$ = 训练查询数，$m$ = 级联深度（=3）。

---

## 1. 论文的优化问题与三个难点

$$
\max_{\mathbf{L},\ \boldsymbol\tau}\ \ \mathbb{E}\big[\,r\big(a,\, f_{L_z}(q)\big)\,\big] \tag{1}
$$

$$
\text{s.t.}\quad
\mathbb{E}\Big[\sum_{i=1}^{z} \tilde c_{L_i,2}\,\|f_{L_i}(q)\| + \tilde c_{L_i,1}\,\|q\| + \tilde c_{L_i,0}\Big] \le b,
\qquad
z = \min\{i:\ g(q, f_{L_i}(q)) \ge \tau_i\}
$$

难在三件事叠加：

1. **混合整数**：$\mathbf{L}$ 离散、$\boldsymbol\tau$ 连续，且通过停机位置 $z$ 互相耦合，无法套标准 MIP 求解器（详见 [Q3](qa/Q3_mip_solver.md)）；
2. **无闭式期望**：目标与约束是对未知查询分布的积分，只能用样本估计（详见 [Q2](qa/Q2_expectation_closed_form.md)）；
3. **阶梯函数**：固定 $\mathbf{L}$ 后，目标对 $\boldsymbol\tau$ 分段常数——每条查询的接受/拒绝只在 $\tau_j$ 跨过其观测距离 $D_{ij}$ 时翻转，没有梯度（推导与图解见 [Q4](qa/Q4_step_function.md)）。

---

## 2. 求解总体框架：谁被枚举、谁被连续优化

**先回答一个关键困惑：排列组合枚举只发生在 $\mathbf{L}$ 上，$\boldsymbol\tau$ 完全不是枚举排列组合。**

- **外层（离散）**：$\mathbf{L}$ 是 $K$ 个 API 取 $m$ 个的**有序排列**，$P(12,3) = 12\times 11\times 10 = 1320$ 条，逐条试（`llmchain.py:122` 的 `itertools.permutations`）。
- **内层（连续）**：固定 $\mathbf{L}$ 后解 $\boldsymbol\tau$。由于 $\tau_3 \equiv 1$（[Q1](qa/Q1_tau3_fallback.md)），自由变量只有 $(\tau_1, \tau_2)$ 两个数；代码改在 quantile 坐标 $\mathbf{u}\in(0,1)^2$ 上做 **40×40 粗网格 + Nelder-Mead 局部细化**（§4.2）。

```
外层（离散，全枚举 1320 条）            内层（连续，网格+局部搜索）
for 每条 L ∈ P(K,3):
    ├─ 预算剪枝：首级太贵 → 跳过
    ├─ 构造 R, C, D（一次查表，之后零 LLM 调用）
    └─ 解内层 ─────────────────────→   搜 u=(u1,u2) ∈ (0,1)²
                                        ├─ 40×40 网格，每点算两个 Σ（§4.1）
                                        └─ fmin（Nelder-Mead）细化
    记录 (acĉ, L, τ*(u*))
最后取 acĉ 最大者 → 存 strategy json
```

---

## 3. 预计算：期望 → 样本平均 → 三张表

训练前离线把每个候选 API 对 $n$ 条训练查询的回答跑一遍并缓存（sqlite completion cache），然后对每条候选链 $\mathbf{L}$ 构造三张表（`optimizer.construct_data`）：

$$
R_{ij} = r\big(a_i,\ f_{L_j}(q_i)\big),
\qquad
C_{ij} = \sum_{k=1}^{j} c_{L_k}(q_i),
\qquad
D_{ij} = 1 - g\big(q_i,\ f_{L_j}(q_i)\big)
$$

其中单次调用成本 $c_L(q) = \tilde c_{L,2}\|f_L(q)\| + \tilde c_{L,1}\|q\| + \tilde c_{L,0}$，与论文约束中的表达式逐项对应——$\tilde c_{L,0}$ 为每次调用的固定费，$\tilde c_{L,1}/\tilde c_{L,2}$ 为输入/输出的每 token 单价，$\|q\|/\|f_L(q)\|$ 为查询/回答的 token 数（实现见 `service/utils.py::compute_cost`）；$C$ 的前缀和由 `optimizer.py:46-47` 的累加实现。

样本平均代替期望（SAA，见 [Q2](qa/Q2_expectation_closed_form.md)）之后，问题 (1) 在训练集上变成**确定性**问题，其后整个优化过程零 LLM 调用、全是查表。

---

## 4. 内层：固定 $\mathbf{L}$，解 $\boldsymbol\tau$

### 4.1 给定 $\boldsymbol\tau$，目标值怎么算——$A$（即代码里的 `e_full`）与两个 Σ

**第一步：朴素接受指示矩阵**

$$\tilde A_{ij}(\boldsymbol\tau) = \mathbf{1}\{D_{ij} < \tau_j\}$$

> ⚠️ **语义提醒**：这里的 $\tau_j$ 是**失败概率** $D = 1-g$ 的上限（越大越宽松），与论文中作用在 $g$ 上的成功门限（$g \ge \tau$，越大越严格）方向相反，$\tau^{\text{code}} = 1 - \tau^{\text{paper}}$，详见 [Q5](qa/Q5_tau_semantics.md)。

**第二步："首站停"修正**——级联在第一个接受的级就停下，其后的级即使满足条件也不算数：

$$
z_i(\boldsymbol\tau) = \min\{j:\ \tilde A_{ij} = 1\},
\qquad
A_{ij}(\boldsymbol\tau) = \tilde A_{ij}\cdot \mathbf{1}\{j = z_i\}
$$

修正后**每行至多一个 1**（位于停机级 $z_i$）。$\tau_3\equiv 1$ 使 $D_{i3}<1$ 几乎必然成立（[Q1](qa/Q1_tau3_fallback.md)），所以每行**恰好一个 1**。

有了 $A$，目标与约束就是两个双求和（这就是代码里突然出现的 Σ）：

$$
\widehat{\mathrm{acc}}(\boldsymbol\tau) = \frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{m} A_{ij}\, R_{ij},
\qquad
\widehat{\mathrm{cost}}(\boldsymbol\tau) = \frac{1}{n}\sum_{i=1}^{n}\sum_{j=1}^{m} A_{ij}\, C_{ij}
\tag{2}
$$

**为什么这两个和是对的**：行内唯一的 1 在 $z_i$ 处，内层求和只剩 $R_{i,z_i}$——"这条查询最终返回的答案对不对"；$C_{i,z_i}$ 正是"到停机为止的累计花费"（$C$ 设计成前缀和就是为了这一步）。预算约束的样本版：

$$
\widehat{\mathrm{cost}} \le b
\qquad\Longleftrightarrow\qquad
\textstyle\sum_{i,j} A_{ij} C_{ij} \le b\cdot n
$$

对应代码（`optimizer.py:64-84`），符号一一对应：

```python
e_full = (d_mat < delta)            # ã_A：D_ij < τ_j
# …逐行只保留第一个 1…               # A：首站停修正（optimizer.py:69-76）
acc  = numpy.sum(e_full * L_mat)    # n·acĉ  = Σ A∘R
cost = numpy.sum(e_full * C_mat)    # n·cost̂ = Σ A∘C
if cost > budget * len(L_mat):      # cost̂ > b
    return 10000                    # 罚函数法处理约束
return -acc
```

一次求值 = 两个矩阵内积，$O(nm)$，微秒级。**注意：给定 $\boldsymbol\tau$ 求值是纯代数；"优化"是接下来决定怎么搜 $\boldsymbol\tau$。**

### 4.2 怎么搜 $\boldsymbol\tau$：quantile 坐标 + 粗网格 + 单纯形

目标对 $\boldsymbol\tau$ 是阶梯函数（难点 3），直接在 τ 空间布网格会大量浪费在"产生同一划分"的等价点上。代码**换坐标系**：不搜 τ 本身，搜"quantile 水平（升级比例）"。

**搜索变量**：$\mathbf{u} = (u_1, u_2) \in (0,1)^2$，$u_j$ = 第 $j$ 级的**升级比例**（到达者中继续存活的比例 = quantile 水平；放行比例为 $1-u_j$。⚠️ 勘误：早期版本误记为"放行比例"，页面可视化实测 $u_1{=}0.359$ → 停于第一级 128/200 $=64\% = 1-u_1$ 后修正，详见 [Q6](qa/Q6_quantile_grid_alive_simplex.md) §4.1）。门限由经验分布反解。存活集合递归定义：

$$
\mathrm{alive}_1 = \{1,\dots,n\},\qquad
\tau_j = \operatorname{Quantile}\big(\{D_{ij}: i \in \mathrm{alive}_j\},\ 1 - u_j\big),
\qquad
\mathrm{alive}_{j+1} = \{i \in \mathrm{alive}_j:\ D_{ij} \ge \tau_j\}
$$

四个细节：

- 分位数**只在上一级存活者上取**（`mask_last` 传播，`optimizer.py:95-112`）——与线上真正到达第 $j$ 级的分布一致；
- 代码强制单调 $u_1 \le u_2$（升级比例不降），违反返回罚值 10000（`optimizer.py:87`）——这是建模选择/正则化而非数学必然，见 [Q6](qa/Q6_quantile_grid_alive_simplex.md) §4.3；
- $\tau_3 \equiv 1$ 不参与搜索（[Q1](qa/Q1_tau3_fallback.md)）；
- $\operatorname{Quantile}(x, q)$：$x$ 从小到大排序后的 $q$ 分位数（numpy 线性插值实现，小例子见 [Q6](qa/Q6_quantile_grid_alive_simplex.md) §B2）。

**搜索算法**（`optimizer.py:124`）：

1. **粗网格**：$\mathbf{u} \in \{1/N_s, \dots, 1-1/N_s\}^2$，$N_s = 40$ → 1600 个格点，逐点执行 §4.1 的两个求和；
2. **局部细化**：`finish=fmin`——Nelder-Mead 单纯形法（无梯度），从网格最优点继续小步搜索；
3. **输出**：最优 $\mathbf{u}^*$，经 `quatile2thres_batch` 反解出 $\boldsymbol\tau^*$，连同 $\widehat{\mathrm{acc}}$ 返回。

**为什么 quantile 坐标是好主意**（这就是论文 "interpolating the objective within a few samples" 的实质）：

1. **单元格对齐**：阶梯函数有意义的门限只有 $n$ 个观测距离值；经验分位数把 u 轴均匀切成 ~n 段、每段恰对应一个单元格——u 网格的每个格点必命中一个**此前未评估的**单元格，零浪费（τ 空间均匀网格会按数据密度重复命中同一单元格、又跳过窄单元格，见 [Q6](qa/Q6_quantile_grid_alive_simplex.md)）；
2. **搜索域规整**：单位盒 $(0,1)^2$，方便等距布网格；
3. **语义清晰**：每个格点有业务含义（"放行 60%、升级 40%"），便于调网格密度。

> 精确化（[Q6](qa/Q6_quantile_grid_alive_simplex.md)）：quantile 网格并非枚举全部 n 个断点，而是在 ~n 个单元格上按质量均匀采样其中 Ns=40 个；40 个格点两两相距 ≥ 25 段，必落互异单元格。

### 4.3 复杂度

每条链：$40^2 + O(10^2)$ 次求值 × 每次 $O(nm)$ 查表 → 毫秒级；1320 条链合计秒~分钟级。真正昂贵的 $K\cdot n$ 次 LLM 调用只发生一次并落缓存。

---

## 5. 外层：枚举 $\mathbf{L}$ 并剪枝

`LLMChain.train`（`llmchain.py:107`）：

1. **枚举**：`itertools.permutations(service_ids, 3)`，只搜**恰好** 3 级的有序排列——论文自述理由："cascade length of 3 … simplifies the optimization space and already demonstrates good results"（`main.tex:179`）；
2. **预算剪枝**：首级平均成本 $\frac1n\sum_i C_{i1} > b$ 的链直接跳过，连内层都不启动（`optimizer.py:60-62`，返回 −999）；
3. **选择**：所有链的结果取 $\widehat{\mathrm{acc}}$ 最大者（`llmchain.py:131-136`），把 `(model_ids, thres, quantile)` 存入 strategy json。

**关于论文的 "answer disagreement" 剪枝**：开源版没有字面实现；其思想被 $D$ 的定义吸收——若链中后级答案几乎总与前级一致，该链与更短的链行为等价但更贵，必在 argmax 中落败。发布规模下（1320 条）全枚举可承受，分歧剪枝是候选池更大时的加速手段。

---

## 6. 数值案例：用 §0 记号完整走一遍

设定：$m=3$ 个 API（A/B/C），$n=5$ 条训练查询，预算 $b=0.01$。单次成本 $c_A=\$0.001$，$c_B=\$0.005$，$c_C=\$0.03$；$D_{ij}=1-g_{ij}$。

| $i$ | $R_{i1}$ | $g_{i1}$ | $R_{i2}$ | $g_{i2}$ | $R_{i3}$ | $g_{i3}$ | $D_{i1}$ | $D_{i2}$ | $D_{i3}$ |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 0.9 | 1 | 0.8 | 1 | 0.95 | 0.1 | 0.2 | 0.05 |
| 2 | 0 | 0.3 | 0 | 0.4 | 1 | 0.90 | 0.7 | 0.6 | 0.10 |
| 3 | 1 | 0.8 | 1 | 0.7 | 1 | 0.90 | 0.2 | 0.3 | 0.10 |
| 4 | 0 | 0.2 | 1 | 0.6 | 1 | 0.85 | 0.8 | 0.4 | 0.15 |
| 5 | 0 | 0.7 | 0 | 0.5 | 0 | 0.30 | 0.3 | 0.5 | 0.70 |

注意 $i=5$：A 答错但 $g=0.7$——打分器不完美，其代价由 $R$ **经验地**计入目标，而非假设打分器全知。

**第一步（剪枝）**：链 $(C,A,B)$ 首级平均成本 \$0.03 > \$0.01 → 跳过（`optimizer.py:60`）。

**第二步（取一个格点 $(u_1,u_2)=(0.4,\,0.7)$）**：

- $\tau_1 = \operatorname{Quantile}(D_{\cdot 1},\, 0.6) \approx 0.46$ → 第 1 级接受 $D_{i1}<0.46$ 的 $\{1,3,5\}$，$\mathrm{alive}_2 = \{2,4\}$；
- $\tau_2 = \operatorname{Quantile}\big(D_{\cdot 2}\big|_{\mathrm{alive}_2},\, 0.3\big) \approx 0.46$ → 行 4 在第 2 级接受；
- 行 2 落到末级，$\tau_3 = 1$ 兜底必接受。

于是接受矩阵与两个和：

$$
A = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \\ 0 & 1 & 0 \\ 1 & 0 & 0 \end{pmatrix},
\qquad
\widehat{\mathrm{acc}} = \tfrac{1}{5}\sum A\circ R = \tfrac{1+0+1+1+0}{5} = 0.8,
\qquad
\widehat{\mathrm{cost}} = \tfrac{1}{5}\sum A\circ C = \tfrac{0.001+0.036+0.001+0.006+0.001}{5} = 0.009 \le 0.01\ \checkmark
$$

（$A\circ R$ 表示逐元素乘，即 `numpy.multiply(e_full, L_mat)`。）

**第三步（搜索其余 1599 个格点 + Nelder-Mead 细化）**：方向感——$u_1$ 调小 → 更多查询落到贵的 C → 可能触发 $\widehat{\mathrm{cost}}>b$ 返 10000；$u_1$ 调大 → A 多接错题 → acc 下降。取预算内最优者。

**对比基线**：单用 A → acc 0.6；单用 C → acc 0.8 但成本 \$0.03/查询。级联用 **30% 的成本达到同样的 0.8**。论文的 Tradeoff 曲线就是把 $b$ 从小到大扫一遍、每个 $b$ 解一次本问题的结果。

---

## 7. 论文 ↔ 代码的三处措辞差异

1. **"answer disagreement" 剪枝**：论文有此描述，开源代码无显式实现（思想吸收进 $D$，见 §5）；
2. **$\tau_3$**：论文公式把 $\boldsymbol\tau$ 写成一般向量，实现固定 $\tau_3\equiv 1$ 为结构性兜底（[Q1](qa/Q1_tau3_fallback.md)）；
3. **$\tau$ 的语义翻转**：论文的 $\tau$ 作用在成功概率 $g$ 上（$g\ge\tau$ 接受，越大越严），代码的 $\tau$ 作用在失败概率 $D=1-g$ 上（$D<\tau$ 接受，越大越松），$\tau^{\text{code}} = 1-\tau^{\text{paper}}$（[Q5](qa/Q5_tau_semantics.md)）。

推理侧一致性：`llmcascade.get_completion` 的接受条件 `score > 1 − score_thres`（`llmcascade.py:139`）即 $1-g < \tau_j$，与训练模拟 $A_{ij} = \mathbf{1}\{D_{ij}<\tau_j\}$ 完全一致。

---

## 8. 问答索引（一问一档，档案位于 `analysis/qa/`）

| # | 问题 | 档案 | 状态 |
|---|---|---|---|
| Q1 | τ₃（末级门限）为什么不需要求解？ | [Q1_tau3_fallback.md](qa/Q1_tau3_fallback.md) | 已解决 |
| Q2 | "无闭式期望"是什么意思？是期望就没有解析形式吗？ | [Q2_expectation_closed_form.md](qa/Q2_expectation_closed_form.md) | 已解决 |
| Q3 | 什么是标准 MIP 求解器？要求什么条件、适合什么问题？ | [Q3_mip_solver.md](qa/Q3_mip_solver.md) | 已解决 |
| Q4 | 目标对 τ 的阶梯函数性质：推导、图解、"没有梯度"的后果 | [Q4_step_function.md](qa/Q4_step_function.md)（含[插图](qa/figures/Q4_step_function.png)） | 已解决 |
| Q5 | 论文 τ 是成功概率门限、代码 τ 是失败概率门限（语义翻转） | [Q5_tau_semantics.md](qa/Q5_tau_semantics.md) | 已解决 |
| Q6 | quantile 网格的真正作用、alive 集合、粗细网格分工、单纯形（含基础补丁） | [Q6_quantile_grid_alive_simplex.md](qa/Q6_quantile_grid_alive_simplex.md) | 已解决 |

> **记录规则**：每个新问题新建 `analysis/qa/Q<n>_<slug>.md`，模板固定为：
> 问题原文 / 结论 TL;DR / 详细分析 / 代码证据（文件:行号）/ 论文证据（main.tex 行号 + 原文）/
> 公式与数学推理 / 边界情况与注意点 / 总结 / 可能的追问方向。
> 问题解决或内容更新后，在上表登记并更新状态。

---

## 9. 待讨论清单

- [ ] 打分器 $g$ 的训练细节：输入格式、`score_test_size=0.55` 的用途、为何每个 API 一个打分器
- [ ] `Quantile` 反解的 numpy 线性插值细节；$\mathrm{alive}$ 集合（`mask_last`）传递的精确语义
- [ ] 罚函数 10000 的后果：对 Nelder-Mead 的误导；预算约束是"训练集上恰好满足"还是泛化保证
- [ ] `scipy.optimize.brute` + `finish=fmin` 的交互：网格点数、终止条件、Nelder-Mead 机制
- [ ] `for ell in range(L_max, L_max+1)` 只搜恰好 3 级——2 级链理论上会被 3 级链覆盖吗
- [ ] 阶梯断点枚举的严格论证（§4.2 第 1 点的形式化）
- [x] $u_1 \le u_2$ 单调约束：已实验——约束出自原作者代码（`optimizer.py:86-88`），与实现的逐级条件语义错配（更像"累计采纳/全样本分位"参数化的遗留物）；演示数据上禁掉了更优漏斗（训练集 +1.5~7.5pt、30 次留出划分测试集 +0.4~2.3pt 且 n=80 时正则化辩护仍不成立），**结论：不应加**，详见 [Q6](qa/Q6_quantile_grid_alive_simplex.md) §4.3；真实数据集建议重复 A/B 后再定
- [ ] 训练/测试两层划分：`test_size=0.01` 与 `score_test_size=0.55` 各防什么（SAA 过拟合，见 Q2 追问方向）
- [ ] 若要实现 disagreement 剪枝：判据与插入位置
- [ ] 若硬把内层写成 big-M MILP：完整模型与弱松弛示例（见 Q3 §5 与追问方向）
