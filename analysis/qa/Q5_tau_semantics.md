# Q5：论文的 τ 是成功概率门限，代码的 τ 是失败概率门限——语义翻转

> **状态**：已解决（可继续追问）
> **日期**：2026-09-13
> **关联**：[总览笔记](../LLM_cascade_optimizer_notes.md) §0/§4 · [Q1](Q1_tau3_fallback.md)（末级常量 1 的另一证据） · [Q4](Q4_step_function.md)（边界不等号）

---

## 问题原文

> 论文里阈值 τ 是成功的概率，而代码实现里是把 τ 作为失败的概率。因为论文里面定义 g ≥ τ 就可以接受，而实现中是 D_ij（失败的概率）小于 τ 就是可接受。我的理解是不是正确的？

---

## 结论（TL;DR）

**理解完全正确。**

- **论文**：接受条件 $g(q, f_{L_i}(q)) \ge \tau_i$，$g$ 是答对（成功）概率 → $\tau^{\text{paper}}$ 越大越**严格**；
- **代码**：接受条件 $D_{ij} < \tau_j$，$D = 1 - g$ 是失败概率 → $\tau^{\text{code}}$ 越大越**宽松**；
- 两者是同一优化问题的仿射重参数化：$\tau^{\text{code}} = 1 - \tau^{\text{paper}}$，可行集双射、目标不变；
- 代码里硬编码的末级常量 `thres[-1] = 1`（而非论文语义下的 0）正是失败概率空间的指纹，反向印证了这一观察。

---

## 详细分析

### 1. 论文语义：τ 是成功概率的门限

- **`main.tex:149`**："it uses the scoring function to generate a score $g(q, f_{L_i}(q))$. **It returns the generation if the score is higher than a threshold $\tau_i$**, and queries the next service otherwise."
- **`main.tex:156`**（约束式）：$z = \arg\min_i\ g(q, f_{L_i}(q)) \ge \tau_i$。

$g \in [0,1]$ 是打分器输出的"答案正确概率"→ 门限越大，越难被接受，越多查询升级到下一级。**τ ↑ = 严格 ↑。**

### 2. 代码语义：τ 是失败概率的上限

三处证据构成完整证据链（训练定义 → 训练使用 → 推理使用）：

**证据 1：定义处把分数翻转为失败概率**（`optimizer.py:9`）：

```python
def compute_dist(answers, scores, query=None):
    dist = 1 - scores[-1]        # D = 1 − g（代码称 distance）
```

**证据 2：训练模拟接受条件为 D < τ**（`optimizer.py:67`）：

```python
e_full = (d_mat < delta)         # 接受 ⟺ D < τ ⟺ 1−g < τ ⟺ g > 1−τ
```

**证据 3：推理路径换算回成功空间比较，与训练严格一致**（`llmcascade.py:139`）：

```python
if score > 1 - score_thres:      # g > 1−τ_code —— 冒烟枪：存储的阈值在失败空间
```

变量名 `score_thres`（`nextAPIandScore` 返回）有误导性——它实际存储的是**失败概率的上限**，不是成功分数的门限。若只读 `llmcascade.py` 而不读 `optimizer.py`，极易把它当成论文语义的 τ。

### 3. 数学关系：仿射重参数化

$$
\tau^{\text{code}}_j = 1 - \tau^{\text{paper}}_j,
\qquad
\underbrace{g \ge \tau^{\text{paper}}_j}_{\text{论文，接受}} \iff \underbrace{1 - g \le \tau^{\text{code}}_j}_{\text{代码，接受}}
$$

- 映射是双射：一个坐标系里的每个可行解在另一坐标系恰有一个像，目标函数、约束、最优解集合全部一一对应；
- **方向相反**是唯一需要小心的地方：论文里调大 τ = 提高标准；代码里调大 τ = 放宽标准（更多查询早停、成本下降、风险上升）；
- 边界差异：论文 $\ge$（含等号），代码严格 `<`（即 $g > 1-\tau$）。在连续分数上相差测度零的点集，不影响任何结论（与 [Q4](Q4_step_function.md) 边界情况同款）。

### 4. 交叉验证：Q1 的末级常量在两个坐标系中取值相反

"末级必答"常量的取值是语义翻转的独立证据：

| 坐标系 | "永远接受"的门限取值 | 代码实际取值 |
|---|---|---|
| 论文（成功门限） | $\tau_m^{\text{paper}} = 0$（$g \ge 0$ 恒真） | — |
| 代码（失败门限） | $\tau_m^{\text{code}} = 1$（$D < 1$ 几乎恒真） | `thres[-1] = 1`（`optimizer.py:106`） |

若作者的存储值与论文同语义，兜底常量应是 0 而非 1——这个 1 就是失败概率空间的指纹。

### 5. 代码为什么选择失败/distance 语义

1. **几何/分歧直觉**：distance 框架下接受条件是"距离低于阈值"（距离小 = 有信心 = 停），呼应论文 "answer disagreement" 的表述（早期级联设想中 distance 是 LLM 答案间分歧，本实现以 $1-g$ 代替）；
2. **quantile 参数化方向一致**：$\tau_j = \operatorname{Quantile}(D_{\cdot j},\ 1-u_j)$，$u_j$（升级比例）↑ → 分位水平 $1-u_j$ ↓ → $\tau_j$ ↓ → 放行 $1-u_j$ 减少、更**严格**，单调方向与"升级比例"的语义一致，搜索时不易搞反。

### 6. 阅读建议

对照论文公式与代码常数时，凡遇 $\tau$ 先问一句：**"这是 $g$ 的门限，还是 $1-g$ 的门限？"** 逐常数直译会得出矛盾（最典型：论文语义下末级应是 0，代码却是 1）。

---

## 公式与数学推理

设 $\phi: \boldsymbol\tau^{\text{paper}} \mapsto \boldsymbol\tau^{\text{code}} = \mathbf{1} - \boldsymbol\tau^{\text{paper}}$（末级除外，$\tau_m$ 两边均为常量）。对每条查询 $i$ 与级 $j$：

$$
\mathbf{1}\{g_{ij} \ge \tau^{\text{paper}}_j\} = \mathbf{1}\{1 - g_{ij} \le 1 - \tau^{\text{paper}}_j\} = \mathbf{1}\{D_{ij} \le \tau^{\text{code}}_j\}
$$

故接受指示矩阵 $A(\boldsymbol\tau^{\text{paper}}) = A'(\phi(\boldsymbol\tau^{\text{paper}}))$（仅在边界 $\le$/$<$ 上差测度零点集），于是

$$
\widehat{\mathrm{acc}}(\boldsymbol\tau^{\text{paper}}) = \widehat{\mathrm{acc}}'(\phi(\boldsymbol\tau^{\text{paper}})),
\qquad
\widehat{\mathrm{cost}}(\boldsymbol\tau^{\text{paper}}) = \widehat{\mathrm{cost}}'(\phi(\boldsymbol\tau^{\text{paper}}))
$$

两目标等值、两约束等价，$\phi$ 是优化问题之间的同构。阶梯结构（[Q4](Q4_step_function.md)）在 $\phi$ 下保持——断点从 $\{D_{ij}\}$ 映到 $\{1 - D_{ij}\} = \{g_{ij}\}$。

---

## 边界情况与注意点

- `score_noise_injection`（`llmcascade.py:136`）在 score 上加噪声，翻转发生在 `1 - score_thres` 之后，两坐标系下行为一致；
- 策略 json 中同时存了 `thres_list`（失败空间）与 `quantile`（各级**升级比例**，放行 $1-u_j$），加载后 `nextAPIandScore` 返回失败空间的 τ——再读 `get_completion` 时务必记得 `1 -` 这一步；
- 早期版本/其他分支若改用成功空间存储，推理比较须同步改为 `score > score_thres`——两处必须成对修改，这是本仓库隐含的耦合约定。

---

## 总结

论文的 $\tau$ 作用在成功概率 $g$ 上（$g \ge \tau$ 接受，越大越严），代码的 $\tau$ 作用在失败概率 $D = 1-g$ 上（$D < \tau$ 接受，越大越松），二者经 $\tau^{\text{code}} = 1 - \tau^{\text{paper}}$ 仿射互转、问题同构。代码的 `d_mat < delta`、`score > 1 - score_thres` 与末级常量 `1`（而非 0）四处证据互相咬合，确认存储/搜索全程在失败空间进行，仅在推理比较的瞬间换算回成功空间。

---

## 可能的追问方向

- 若把整个优化器改写到成功空间（$g$ 上搜 quantile），哪些公式变号、哪些不变
- `score_thres` 命名重构建议与影响面（json 兼容性）
- 早期 FrugalGPT 设想中"答案间分歧距离"与 $1-g$ 打分距离的异同（论文 disagreement 剪枝的语言源头）
