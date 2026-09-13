# Q1：τ₃（末级门限）为什么不需要求解？

> **状态**：已解决（可继续追问）
> **日期**：2026-09-13
> **关联**：[总览笔记](../LLM_cascade_optimizer_notes.md) · 涉及文件 `src/FrugalGPT/optimizer.py`、`src/FrugalGPT/llmchain.py`、`src/FrugalGPT/llmcascade.py`、`src/FrugalGPT/scoring.py`

---

## 问题原文

> 实际上 τ₃ 应该不需要求解吧？因为如果到了第三层，不管结果如何应该都是采纳？代码中是如何设计的？

---

## 结论（TL;DR）

**理解完全正确。** 级联末级的语义是"兜底必答"——落到最后一级的查询必须被采纳，否则级联没有答案可返回。因此 **τ₃ ≡ 1 是常量，不是优化变量**。代码在三处体现这一设计：

1. **训练侧**：`thres[-1] = 1` 硬编码，搜索空间只有 m−1 = 2 维（网格 40² 而非 40³）；
2. **离线推理**（`predict_one`）：末级门限根本不被查询，循环跌落后无条件返回末级答案（结构性兜底）；
3. **在线推理**（`get_completion`）：末级仍过判断但条件恒真（`score > 1 − 1 = 0`），外加 `None` 兜底，双保险。

---

## 详细分析

### 为什么末级必须"必答"（可行性要求）

级联的输出语义是：逐级询问，谁先被采纳就返回谁的答案。若末级也可以拒绝，
则存在"所有级都拒绝"的查询，此时级联**没有答案可返回**，目标函数 E[r(a, f_Lz(q))] 中的 z 无定义——问题不可行。
所以任何可行解都要求末级对（几乎）所有查询接受。

### 数学上：τ_m 是一个"等价类"，1 是其规范代表

目标函数与成本约束只通过"每条查询在哪些级被接受"这一**离散划分**依赖 τ。
对末级而言，任何 `τ_m ≥ max_i d_m(q_i)` 都使末列接受谓词恒为真，产生的划分完全相同，
目标与约束的值也完全相同——即这些 τ_m 在解空间中是**同一个点**。
优化器无需搜索这个维度，取一个规范代表即可；取区间上界 1 最自然（d ∈ [0,1) 必然满足 d < 1）。

> 注意：规范代表的**数值**依赖坐标系——1 是代码失败空间（接受 ⟺ $d < \tau$）的答案；
> 若按论文的成功门限语义（接受 ⟺ $g \ge \tau$），同等等价类的规范代表是 **0**。
> 语义翻转详见 [Q5](Q5_tau_semantics.md)。

推论：**深度 m 的级联，自由门限只有 m−1 个**，内层 `brute` 的复杂度是 O(Ns^(m−1))。

### 兜底不是免费的

落到末级的查询要支付**全部 m 次调用**的费用（`C_mat` 的末列是累计成本）。
预算约束 `Σ e_full ∘ C_mat ≤ budget × n` 把这部分计入。因此压低 q₁、q₂（提高早停比例）的动机之一，
正是控制"全程走完"的贵查询占比——这是自由门限与固定末级之间的耦合方式。

---

## 代码证据

### 证据 1：训练侧硬编码 + 搜索空间少一维（`optimizer.py:104-118`）

```python
def quatile2thres_batch(qual):
    thres = numpy.zeros(L_mat.shape[1])
    thres[-1] = 1                      # ← 末级门限固定为 1，不在循环内
    mask_last = mask_full
    for i in range(0, len(thres)-1):   # ← 只解前 d-1 级
        thres[i], mask_last = quatile2thres(qual[i], i, mask_last)
    return thres
```

```python
qual_ranges = [(1e-5, 1-1e-5)] * (L_mat.shape[1]-1)   # optimizer.py:118，长度 d-1 = 2
```

`scipy.optimize.brute` 只在 (q₁, q₂) 的 40×40 网格上搜索；τ₃ 永远是 1。

### 证据 2：模拟器里末级"必然接受"（`optimizer.py:67`）

训练模拟 `f(delta)` 的接受谓词是 `e_full = d_mat < delta`，其中 `delta[-1] = 1`。
d = 1 − score ∈ [0,1)，故末列几乎全为 True；配合 first-accept-wins 循环（`optimizer.py:69-76`），
效果即"落到末级的查询一律采纳"。

### 证据 3：离线推理——结构性兜底，末级门限不被查询（`llmchain.py:180-194`）

```python
def predict_one(self, data1, cost1, dist_1):
    apis = []
    full_cost = 0
    for i in range(0, len(self.model_ids)-1):   # ← 只遍历前 m-1 级
        dist = dist_1[i]
        full_cost += cost1[i]
        apis.append(self.model_ids[i])
        if dist < self.thres[i]:
            return data1[i], full_cost, apis
    full_cost += cost1[-1]                       # ← 循环自然跌落
    apis.append(self.model_ids[-1])
    return data1[-1], full_cost, apis            # ← 无条件返回末级答案
```

注意 `self.thres[-1]`（=1）在这里**从未被读取**——末级接受是循环结构保证的，不依赖门限值。

### 证据 4：在线推理——恒真判断 + None 兜底（`llmcascade.py:127-141`）

```python
while(1):
    service_name, score_thres = LLMChain.nextAPIandScore()   # 末级返回 score_thres = 1
    if service_name == None:
        break                                                 # ← 安全网：列表耗尽
    res = MyLLMEngine.get_completion(...)
    ...
    score = self.MyScores[service_name].get_score(...)
    if score > 1 - score_thres:    # 末级: score > 1 - 1 = 0 → 恒真
        break
```

两条路径殊途同归：`predict_one` 不查末级门限；`get_completion` 查了但恒真，且即便不成立也会因列表耗尽而 break，返回已拿到的 `res`。

### 证据 5：score 的取值范围保证"恒真"成立（`scoring.py:214-216`）

```python
def get_score(self, text):
    prob = self.predict("", text)
    return prob[1]        # softmax 正类概率 ∈ (0,1)，实际不会精确为 0
```

### 证据 6：策略存储/加载的往返一致性（`llmchain.py:60-86`）

`savestrategy` 保存 `thres_list`（长度 m，末位为 1）与 `quantile`（长度 m−1）；
`loadstrategy` 原样读回 `self.thres` → `nextAPIandScore()` 在末级返回 score_thres=1，与证据 4 呼应。

---

## 论文证据

- **`main.tex:149`**（路由行为描述）：
  "It returns the generation if the score is higher than a threshold **τ_i**, and queries the next service otherwise."
  ——只描述了逐级判断，未说明末级失败时的行为。
- **`main.tex:156`**（约束公式）：
  `z = argmin_i g(q, f_Li(q)) ≥ τ_i`
  ——按字面理解，若所有级都不满足 g ≥ τᵢ，则 z 无定义，目标函数无值。实现以"末级必答"补全了这一缺口。
- **结论**：论文公式把 τ 写成长度 m 的向量，但**没有明说 τ_m 是常量**；"末级必答"是实现补充的结构性细节，
  也是对公式可行性的隐式修复。

---

## 公式与数学推理

记第 i 级的"距离"与接受谓词：

$$d_i(q) = 1 - g(q, f_{L_i}(q)) \in (0,1), \qquad A_i(q;\tau_i) = \mathbb{1}\{d_i(q) < \tau_i\}$$

级联返回位置 `z(q) = min{i : A_i(q) = 1}`，则

$$\text{acc}(\tau) = \frac{1}{n}\sum_q r\big(f_{L_{z(q)}}(q)\big), \qquad \text{cost}(\tau) = \frac{1}{n}\sum_q \sum_{i \le z(q)} c_i(q)$$

**阶梯不变性**：acc 与 cost 只通过划分 {A_i} 依赖 τ。若两组 τ′, τ″ 对所有查询产生相同划分，则目标值相同。

**末级分析**：
- 任意 τ_m ≥ max_q d_m(q) ⟹ A_m ≡ 1（全接受）⟹ z(q) 总有定义；
- 训练集上 max_q d_m(q) = 1 − min_q score_m(q) < 1（softmax 概率严格为正）；
- 故 τ_m = 1 与所有 τ_m ∈ [max_q d_m(q), +∞) 属同一等价类，取规范代表 **τ_m = 1**；
- 反方向：τ_m < 1 时，若存在查询使 d_m(q) ≥ τ_m，则 z 无定义 ⟹ **不可行**。
  所以"可行域 ⟹ τ_m ≥ max d_m"，不失一般性取 1。

**自由维度**：m 级级联的可搜索参数为 τ₁…τ_{m−1}，共 m−1 个；代码中对应 m−1 个 quantile 变量。

**死角（理论）**：d_m = 1 ⟺ score = 0.0（float32 softmax 下溢）时，1{1 < 1} = 0，
该查询在训练模拟中该行全 False——零成本、零正确，等于静默丢弃。实际概率 ≈ 0，不触发；
这也解释了为什么 `get_completion` 仍保留恒真判断和 None 兜底（防御性编程）。

---

## 边界情况与注意点

| 情形 | 行为 |
|---|---|
| 查询早停于第 1/2 级 | 只付对应前缀成本，末级费用不发生 |
| 查询走到末级 | 无条件采纳，付全部 3 次调用费用（`C_mat[:,-1]` 计入预算） |
| score 下溢为 0（理论） | 训练模拟中静默丢弃；线上因 None-break 仍返回末级答案 |
| `predict_one` 与 `get_completion` | 前者结构性兜底，后者恒真判断 + None 兜底，行为一致 |

---

## 总结

τ₃ 不需要求解，因为级联末级的语义是**兜底必答**：这是可行性要求（必须有答案返回），
而"接受一切"的所有门限值在目标函数上彼此等价，故取规范常量 τ_m ≡ 1。
代码在训练（硬编码 + 搜索空间降为 m−1 维）、离线推理（循环跌落，不查末级门限）、
在线推理（恒真判断 + None 兜底）三处一致地落实了这一语义；
兜底查询的全价成本则通过累计成本矩阵被预算约束如实计入。

---

## 可能的追问方向

- τ₁、τ₂ 的求解：为什么搜 quantile 而不直接搜 threshold（阶梯断点枚举）→ 建议下一问
- q₁ ≤ q₂ 单调约束（`optimizer.py:87`）为什么成立、违反时发生什么
- 罚函数 10000 对 Nelder-Mead 局部搜索的影响
- 存活集（mask_last）逐级收缩的精确语义
