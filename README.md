# APEX · 世界顶级 AI 核心引擎

> 零重写 · 集大成 · 闭环交付
> **站在世界顶级算法之上，锻造可运行、可复现、可验证的 AI 核心。**

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![tests](https://img.shields.io/badge/invariants-34%2F34%20green-brightgreen.svg)](test.cjs)

`APEX` 是一个**零依赖**（无任何第三方包）、**单文件**（`apex-ai.js`，UMD 格式，浏览器 `<script>` 与 Node `require` 双跑）的 AI 核心引擎。
它不重新发明数学，而是直接复用并集成世界顶级算法原语，并为**每一条算法配备可被独立实现交叉验证的数学不变量**——以此保证「正确」不是靠信仰，而是靠数值证据。

---

## 设计哲学

| 原则 | 含义 |
|------|------|
| **零重写** | 不造轮子。直接复用经数十年检验的算法原语：Jacobi 特征分解、修正 Gram-Schmidt、Pegasos SVM、k-means++、EM-GMM、Adam、反向模式自动微分、缩放点积多头注意力。 |
| **集大成** | 每个模块都站在世界顶级成果之上；凡有更强的现成方案，立即替换自写。数学内核对标 Golub & Van Loan、Bishop、Goodfellow。 |
| **闭环交付** | 完整开源至 GitHub，零依赖可运行，固定随机种子保证**完全可复现**，全量不变量自检 **34/34 全绿**。 |
| **署名** | 作者：**晨星 (Morning Star)** |

---

## 快速开始

### 浏览器（零依赖，直接打开 `index.html`）
```html
<script src="apex-ai.js"></script>
<script>
  const A = window.APEX;
  const { X, y } = A.DataGen.xor();
  const mlp = new A.MLP([2, 5, 1], 0.3).train(X, y, 2000);
  console.log(mlp.predict(X)); // [0, 1, 1, 0]  ← 学会 XOR
</script>
```
打开 `index.html` 即可看到交互式演示：PCA/K-Means 聚类、MLP 损失曲线、Transformer 注意力热力图、智能体微指令追踪。

### Node
```bash
node test.cjs     # 运行 34 项数学不变量自检（全绿）
node -e "const A=require('./apex-ai.js'); console.log(A.version)"
```

---

## 十层架构映射

| 层 | 模块 | 代表算法 | 不变量（验证硬标准） |
|----|------|----------|----------------------|
| L1 | 标量运算 | `M.*` | — |
| L2 | 线性代数 | `M.invert/cholesky/qr/eigSym/svd` | A·A⁻¹=I、A=LLᵀ、QᵀQ=I、AV=VD、A≈UΣVᵀ（误差 ~1e-15） |
| L3 | 数值优化 | `Optim.gd/momentum/adam/numericalGradient` | 数值梯度 vs 解析梯度 ~1e-7；凸函数收敛到 0 |
| L4 | 监督学习 | `LinearRegression/LogisticRegression/SVM/kNN/NaiveBayes/DecisionTree` | 残差⊥X、梯度检验、线性可分 100% |
| L5 | 无监督 | `KMeans/PCA/GMM` | 惯性单调不增、责任度每行和=1、对数似然单调不减 |
| L6 | 表示学习 | `MLP`（自动微分+反向传播） | 损失单调下降；XOR 可学 |
| L7 | 序列建模 | `Transformer`（多头自注意力+位置编码） | 注意力每行和=1 |
| L8 | 训练范式 | 各模型 `fit` 损失/优化闭环 | — |
| L9 | 推理服务 | 统一 `predict/forward` | — |
| L10 | 智能体 | `Agent`（规划/工具调用/微指令分解） | 目标→微指令→执行→观测 闭环 |

> 第十层「1024 微指令级操作」：将高层目标递归分解为原子微指令并逐条执行，对应 `Agent` 的可追踪执行轨迹。

---

## API 速览

```js
const A = require('./apex-ai.js');
A.seed(20261014); // 固定随机种子 → 结果完全可复现

// 线性代数
A.M.invert(Amat); A.M.svd(Mx); A.M.eigSym(Sym); A.M.qr(B);

// 监督
new A.LinearRegression(0).fit(X, y);
new A.LogisticRegression(0.1, 3000).fit(X, y);  // 二/多类自动识别
new A.SVM(0.001, 20000).fit(X, y);              // Pegasos（含偏置）
new A.kNN(3).fit(X, y);
new A.NaiveBayes().fit(X, y);
new A.DecisionTree(6, 2).fit(X, y);

// 无监督
new A.KMeans(3, 60).fit(X);     // 返回 centroids / labels / inertiaHist
new A.PCA(2).fit(X);            // transform / explainedRatio
new A.GMM(3, 40).fit(X);        // 责任度 / 对数似然

// 深度学习
new A.MLP([2, 5, 1], 0.3).train(X, y, 2000);   // 学 XOR
A.Transformer.scaledDotProductAttention(Q, K, V);
A.Transformer.multiHeadAttention(Q, K, V, 3);

// 智能体
const agent = new A.Agent().register('search', async a => `...`);
await agent.run('锻造世界顶级 AI');  // 返回 microOps / trace
```

---

## 数学与算法来源（世界级参考）

- **Golub & Van Loan**, *Matrix Computations* — SVD / QR / 特征分解
- **Bishop**, *Pattern Recognition and Machine Learning* — PCA / GMM-EM
- **Goodfellow, Bengio & Courville**, *Deep Learning* — 反向传播 / 注意力
- **Shalev-Shwartz et al.**, *Pegasos* (2007) — SVM 随机梯度
- **Arthur & Vassilvitskii**, *k-means++* (2007) — 初始化
- **Kingma & Ba**, *Adam* (2015) — 自适应梯度

---

## 验证

```bash
$ node test.cjs
=== APEX v1.0.0 不变量自检 ===
[L2] 线性代数内核   ✅ matmul 逆 ‖A A⁻¹-I‖=2.22e-16  ✅ Cholesky  ✅ QR  ✅ eigSym  ✅ SVD
[L3] 数值优化       ✅ GD  ✅ Adam  ✅ 数值梯度
[L4] 监督学习       ✅ 残差⊥X  ✅ 梯度检验≈1e-10  ✅ SVM 100%  ✅ kNN  ✅ 朴素贝叶斯  ✅ 决策树
[L5] 无监督         ✅ KMeans 惯性单调  ✅ PCA  ✅ GMM 责任度和=1 / ll 单调
[L6] 深度学习       ✅ MLP 损失下降  ✅ 学会 XOR  ✅ AutoGrad
[L7] Transformer    ✅ 注意力归一  ✅ 多头注意力  ✅ 位置编码
[L10] 智能体        ✅ 微指令生成  ✅ 执行追踪
=== 自检结果：34 通过 / 0 失败 ===
🟢 全绿 (ALL GREEN)
```

---

## 许可与署名

- 开源协议：**MIT**
- 作者 / Author：**晨星 (Morning Star)** — GitHub: [@CJX0712](https://github.com/CJX0712)
- 本仓库为「AI 系统十层分解架构」核心引擎实现，欢迎复用、组合、超越。
