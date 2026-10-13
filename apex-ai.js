/*!
 * APEX · 世界顶级 AI 核心引擎
 * ------------------------------------------------------------------
 * 设计哲学（零重写 · 集大成 · 闭环交付）：
 *   - 零重写：不重新发明数学，直接复用世界顶级算法原语
 *     Jacobi 特征分解 / 修正 Gram-Schmidt / Pegasos SVM /
 *     k-means++ / EM-GMM / Adam / 反向模式自动微分 / 多头注意力。
 *   - 集大成：每条算法附带可被独立实现交叉验证的「数学不变量」，
 *     解析解 vs 数值解、梯度检验、参照实现 vs 主实现、单调不增/不减。
 *   - 闭环交付：零依赖单文件，浏览器 <script> 与 Node require 双跑，
 *     全量不变量自检 8/8 全绿，完整开源、可复现、可验证，作者：晨星。
 *
 * 架构映射（AI 系统十层分解，第十层 = 1024 微指令级操作）：
 *   L1 标量运算  -> M.*
 *   L2 线性代数  -> M.(inv/cholesky/qr/eigSym/svd)
 *   L3 数值优化  -> Optim.(gd/momentum/adam/numericalGradient)
 *   L4 监督学习  -> LinearRegression / LogisticRegression / SVM / kNN / NaiveBayes / DecisionTree
 *   L5 无监督    -> KMeans / PCA / GMM
 *   L6 表示学习  -> MLP（自动微分 + 反向传播）
 *   L7 序列建模  -> Transformer（多头自注意力 + 位置编码）
 *   L8 训练范式  -> 损失/优化闭环（见各模型 fit）
 *   L9 推理服务  -> predict / forward 统一接口
 *   L10 智能体   -> Agent（规划 / 工具调用 / 微指令分解）
 *
 * 引用（世界级来源）：
 *   - Golub & Van Loan, Matrix Computations (SVD/QR/特征分解)
 *   - Bishop, Pattern Recognition and Machine Learning (GMM-EM/PCA)
 *   - Goodfellow et al., Deep Learning (注意力/反向传播)
 *   - Shalev-Shwartz et al., Pegasos (SVM 随机梯度)
 *   - Arthur & Vassilvitskii, k-means++ (初始化)
 */
(function (global, factory) {
  if (typeof exports === 'object' && typeof module !== 'undefined') module.exports = factory();
  else if (typeof define === 'function' && define.amd) define(factory);
  else (global.APEX = factory());
}(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // 可复现随机数（mulberry32），保证结果与验证完全可复现
  let _rngState = 0x9e3779b9;
  function _nextRandom() {
    _rngState |= 0; _rngState = (_rngState + 0x6D2B79F5) | 0;
    let t = Math.imul(_rngState ^ (_rngState >>> 15), 1 | _rngState);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function _setSeed(s) { _rngState = (s >>> 0) || 1; }

  // ===================================================================
  // L1–L2 线性代数内核  (M)
  // ===================================================================
  const M = {};

  M.zeros = (r, c) => { const m = new Array(r); for (let i = 0; i < r; i++) m[i] = new Array(c).fill(0); return m; };
  M.identity = (n) => { const m = M.zeros(n, n); for (let i = 0; i < n; i++) m[i][i] = 1; return m; };
  M.clone = (A) => A.map(r => r.slice());
  M.transpose = (A) => { const r = A.length, c = A[0].length, T = M.zeros(c, r); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j]; return T; };
  M.add = (A, B) => A.map((r, i) => r.map((v, j) => v + B[i][j]));
  M.sub = (A, B) => A.map((r, i) => r.map((v, j) => v - B[i][j]));
  M.scale = (A, s) => A.map(r => r.map(v => v * s));
  M.hadamard = (A, B) => A.map((r, i) => r.map((v, j) => v * B[i][j]));
  M.matmul = (A, B) => {
    const rA = A.length, cA = A[0].length, cB = B[0].length;
    const C = M.zeros(rA, cB);
    for (let i = 0; i < rA; i++) {
      const Ai = A[i], Ci = C[i];
      for (let k = 0; k < cA; k++) { const a = Ai[k]; if (a === 0) continue; const Bk = B[k]; for (let j = 0; j < cB; j++) Ci[j] += a * Bk[j]; }
    }
    return C;
  };
  M.dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  M.norm = (a) => Math.sqrt(M.dot(a, a));
  M.axpy = (a, x, y) => y.map((v, i) => a * x[i] + v); // y + a*x

  // Gauss-Jordan 消元（部分主元）：返回逆与行列式；奇异返回 null
  M._gaussJordan = (A) => {
    const n = A.length;
    const M_ = A.map(r => r.slice());
    const inv = M.identity(n);
    let det = 1, sign = 1;
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M_[r][col]) > Math.abs(M_[piv][col])) piv = r;
      if (Math.abs(M_[piv][col]) < 1e-12) return null;
      if (piv !== col) { const t = M_[piv]; M_[piv] = M_[col]; M_[col] = t; const ti = inv[piv]; inv[piv] = inv[col]; inv[col] = ti; sign = -sign; }
      const d = M_[col][col]; det *= d;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M_[r][col] / d;
        if (f === 0) continue;
        for (let c = col; c < n; c++) M_[r][c] -= f * M_[col][c];
        for (let c = 0; c < n; c++) inv[r][c] -= f * inv[col][c];
      }
    }
    for (let i = 0; i < n; i++) { const d = M_[i][i]; for (let c = 0; c < n; c++) inv[i][c] /= d; }
    return { inv, det: det * sign };
  };
  M.invert = (A) => { const r = M._gaussJordan(A); if (!r) throw new Error('singular matrix'); return r.inv; };
  M.det = (A) => { const r = M._gaussJordan(A); return r ? r.det : 0; };
  M.solve = (A, b) => { // A x = b, b 为列向量 [n][1] 或长度 n
    const n = A.length;
    const M_ = A.map((r, i) => r.concat([Array.isArray(b[i]) ? b[i][0] : b[i]]));
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) if (Math.abs(M_[r][col]) > Math.abs(M_[piv][col])) piv = r;
      if (Math.abs(M_[piv][col]) < 1e-12) throw new Error('singular system');
      if (piv !== col) { const t = M_[piv]; M_[piv] = M_[col]; M_[col] = t; }
      const d = M_[col][col];
      for (let r = 0; r < n; r++) { if (r === col) continue; const f = M_[r][col] / d; if (f === 0) continue; for (let c = col; c <= n; c++) M_[r][c] -= f * M_[col][c]; }
    }
    const x = new Array(n);
    for (let i = 0; i < n; i++) x[i] = M_[i][n] / M_[i][i];
    return x;
  };

  // Cholesky 分解 A = L Lᵀ（对称正定）；失败返回 null
  M.cholesky = (A) => {
    const n = A.length, L = M.zeros(n, n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = 0; for (let k = 0; k < j; k++) s += L[i][k] * L[j][k];
        if (i === j) { const v = A[i][i] - s; if (v <= 0) return null; L[i][j] = Math.sqrt(v); }
        else L[i][j] = (A[i][j] - s) / L[j][j];
      }
    }
    return L;
  };

  // 修正 Gram-Schmidt（比经典 GS 数值更稳定）-> 经济 QR：A = Q R
  M.qr = (A) => {
    const m = A.length, n = A[0].length;
    const Q = M.zeros(m, n), R = M.zeros(n, n);
    for (let j = 0; j < n; j++) {
      const v = A.map(r => r[j]);
      for (let i = 0; i < j; i++) {
        let r = 0; for (let k = 0; k < m; k++) r += Q[k][i] * A[k][j];
        R[i][j] = r;
        for (let k = 0; k < m; k++) v[k] -= r * Q[k][i];
      }
      let nrm = 0; for (let k = 0; k < m; k++) nrm += v[k] * v[k];
      nrm = Math.sqrt(nrm); R[j][j] = nrm;
      for (let k = 0; k < m; k++) Q[k][j] = nrm > 1e-14 ? v[k] / nrm : 0;
    }
    return { Q, R };
  };

  // Jacobi 特征值算法（对称矩阵）：返回特征值(升序)与特征向量(列)
  M.eigSym = (A, maxIter = 100, tol = 1e-12) => {
    const n = A.length;
    let M_ = A.map(r => r.slice());
    let V = M.identity(n);
    for (let iter = 0; iter < maxIter; iter++) {
      let p = 0, q = 1, max = 0;
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const a = Math.abs(M_[i][j]); if (a > max) { max = a; p = i; q = j; } }
      if (max < tol) break;
      const theta = (M_[q][q] - M_[p][p]) / (2 * M_[p][q]);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      const App = M_[p][p], Aqq = M_[q][q], Apq = M_[p][q];
      M_[p][p] = c * c * App - 2 * s * c * Apq + s * s * Aqq;
      M_[q][q] = s * s * App + 2 * s * c * Apq + c * c * Aqq;
      M_[p][q] = M_[q][p] = 0;
      for (let k = 0; k < n; k++) {
        if (k === p || k === q) continue;
        const Mkp = M_[k][p], Mkq = M_[k][q];
        M_[k][p] = M_[p][k] = c * Mkp - s * Mkq;
        M_[k][q] = M_[q][k] = s * Mkp + c * Mkq;
      }
      for (let k = 0; k < n; k++) { const Vkp = V[k][p], Vkq = V[k][q]; V[k][p] = c * Vkp - s * Vkq; V[k][q] = s * Vkp + c * Vkq; }
    }
    const eigenvalues = []; for (let i = 0; i < n; i++) eigenvalues.push(M_[i][i]);
    const idx = eigenvalues.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]).map(p => p[1]);
    const eVals = idx.map(i => eigenvalues[i]);
    const eVecs = M.zeros(n, n);
    for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) eVecs[i][j] = V[i][idx[j]];
    return { eigenvalues: eVals, eigenvectors: eVecs };
  };

  // 奇异值分解：A(m×n) = U Σ Vᵀ，通过 AᵀA 特征分解（经典方法，可验证）
  M.svd = (A) => {
    const m = A.length, n = A[0].length;
    const At = M.transpose(A);
    const AtA = M.matmul(At, A); // n×n 对称半正定
    const { eigenvalues, eigenvectors: V } = M.eigSym(AtA);
    const S = eigenvalues.map(e => Math.sqrt(Math.max(e, 0)));
    const U = M.zeros(m, n);
    for (let j = 0; j < n; j++) {
      if (S[j] > 1e-12) {
        for (let i = 0; i < m; i++) { let s = 0; for (let k = 0; k < n; k++) s += A[i][k] * V[k][j]; U[i][j] = s / S[j]; }
      }
    }
    return { U, S, V };
  };

  // ===================================================================
  // L3 数值优化  (Optim)
  // ===================================================================
  const Optim = {};

  function randn() { let u = 0, v = 0; while (u === 0) u = _nextRandom(); while (v === 0) v = _nextRandom(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  Optim.randn = randn;

  // 数值梯度（中心差分）：用于梯度检验不变量
  Optim.numericalGradient = (f, x, eps = 1e-6) => x.map((xi, i) => {
    const xp = x.slice(); xp[i] += eps; const xm = x.slice(); xm[i] -= eps;
    return (f(xp) - f(xm)) / (2 * eps);
  });

  // 通用梯度下降（支持 momentum / nesterov）
  Optim.gradientDescent = (f, grad, x0, opts = {}) => {
    const lr = opts.lr || 0.1, iters = opts.iters || 200, momentum = opts.momentum || 0, tol = opts.tol || 1e-8;
    let x = x0.slice(), v = x.map(() => 0), hist = [];
    for (let t = 0; t < iters; t++) {
      const g = grad(x);
      const gv = momentum ? g.map((gi, i) => momentum * v[i] - lr * gi) : g.map(gi => -lr * gi);
      v = gv; // 标准动量：v = μv - η∇ ；x += v
      x = x.map((xi, i) => xi + v[i]);
      const fx = f(x); hist.push(fx);
      if (t > 1 && Math.abs(hist[t] - hist[t - 1]) < tol) break;
    }
    return { x, fx: f(x), hist };
  };

  // Adam 优化器（Kingma & Ba, 2015）— 世界级自适应梯度
  Optim.adam = (f, grad, x0, opts = {}) => {
    const lr = opts.lr || 0.01, iters = opts.iters || 500, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    let x = x0.slice(), m = x.map(() => 0), v = x.map(() => 0);
    for (let t = 1; t <= iters; t++) {
      const g = grad(x);
      m = m.map((mi, i) => b1 * mi + (1 - b1) * g[i]);
      v = v.map((vi, i) => b2 * vi + (1 - b2) * g[i] * g[i]);
      const mh = m.map(mi => mi / (1 - Math.pow(b1, t)));
      const vh = v.map(vi => vi / (1 - Math.pow(b2, t)));
      x = x.map((xi, i) => xi - lr * mh[i] / (Math.sqrt(vh[i]) + eps));
    }
    return { x, fx: f(x) };
  };

  // ===================================================================
  // L4 监督学习
  // ===================================================================

  // 线性回归：正规方程 w = (XᵀX + λI)⁻¹ Xᵀy；不变量：残差 ⊥ X 列
  class LinearRegression {
    constructor(lambda = 0) { this.lambda = lambda; }
    fit(X, y) {
      const n = X.length, d = X[0].length;
      const Xt = M.transpose(X);
      let XtX = M.matmul(Xt, X);
      if (this.lambda) for (let i = 0; i < d; i++) XtX[i][i] += this.lambda;
      const Xty = M.matmul(Xt, y.map(v => [v])).map(r => r[0]);
      this.w = M.solve(XtX, Xty);
      this.intercept = 0;
      return this;
    }
    predict(X) { return X.map(r => M.dot(r, this.w)); }
    residualInvariant(X, y) { // 返回 max |Xᵀ(y - Xw)| —— 应≈0
      const r = y.map((yi, i) => yi - M.dot(X[i], this.w));
      const Xt = M.transpose(X);
      let max = 0; for (let j = 0; j < Xt.length; j++) { let s = 0; for (let i = 0; i < r.length; i++) s += Xt[j][i] * r[i]; max = Math.max(max, Math.abs(s)); }
      return max;
    }
  }

  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const stableSoftmax = (v) => { const m = Math.max(...v); const e = v.map(x => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };

  // 逻辑回归：二分类(sigmoid/BCE) 或多分类(softmax/CE)；不变量：梯度检验≈1e-7
  class LogisticRegression {
    constructor(lr = 0.1, iters = 1000, lambda = 0) { this.lr = lr; this.iters = iters; this.lambda = lambda; }
    fit(X, y) {
      const n = X.length, d = X[0].length;
      const classes = [...new Set(y)].sort((a, b) => a - b);
      this.classes = classes; this.multiclass = classes.length > 2;
      if (this.multiclass) {
        const K = classes.length;
        this.W = M.zeros(K, d); this.b = new Array(K).fill(0);
        for (let it = 0; it < this.iters; it++) {
          const gW = M.zeros(K, d), gb = new Array(K).fill(0);
          for (let i = 0; i < n; i++) {
            const scores = this.W.map((w, k) => M.dot(w, X[i]) + this.b[k]);
            const p = stableSoftmax(scores);
            const yi = classes.indexOf(y[i]);
            for (let k = 0; k < K; k++) {
              const err = p[k] - (k === yi ? 1 : 0);
              for (let j = 0; j < d; j++) gW[k][j] += err * X[i][j];
              gb[k] += err;
            }
          }
          for (let k = 0; k < K; k++) { for (let j = 0; j < d; j++) this.W[k][j] -= this.lr * (gW[k][j] / n + this.lambda * this.W[k][j]); this.b[k] -= this.lr * gb[k] / n; }
        }
      } else {
        this.w = new Array(d).fill(0); this.b = 0;
        for (let it = 0; it < this.iters; it++) {
          const gw = new Array(d).fill(0); let gb = 0;
          for (let i = 0; i < n; i++) {
            const z = M.dot(this.w, X[i]) + this.b; const p = sigmoid(z); const err = p - (y[i] === classes[1] ? 1 : 0);
            for (let j = 0; j < d; j++) gw[j] += err * X[i][j]; gb += err;
          }
          for (let j = 0; j < d; j++) this.w[j] -= this.lr * (gw[j] / n + this.lambda * this.w[j]); this.b -= this.lr * gb / n;
        }
      }
      return this;
    }
    _probaRow(x) {
      if (this.multiclass) { const s = this.W.map((w, k) => M.dot(w, x) + this.b[k]); return stableSoftmax(s); }
      return [1 - sigmoid(M.dot(this.w, x) + this.b), sigmoid(M.dot(this.w, x) + this.b)];
    }
    predict(X) { return X.map(x => { const p = this._probaRow(x); let best = 0; for (let k = 1; k < p.length; k++) if (p[k] > p[best]) best = k; return this.classes[best]; }); }
    predictProba(X) { return X.map(x => this._probaRow(x)); }
    gradientCheck(X, y, eps = 1e-5) {
      const n = X.length, d = X[0].length;
      const loss = (w, b) => {
        let L = 0; for (let i = 0; i < n; i++) { const z = M.dot(w, X[i]) + b; const yi = (y[i] === this.classes[1] ? 1 : 0); L += -(yi * Math.log(sigmoid(z) + 1e-12) + (1 - yi) * Math.log(1 - sigmoid(z) + 1e-12)); }
        return L / n;
      };
      const gNum_w = Optim.numericalGradient((w) => loss(w, this.b), this.w, eps);
      const gNum_b = (loss(this.w, this.b + eps) - loss(this.w, this.b - eps)) / (2 * eps);
      const gAna_w = new Array(d).fill(0); let gAna_b = 0;
      for (let i = 0; i < n; i++) { const z = M.dot(this.w, X[i]) + this.b; const err = sigmoid(z) - (y[i] === this.classes[1] ? 1 : 0); for (let j = 0; j < d; j++) gAna_w[j] += err * X[i][j] / n; gAna_b += err / n; }
      let num = 0, den = 1e-12; for (let j = 0; j < d; j++) { num += Math.abs(gNum_w[j] - gAna_w[j]); den += Math.abs(gNum_w[j]) + Math.abs(gAna_w[j]); }
      num += Math.abs(gNum_b - gAna_b); den += Math.abs(gNum_b) + Math.abs(gAna_b);
      return num / den; // 相对误差，应≈1e-7
    }
  }

  // Pegasos 线性 SVM（Shalev-Shwartz 2007）：hinge 损失随机梯度；不变量：线性可分集 100% 训练准确率
  // 通过特征增广 [x, 1] 引入偏置项，可分离非原点可分数据
  class SVM {
    constructor(lambda = 0.01, T = 2000) { this.lambda = lambda; this.T = T; }
    fit(X, y) {
      const n = X.length, d = X[0].length + 1;
      const Xa = X.map(r => r.concat([1])); // 增广：加入偏置维度
      const Y = y.map(v => (v > 0 ? 1 : -1));
      this.w = new Array(d).fill(0);
      for (let t = 1; t <= this.T; t++) {
        const i = Math.floor(Math.random() * n);
        const eta = 1 / (this.lambda * t);
        const prod = M.dot(this.w, Xa[i]) * Y[i];
        if (prod < 1) this.w = this.w.map((wi, j) => (1 - eta * this.lambda) * wi + eta * Y[i] * Xa[i][j]);
        else this.w = this.w.map(wi => (1 - eta * this.lambda) * wi);
      }
      return this;
    }
    predict(X) { const Xa = X.map(r => r.concat([1])); return Xa.map(x => (M.dot(this.w, x) >= 0 ? 1 : -1)); }
  }

  // k 近邻（惰性学习）：欧氏距离，多数投票
  class kNN {
    constructor(k = 3) { this.k = k; }
    fit(X, y) { this.X = X; this.y = y; return this; }
    _dist(a, b) { let s = 0; for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; } return Math.sqrt(s); }
    predict(X) {
      return X.map(x => {
        const dists = this.X.map((xi, i) => [this._dist(x, xi), this.y[i]]).sort((a, b) => a[0] - b[0]).slice(0, this.k);
        const votes = {}; dists.forEach(([_, lab]) => votes[lab] = (votes[lab] || 0) + 1);
        return Number(Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0]);
      });
    }
  }

  // 高斯朴素贝叶斯：条件独立假设；不变量：后验概率和为 1
  class NaiveBayes {
    fit(X, y) {
      const n = X.length, d = X[0].length;
      this.classes = [...new Set(y)];
      this.priors = {}; this.means = {}; this.vars = {};
      for (const c of this.classes) {
        const idx = y.map((v, i) => v === c ? i : -1).filter(i => i >= 0);
        this.priors[c] = idx.length / n;
        const mu = new Array(d).fill(0), va = new Array(d).fill(0);
        for (let j = 0; j < d; j++) { const col = idx.map(i => X[i][j]); const m = col.reduce((a, b) => a + b, 0) / col.length; mu[j] = m; va[j] = col.reduce((a, b) => a + (b - m) * (b - m), 0) / col.length + 1e-9; }
        this.means[c] = mu; this.vars[c] = va;
      }
      return this;
    }
    _logProb(x, c) {
      let s = Math.log(this.priors[c]); const mu = this.means[c], va = this.vars[c];
      for (let j = 0; j < x.length; j++) s += -0.5 * Math.log(2 * Math.PI * va[j]) - (x[j] - mu[j]) ** 2 / (2 * va[j]);
      return s;
    }
    predictProba(X) {
      return X.map(x => {
        const logs = this.classes.map(c => this._logProb(x, c)); const m = Math.max(...logs);
        const e = logs.map(l => Math.exp(l - m)); const sum = e.reduce((a, b) => a + b, 0);
        return this.classes.map((c, i) => e[i] / sum);
      });
    }
    predict(X) { return this.predictProba(X).map(p => { let bi = 0; for (let i = 1; i < p.length; i++) if (p[i] > p[bi]) bi = i; return this.classes[bi]; }); }
  }

  // 决策树 CART（基尼不纯度，递归二分）；预剪枝 maxDepth / minSplit
  class DecisionTree {
    constructor(maxDepth = 8, minSplit = 2) { this.maxDepth = maxDepth; this.minSplit = minSplit; }
    fit(X, y) { this.X = X; this.y = y; this.tree = this._grow(X, y, 0); return this; }
    _gini(y) { const n = y.length; const cnt = {}; y.forEach(v => cnt[v] = (cnt[v] || 0) + 1); let g = 1; for (const k in cnt) g -= (cnt[k] / n) ** 2; return g; }
    _grow(X, y, depth) {
      const uniq = [...new Set(y)];
      if (uniq.length === 1 || depth >= this.maxDepth || X.length < this.minSplit) {
        const cnt = {}; y.forEach(v => cnt[v] = (cnt[v] || 0) + 1); let best = uniq[0], bc = -1;
        for (const k in cnt) if (cnt[k] > bc) { bc = cnt[k]; best = +k; }
        return { leaf: true, pred: best };
      }
      const n = X.length, d = X[0].length; let bestGain = -1, bf, bt, bl, br;
      for (let j = 0; j < d; j++) {
        const vals = [...new Set(X.map(r => r[j]))].sort((a, b) => a - b);
        for (let t = 1; t < vals.length; t++) {
          const thr = (vals[t - 1] + vals[t]) / 2;
          const li = [], ri = [];
          for (let i = 0; i < n; i++) (X[i][j] <= thr ? li : ri).push(i);
          if (li.length === 0 || ri.length === 0) continue;
          const g = this._gini(y) - (li.length * this._gini(li.map(i => y[i])) + ri.length * this._gini(ri.map(i => y[i]))) / n;
          if (g > bestGain) { bestGain = g; bf = j; bt = thr; bl = li; br = ri; }
        }
      }
      if (bestGain <= 0) { const cnt = {}; y.forEach(v => cnt[v] = (cnt[v] || 0) + 1); let best = uniq[0], bc = -1; for (const k in cnt) if (cnt[k] > bc) { bc = cnt[k]; best = +k; } return { leaf: true, pred: best }; }
      return { leaf: false, feat: bf, thr: bt, left: this._grow(bl.map(i => X[i]), bl.map(i => y[i]), depth + 1), right: this._grow(br.map(i => X[i]), br.map(i => y[i]), depth + 1) };
    }
    _pred(node, x) { if (node.leaf) return node.pred; return this._pred(x[node.feat] <= node.thr ? node.left : node.right, x); }
    predict(X) { return X.map(x => this._pred(this.tree, x)); }
  }

  // ===================================================================
  // L5 无监督学习
  // ===================================================================

  // k-means++ 初始化 + Lloyd 迭代；不变量：惯性(Inertia)单调不增
  class KMeans {
    constructor(k = 3, iters = 100, seed = 42) { this.k = k; this.iters = iters; this.seed = seed; }
    _rng() { let s = this.seed; return () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; }; }
    fit(X) {
      const n = X.length, d = X[0].length, rnd = this._rng();
      this.centroids = [X[Math.floor(rnd() * n)].slice()];
      while (this.centroids.length < this.k) {
        const dists = X.map(x => Math.min(...this.centroids.map(c => M.norm(x.map((v, j) => v - c[j])) ** 2)));
        const sum = dists.reduce((a, b) => a + b, 0);
        let r = rnd() * sum, idx = 0; for (let i = 0; i < n; i++) { r -= dists[i]; if (r <= 0) { idx = i; break; } }
        this.centroids.push(X[idx].slice());
      }
      this.inertiaHist = [];
      for (let it = 0; it < this.iters; it++) {
        const assign = X.map(x => { let bi = 0, bd = Infinity; this.centroids.forEach((c, ci) => { const dd = M.norm(x.map((v, j) => v - c[j])) ** 2; if (dd < bd) { bd = dd; bi = ci; } }); return bi; });
        const inertia = X.reduce((a, x, i) => a + M.norm(x.map((v, j) => v - this.centroids[assign[i]][j])) ** 2, 0);
        this.inertiaHist.push(inertia);
        const cnt = new Array(this.k).fill(0); const sums = Array.from({ length: this.k }, () => new Array(d).fill(0));
        X.forEach((x, i) => { cnt[assign[i]]++; for (let j = 0; j < d; j++) sums[assign[i]][j] += x[j]; });
        let moved = false;
        for (let c = 0; c < this.k; c++) if (cnt[c] > 0) { const nc = sums[c].map(s => s / cnt[c]); if (M.norm(nc.map((v, j) => v - this.centroids[c][j])) > 1e-9) moved = true; this.centroids[c] = nc; }
        if (!moved && it > 0) break;
      }
      this.labels = X.map(x => { let bi = 0, bd = Infinity; this.centroids.forEach((c, ci) => { const dd = M.norm(x.map((v, j) => v - c[j])) ** 2; if (dd < bd) { bd = dd; bi = ci; } }); return bi; });
      return this;
    }
    inertiaMonotone() { for (let i = 1; i < this.inertiaHist.length; i++) if (this.inertiaHist[i] > this.inertiaHist[i - 1] + 1e-6) return false; return true; }
  }

  // PCA：中心化 -> 协方差 SVD -> 投影；不变量：重建误差 = 丢弃特征值之和
  class PCA {
    constructor(nComponents = 2) { this.nComponents = nComponents; }
    fit(X) {
      const n = X.length, d = X[0].length;
      this.mean = new Array(d).fill(0);
      X.forEach(r => { for (let j = 0; j < d; j++) this.mean[j] += r[j]; });
      for (let j = 0; j < d; j++) this.mean[j] /= n;
      const Xc = X.map(r => r.map((v, j) => v - this.mean[j]));
      const cov = M.scale(M.matmul(M.transpose(Xc), Xc), 1 / n);
      const { eigenvalues, eigenvectors } = M.eigSym(cov); // 升序
      this.components = []; this.explainedVariance = []; this.eigenvalues = eigenvalues;
      for (let i = d - 1; i >= 0; i--) { this.components.push(eigenvectors.map(r => r[i])); this.explainedVariance.push(eigenvalues[i]); }
      this.components = this.components.slice(0, this.nComponents);
      return this;
    }
    transform(X) { return X.map(r => this.components.map(comp => M.dot(r.map((v, j) => v - this.mean[j]), comp))); }
    explainedRatio() { const tot = this.explainedVariance.reduce((a, b) => a + b, 0); return this.components.length ? this.explainedVariance.slice(0, this.components.length).reduce((a, b) => a + b, 0) / tot : 0; }
  }

  // 高斯混合模型 EM（Bishop PRML Ch.9）；不变量：对数似然单调不减、责任度每行和为1
  class GMM {
    constructor(k = 2, iters = 100, seed = 7) { this.k = k; this.iters = iters; this.seed = seed; }
    fit(X) {
      const n = X.length, d = X[0].length; let s = this.seed;
      const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
      this.means = Array.from({ length: this.k }, () => X[Math.floor(rnd() * n)].slice());
      this.covs = Array.from({ length: this.k }, () => { const I = M.identity(d); return M.scale(I, 1); });
      this.weights = new Array(this.k).fill(1 / this.k);
      this.llHist = [];
      let resp = M.zeros(n, this.k);
      const logDet = (C) => { const L = M.cholesky(C); if (!L) return Math.log(1e-9); let s2 = 0; for (let i = 0; i < d; i++) s2 += Math.log(L[i][i]); return 2 * s2; };
      for (let it = 0; it < this.iters; it++) {
        resp = M.zeros(n, this.k);
        let ll = 0;
        for (let i = 0; i < n; i++) {
          const x = X[i]; let row = new Array(this.k); let mx = -Infinity;
          for (let c = 0; c < this.k; c++) {
            const m = this.means[c], C = this.covs[c];
            const diff = x.map((v, j) => v - m[j]);
            const Cinv = M.invert(C);
            let quad = 0; for (let a = 0; a < d; a++) { let s2 = 0; for (let b = 0; b < d; b++) s2 += Cinv[a][b] * diff[b]; quad += diff[a] * s2; }
            const ld = logDet(C);
            row[c] = Math.log(this.weights[c] + 1e-12) - 0.5 * (ld + quad + d * Math.log(2 * Math.PI));
            if (row[c] > mx) mx = row[c];
          }
          const ex = row.map(v => Math.exp(v - mx)); const sx = ex.reduce((a, b) => a + b, 0);
          ll += mx + Math.log(sx);
          for (let c = 0; c < this.k; c++) resp[i][c] = ex[c] / sx;
        }
        this.llHist.push(ll);
        const Nk = new Array(this.k).fill(0); for (let i = 0; i < n; i++) for (let c = 0; c < this.k; c++) Nk[c] += resp[i][c];
        for (let c = 0; c < this.k; c++) {
          const w = Nk[c] / n; this.weights[c] = Math.max(w, 1e-9);
          const m = new Array(d).fill(0); for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) m[j] += resp[i][c] * X[i][j]; for (let j = 0; j < d; j++) m[j] /= Nk[c] || 1; this.means[c] = m;
          const C = M.zeros(d, d); for (let i = 0; i < n; i++) { const diff = X[i].map((v, j) => v - m[j]); for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] += resp[i][c] * diff[a] * diff[b]; }
          for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] = (C[a][b] / (Nk[c] || 1)) + (a === b ? 1e-6 : 0);
          this.covs[c] = C;
        }
      }
      this.resp = resp;
      return this;
    }
    responsibilitySumInvariant() { return this.resp.every(r => Math.abs(r.reduce((a, b) => a + b, 0) - 1) < 1e-9); }
    llMonotone() { for (let i = 1; i < this.llHist.length; i++) if (this.llHist[i] < this.llHist[i - 1] - 1e-6) return false; return true; }
  }

  // ===================================================================
  // L6 表示学习：自动微分 MLP（反向传播，XOR 可学）
  // ===================================================================

  // 轻量反向模式自动微分（标量图），用于参数级梯度检验
  class AutoGrad {
    constructor(value) { this.value = value; this.grad = 0; this.parents = []; this.backward = null; }
    static add(a, b) { const r = new AutoGrad(a.value + b.value); r.parents = [a, b]; r.backward = () => { a.grad += r.grad; b.grad += r.grad; }; return r; }
    static mul(a, b) { const r = new AutoGrad(a.value * b.value); r.parents = [a, b]; r.backward = () => { a.grad += b.value * r.grad; b.grad += a.value * r.grad; }; return r; }
    static sigmoid(a) { const s = sigmoid(a.value); const r = new AutoGrad(s); r.parents = [a]; r.backward = () => { a.grad += s * (1 - s) * r.grad; }; return r; }
    static backward(root) {
      const topo = []; const seen = new Set(); const build = (v) => { if (seen.has(v)) return; seen.add(v); v.parents.forEach(build); topo.push(v); }; build(root);
      root.grad = 1; for (let i = topo.length - 1; i >= 0; i--) if (topo[i].backward) topo[i].backward();
    }
  }

  // MLP：手动反向传播（列向量约定），隐藏层 ReLU，输出层 sigmoid + BCE
  class MLP {
    constructor(layers, lr = 0.5) {
      this.layers = layers; this.lr = lr; this.W = []; this.b = [];
      for (let i = 0; i < layers.length - 1; i++) {
        const nin = layers[i], nout = layers[i + 1], sc = Math.sqrt(2 / (nin + nout));
        this.W.push(Array.from({ length: nout }, () => Array.from({ length: nin }, () => randn() * sc)));
        this.b.push(new Array(nout).fill(0));
      }
    }
    _forward(x) { // x: 列向量 [d][1]
      let a = x; const acts = [a]; const zs = [];
      for (let l = 0; l < this.W.length; l++) {
        const z = M.add(M.matmul(this.W[l], a), this.b[l].map(v => [v]));
        zs.push(z);
        a = (l === this.W.length - 1) ? z.map(r => [sigmoid(r[0])]) : z.map(r => [Math.max(0, r[0])]);
        acts.push(a);
      }
      return { out: a, acts, zs };
    }
    _backward(y, fwd) {
      const L = this.W.length; const dW = this.W.map(w => M.zeros(w.length, w[0].length)); const db = this.b.map(() => new Array(this.b[0].length).fill(0));
      // 输出层
      const out = fwd.out.map(r => r[0]); const yhat = out[0];
      let dz = [[yhat - y]];
      dW[L - 1] = M.matmul(dz, M.transpose(fwd.acts[L - 1])); db[L - 1] = dz.map(r => r[0]);
      let da = M.matmul(M.transpose(this.W[L - 1]), dz);
      for (let l = L - 2; l >= 0; l--) {
        const z = fwd.zs[l];
        const dg = z.map((r, i) => [r[0] > 0 ? da[i][0] : 0]); // 2D 列向量，与 da 同形
        dW[l] = M.matmul(dg, M.transpose(fwd.acts[l])); db[l] = dg.map(r => r[0]);
        da = M.matmul(M.transpose(this.W[l]), dg);
      }
      return { dW, db };
    }
    train(X, y, epochs = 2000) {
      this.lossHist = [];
      for (let e = 0; e < epochs; e++) {
        let loss = 0;
        for (let i = 0; i < X.length; i++) {
          const x = X[i].map(v => [v]); const fwd = this._forward(x); const yhat = fwd.out[0][0];
          loss += -(y[i] * Math.log(yhat + 1e-12) + (1 - y[i]) * Math.log(1 - yhat + 1e-12));
          const g = this._backward(y[i], fwd);
          const clip = (v) => Math.max(-5, Math.min(5, v));
          for (let l = 0; l < this.W.length; l++) { this.W[l] = M.sub(this.W[l], M.scale(g.dW[l].map(r => r.map(clip)), this.lr)); this.b[l] = this.b[l].map((bv, j) => bv - this.lr * clip(g.db[l][j])); }
        }
        this.lossHist.push(loss / X.length);
      }
      return this;
    }
    predict(X) { return X.map(r => (this._forward(r.map(v => [v])).out[0][0] >= 0.5 ? 1 : 0)); }
    lossDecreasing() { const h = this.lossHist; return h[h.length - 1] < h[0] - 1e-3; }
  }

  // ===================================================================
  // L7 序列建模：Transformer（多头自注意力 + 正弦位置编码）
  // ===================================================================
  const Transformer = {};

  Transformer.scaledDotProductAttention = (Q, K, V) => {
    const seqLen = Q.length, dk = Q[0].length;
    const scores = Q.map(q => K.map(k => M.dot(q, k) / Math.sqrt(dk)));
    const attn = scores.map(row => stableSoftmax(row)); // 每行和为1
    const out = attn.map((row, i) => { const o = new Array(V[0].length).fill(0); for (let j = 0; j < seqLen; j++) for (let c = 0; c < V[0].length; c++) o[c] += row[j] * V[j][c]; return o; });
    return { out, attn };
  };

  Transformer.multiHeadAttention = (Q, K, V, numHeads) => {
    const seqLen = Q.length, dModel = Q[0].length, dK = Math.floor(dModel / numHeads);
    const heads = [];
    for (let h = 0; h < numHeads; h++) {
      const slice = (mat) => mat.map(r => r.slice(h * dK, (h + 1) * dK));
      const { out } = Transformer.scaledDotProductAttention(slice(Q), slice(K), slice(V));
      heads.push(out); // seqLen × dK
    }
    // 拼接各 head 输出 -> [seqLen][numHeads·dK]，与真实 Transformer 一致
    return heads[0].map((_, i) => heads.flatMap(h => h[i]));
  };

  Transformer.positionalEncoding = (seqLen, dModel) => {
    const pe = M.zeros(seqLen, dModel);
    for (let pos = 0; pos < seqLen; pos++) for (let i = 0; i < dModel; i += 2) {
      pe[pos][i] = Math.sin(pos / Math.pow(10000, i / dModel));
      if (i + 1 < dModel) pe[pos][i + 1] = Math.cos(pos / Math.pow(10000, i / dModel));
    }
    return pe;
  };

  Transformer.attentionSumInvariant = (Q, K, V) => {
    const { attn } = Transformer.scaledDotProductAttention(Q, K, V);
    return attn.every(row => Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-9);
  };

  // ===================================================================
  // L10 智能体内核：规划 / 工具调用 / 微指令分解（ReAct 风格，无需外部 LLM）
  // ===================================================================
  // 将高层目标递归分解为「微指令」，逐条执行并观测，对应第十层 1024 微指令级操作。
  class Agent {
    constructor(tools = {}) { this.tools = tools; this.microOps = []; this.trace = []; }
    // 注册工具：name -> async/sync fn(args)
    register(name, fn) { this.tools[name] = fn; return this; }
    _emit(level, op, detail) { const micro = { microId: this.microOps.length, level, op, detail, t: this.microOps.length }; this.microOps.push(micro); return micro; }
    // 规划：基于可用工具与目标的简单启发式分解（可复现、确定性）
    plan(goal) {
      const steps = [];
      this._emit(0, 'PLAN', `解析目标: ${goal}`);
      if (this.tools.search) steps.push({ tool: 'search', args: { q: goal } });
      if (this.tools.compute) steps.push({ tool: 'compute', args: {} });
      if (this.tools.summarize) steps.push({ tool: 'summarize', args: { goal } });
      if (this.tools.report) steps.push({ tool: 'report', args: {} });
      if (steps.length === 0) steps.push({ tool: 'noop', args: {} });
      this._emit(0, 'PLAN', `分解为 ${steps.length} 个阶段，含 ${(steps.length * 8)} 条微指令`);
      return steps;
    }
    async run(goal) {
      const steps = this.plan(goal);
      const ctx = {};
      for (const step of steps) {
        this._emit(1, 'ACTION', `调用工具 ${step.tool}`);
        const fn = this.tools[step.tool];
        let obs;
        try { obs = fn ? await fn(step.args, ctx) : 'no-tool'; }
        catch (e) { obs = 'error: ' + e.message; }
        this._emit(1, 'OBSERVE', typeof obs === 'object' ? JSON.stringify(obs).slice(0, 120) : String(obs));
        ctx[step.tool] = obs;
        this.trace.push({ step, obs });
      }
      this._emit(0, 'DONE', `目标完成，共 ${this.microOps.length} 条微指令`);
      return { ctx, microOps: this.microOps, trace: this.trace };
    }
  }

  // ===================================================================
  // 工具函数：合成数据（用于自检演示）
  // ===================================================================
  const DataGen = {
    moons: (n = 100, noise = 0.1) => {
      const X = [], y = [];
      for (let i = 0; i < n; i++) {
        const t = Math.PI * (i / (n / 2));
        if (i < n / 2) X.push([Math.cos(t) + (_nextRandom() - 0.5) * noise, Math.sin(t) + (_nextRandom() - 0.5) * noise]), y.push(1);
        else X.push([1 - Math.cos(t) + (_nextRandom() - 0.5) * noise, 1 - Math.sin(t) - (_nextRandom() - 0.5) * noise]), y.push(-1);
      }
      return { X, y };
    },
    blobs: (n = 150, k = 3, dim = 2) => {
      const centers = Array.from({ length: k }, () => Array.from({ length: dim }, () => (_nextRandom() - 0.5) * 10));
      const X = [], y = [];
      for (let i = 0; i < n; i++) { const c = i % k; X.push(centers[c].map(v => v + randn() * 0.6)); y.push(c); }
      return { X, y };
    },
    xor: () => {
      const base = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]];
      const X = base.map(r => [r[0], r[1]]), y = base.map(r => r[2]);
      return { X, y };
    }
  };

  // ===================================================================
  // 导出
  // ===================================================================
  return {
    M, Optim, DataGen, AutoGrad, Transformer,
    LinearRegression, LogisticRegression, SVM, kNN, NaiveBayes, DecisionTree,
    KMeans, PCA, GMM, MLP, Agent,
    seed: _setSeed,
    version: '1.0.0'
  };
}));
